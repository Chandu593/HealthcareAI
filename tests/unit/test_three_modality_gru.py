"""Stage 3a -- first-24-hour GRU inputs (gru_inputs.py), the generic pooled focal multitask trainer (multitask_training.py) and the GRU encoder stage (gru_encoder.py).
Torch-free tests cover the inputs; the training/encoder tests are skipped without torch."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _tcd_synth import make_dataset, make_manifest  # noqa: E402

from src.three_modality import artifacts  # noqa: E402
from src.three_modality import gru_inputs as gi  # noqa: E402
from src.three_modality import tcd_features as tf  # noqa: E402
from src.three_modality.config import THREE_MODALITY_DEFAULTS, validate_gru_cfg, validate_tcd_cfg  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False


def tcd_cfg():
    c = copy.deepcopy(THREE_MODALITY_DEFAULTS["tcd"]); c["variables"]["chunksize"] = 4000; c["audit"]["n_patients"] = 0
    return validate_tcd_cfg(c, 24)


def gru_cfg(**over):
    c = copy.deepcopy(THREE_MODALITY_DEFAULTS["gru"]); c["audit"]["n_patients"] = 40
    c["encoder"].update({"hidden_dim": 8, "num_layers": 1}); c["training"].update({"max_epochs": 8, "patience": 3, "batch_size": 64, "lr": 0.003})
    for k, v in over.items():
        a = c
        parts = k.split(".")
        for p in parts[:-1]:
            a = a[p]
        a[parts[-1]] = v
    return validate_gru_cfg(c, 1440)


def ser(**kw):
    return {k: (np.array(p, np.int64), np.array(t, float), np.array(v, float)) for k, (p, t, v) in kw.items()}


class TestBinning(unittest.TestCase):
    def bins(self, times, values=None, we=1440.0):
        t = np.array(times, float); v = np.arange(1, len(t) + 1, dtype=float) if values is None else np.array(values, float)
        vals, obs = gi.bin_series(ser(x=([0] * len(t), t, v)), np.array([we]), 1440, 60, 24, ["x"])
        return vals[0, :, 0], obs[0, :, 0]

    def test_bin_boundaries_offset_zero_hour_edges_and_the_last_minute(self):
        for t, b in [(0, 0), (59, 0), (60, 0), (61, 1), (120, 1), (121, 2), (1380, 22), (1381, 23), (1440, 23)]:
            vals, obs = self.bins([t])
            self.assertEqual(np.flatnonzero(obs).tolist(), [b], f"offset {t} -> bin {b}")
        _, obs = self.bins([1441])
        self.assertFalse(obs.any())                                                                      # minute 1441 is outside the window

    def test_the_last_value_in_an_hour_bin_is_kept(self):
        vals, obs = self.bins([10, 30, 55, 70], [1.0, 2.0, 3.0, 9.0])
        self.assertEqual((vals[0], vals[1]), (3.0, 9.0)); self.assertEqual(obs.sum(), 2)

    def test_unobserved_hours_are_zero_with_mask_zero_never_carried_forward(self):
        vals, obs = self.bins([200], [5.0])                                                              # one measurement in hour 4
        self.assertEqual(obs.tolist(), [False] * 3 + [True] + [False] * 20)
        self.assertTrue((vals[~obs] == 0).all())                                                         # no last-observation-carried-forward, no fabricated value

    def test_rows_after_the_window_or_before_icu_cannot_change_anything(self):
        clean, oc = self.bins([100, 700, 1440], [1.0, 2.0, 3.0])
        dirty, od = self.bins([100, 700, 1440, 1441, 2000, 5000, -30, -500], [1.0, 2.0, 3.0, 99.0, -7.0, 1e6, 55.0, 1e3])
        np.testing.assert_array_equal(clean, dirty); np.testing.assert_array_equal(oc, od)

    def test_a_patient_discharged_early_has_unobserved_bins_after_exit_and_a_fixed_length(self):
        vals, obs = self.bins([100, 500, 700, 1000], [1.0, 2.0, 50.0, 60.0], we=600.0)                   # ICU exit at minute 600
        self.assertEqual(vals.shape, (24,))                                                              # length is fixed at 24 whatever the stay
        self.assertEqual(np.flatnonzero(obs).tolist(), [1, 8])                                           # rows at 700/1000 are after exit -> dropped
        self.assertFalse(obs[10:].any())

    def test_multiple_patients_and_variables_are_independent(self):
        s = ser(a=([0, 1, 1], [30, 30, 90], [1.0, 2.0, 3.0]), b=([1], [30], [7.0]))
        vals, obs = gi.bin_series(s, np.full(2, 1440.0), 1440, 60, 24, ["a", "b", "missing_var"])
        self.assertEqual((vals[0, 0, 0], vals[1, 0, 0], vals[1, 1, 0], vals[1, 0, 1]), (1.0, 2.0, 3.0, 7.0))
        self.assertFalse(obs[:, :, 2].any()); self.assertFalse(obs[0, :, 1].any())                       # a variable absent for a patient / globally stays unobserved


class TestScaling(unittest.TestCase):
    def test_scaling_is_fitted_on_training_cells_only_and_standardises_them(self):
        rs = np.random.default_rng(0); n = 300
        split = np.array(["train"] * 200 + ["val"] * 50 + ["test"] * 50)
        vals = rs.normal(5, 2, (n, 24, 2)).astype(np.float32); obs = rs.random((n, 24, 2)) < 0.3
        sc = gi.fit_scaling(vals, obs, split == "train", ["a", "b"], [0.5, 99.5])
        bad = vals.copy(); bad[split != "train"] = 1e9
        sc2 = gi.fit_scaling(bad, obs, split == "train", ["a", "b"], [0.5, 99.5])
        self.assertEqual(sc, sc2)                                                                         # validation/test cells cannot influence the fit
        z = gi.apply_scaling(vals, obs, sc, ["a", "b"])
        tr_obs = obs[:200][:, :, 0]
        self.assertAlmostEqual(float(z[:200][:, :, 0][tr_obs].mean()), 0.0, places=1); self.assertAlmostEqual(float(z[:200][:, :, 0][tr_obs].std()), 1.0, places=1)
        self.assertTrue((z[~obs] == 0).all())                                                             # unobserved cells stay exactly 0

    def test_outliers_are_winsorised_and_constant_or_never_observed_variables_are_handled(self):
        vals = np.zeros((100, 24, 3), np.float32); obs = np.zeros((100, 24, 3), bool)
        vals[:, 0, 0] = np.linspace(0, 10, 100); vals[0, 0, 0] = 1e6; obs[:, 0, 0] = True      # outlier
        vals[:, 0, 1] = 4.0; obs[:, 0, 1] = True                                                 # constant
        sc = gi.fit_scaling(vals, obs, np.ones(100, bool), ["o", "c", "n"], [0.5, 99.5])         # 'n' never observed
        z = gi.apply_scaling(vals, obs, sc, ["o", "c", "n"])
        self.assertLess(float(np.abs(z[:, 0, 0]).max()), 10)                                      # the 1e6 did not propagate
        self.assertTrue(sc["variables"]["c"]["degenerate"]); self.assertTrue((z[:, 0, 1] == 0).all())
        self.assertEqual(sc["variables"]["n"]["n_observed_train_cells"], 0); self.assertTrue(np.isfinite(z).all())


class TestConfig(unittest.TestCase):
    def test_the_grid_must_cover_exactly_the_window(self):
        with self.assertRaises(ValueError):
            gru_cfg(n_bins=48)
        with self.assertRaises(ValueError):
            gru_cfg(bin_minutes=30)
        self.assertEqual(gru_cfg(bin_minutes=120, n_bins=12)["n_bins"], 12)

    def test_focal_parameters_are_validated_per_task(self):
        c = copy.deepcopy(THREE_MODALITY_DEFAULTS["gru"]); c["training"]["focal"]["sepsis"]["alpha"] = 1.5
        with self.assertRaises(ValueError):
            validate_gru_cfg(c, 1440)
        c = copy.deepcopy(THREE_MODALITY_DEFAULTS["gru"]); c["training"]["task_weights"] = {"mortality": 0.0, "sepsis": 0.0}
        with self.assertRaises(ValueError):
            validate_gru_cfg(c, 1440)


class TestInputsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.pats, cls.lab = make_dataset(cls.root / "data", n_pat=220, seed=2, post_exit_poison=True)
        cls.man = make_manifest(cls.pats)
        cls.tcfg = tcd_cfg(); cls.gcfg = gru_cfg()
        cls.src = {"data_root": str(cls.root / "data")}
        cls.feat = tf.build_tcd_features(cls.src, cls.pats, cls.man, cls.tcfg, cls.root / "f")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, out, gcfg=None, feat=None, **kw):
        return gi.build_gru_inputs(self.src, (feat or self.feat)["run_dir"], gcfg or self.gcfg, self.tcfg, self.root / out, **kw)

    def test_artifacts_ordering_shapes_and_the_audit(self):
        r = self.build("g1", force=True); d = Path(r["run_dir"])
        for n in gi.INPUT_FILES + ["manifest.json"]:
            self.assertTrue((d / n).is_file(), n)
        x = gi.load_gru_inputs(d); t = tf.load_tcd_features(self.feat["run_dir"])
        pd.testing.assert_frame_equal(x["patients"], t["patients"])                                       # one patient ordering for every modality
        self.assertEqual(x["variables"], t["schema"]["selected_variables"])
        self.assertEqual(tuple(x["values"].shape), (len(self.pats), 24, len(x["variables"]))); self.assertEqual(x["values"].shape, x["mask"].shape)
        self.assertEqual(x["values"].dtype, np.float32); self.assertTrue(np.isfinite(np.asarray(x["values"])).all()); self.assertEqual(set(np.unique(x["mask"])), {0, 1})
        rep = json.loads((d / "input_report.json").read_text())
        self.assertFalse(rep["sequence_length_derived_from_icu_exit"]); self.assertEqual(rep["fixed_sequence_length"], 24)
        a = rep["cutoff_audit"]; self.assertTrue(a["passed"]); self.assertGreater(a["rows_after_window_present_and_corrupted"], 0)
        self.assertTrue(r["manifest"]["cutoff_audit_passed"])
        self.assertTrue((np.asarray(x["values"])[np.asarray(x["mask"]) == 0] == 0).all())

    def test_rows_after_icu_exit_never_enter_the_sequences(self):
        r = self.build("g2", force=True); x = gi.load_gru_inputs(r["run_dir"])
        we = x["patients"]["window_end_minutes"].to_numpy(); m = np.asarray(x["mask"])
        for p in np.flatnonzero(we < 1440)[:60]:
            first_unobservable_bin = int(np.ceil(we[p] / 60.0))                                           # bins strictly after the exit hour
            self.assertFalse(m[p, first_unobservable_bin:, :].any(), f"patient {p}")
        j = x["variables"].index("creatinine"); self.assertLess(np.asarray(x["values"])[:, :, j].max(), 50)   # the planted 999 (after exit) never appears

    def test_training_only_scaling_through_the_pipeline(self):
        sp = {int(k): v for k, v in self.man.patient_to_split.items()}
        lab = self.lab.copy(); m = lab["patientunitstayid"].map(lambda i: sp[int(i)] != "train")
        lab.loc[m, "labresult"] = pd.to_numeric(lab.loc[m, "labresult"], errors="coerce") * 1000 + 777
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "B").mkdir(); self.pats.to_csv(root / "B" / "patient.csv", index=False); lab.to_csv(root / "B" / "lab.csv", index=False)
            srcB = {"data_root": str(root / "B")}
            fB = tf.build_tcd_features(srcB, self.pats, self.man, self.tcfg, root / "fB")
            rA = self.build("g3a", force=True)
            rB = gi.build_gru_inputs(srcB, fB["run_dir"], self.gcfg, self.tcfg, root / "gB")
            xa, xb = gi.load_gru_inputs(rA["run_dir"]), gi.load_gru_inputs(rB["run_dir"])
            self.assertEqual(xa["scaling"], xb["scaling"])                                                # fitted on TRAIN only
            tr = (xa["patients"]["split"] == "train").to_numpy()
            np.testing.assert_array_equal(np.asarray(xa["values"])[tr], np.asarray(xb["values"])[tr])
            self.assertFalse(np.array_equal(np.asarray(xa["values"])[~tr], np.asarray(xb["values"])[~tr]))   # (sanity) the corruption reached val/test

    def test_cache_reuse_force_tamper_and_identity(self):
        r1 = self.build("g4"); self.assertFalse(r1["cache_hit"]); d1 = Path(r1["run_dir"]); h1 = artifacts.sha256_file(d1 / "gru_values.npy")
        r2 = self.build("g4"); self.assertTrue(r2["cache_hit"]); self.assertEqual(r2["run_id"], r1["run_id"])
        r3 = self.build("g4", force=True, run_id="forced"); self.assertFalse(r3["cache_hit"])
        self.assertEqual(artifacts.sha256_file(Path(r3["run_dir"]) / "gru_values.npy"), h1)               # deterministic
        with self.assertRaises(FileExistsError):
            self.build("g4", force=True, run_id="forced")
        r4 = self.build("g4", gcfg=gru_cfg(**{"scaling.winsor_pct": [1.0, 99.0]})); self.assertFalse(r4["cache_hit"])
        self.assertTrue(self.build("g4", gcfg=gru_cfg(**{"encoder.hidden_dim": 4}))["cache_hit"])                      # encoder/training settings do not invalidate inputs
        b = bytearray((d1 / "gru_mask.npy").read_bytes()); b[len(b) // 2] ^= 1; (d1 / "gru_mask.npy").write_bytes(bytes(b))
        r5 = self.build("g4", force=False)                                                                # r1 is tampered, but the intact forced run r3 is still a valid identical run
        self.assertTrue(r5["cache_hit"]); self.assertEqual(r5["run_id"], "forced")
        pf = Path(r3["run_dir"]) / "gru_mask.npy"; b = bytearray(pf.read_bytes()); b[len(b) // 2] ^= 1; pf.write_bytes(bytes(b))
        self.assertFalse(self.build("g4")["cache_hit"])                                                   # every identical run tampered -> recompute

    def test_a_changed_tcd_feature_run_means_new_inputs(self):
        r1 = self.build("g5", force=True)
        feat2 = tf.build_tcd_features(self.src, self.pats, self.man, {**self.tcfg, "half_split_minutes": 600}, self.root / "f2")
        r2 = self.build("g5", feat=feat2); self.assertFalse(r2["cache_hit"])

    def test_the_audit_fails_if_the_clipping_is_removed(self):
        orig = gi.clip_series
        gi.clip_series = lambda series, we, w: series
        try:
            with self.assertRaises(AssertionError):
                self.build("g6", force=True)
        finally:
            gi.clip_series = orig


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestTrainerAndModel(unittest.TestCase):
    def _task(self, n=500, v=4, seed=0):
        rs = np.random.default_rng(seed)
        values = rs.normal(size=(n, 24, v)).astype(np.float32); mask = (rs.random((n, 24, v)) < 0.4).astype(np.float32)
        sig = (values[:, :, 0] * mask[:, :, 0]).sum(axis=1)
        mort = (sig + 0.5 * rs.normal(size=n) > 0.8).astype(float)
        sep = (sig + 0.5 * rs.normal(size=n) > 0.8).astype(float); sep[rs.random(n) < 0.4] = np.nan
        split = np.array(["train"] * 350 + ["val"] * 75 + ["test"] * 75)
        return values, mask, mort, sep, split

    def _run(self, task, cfg_over=None, epochs=6, **kw):
        import torch
        from src.three_modality.gru_encoder import make_gru_model
        from src.three_modality.multitask_training import train_multitask
        values, mask, mort, sep, split = task
        cfg = copy.deepcopy(gru_cfg()["training"]); cfg.update({"max_epochs": epochs, "patience": 99, "weight_decay": 0.0}); cfg.update(cfg_over or {})
        factory = lambda: make_gru_model(values.shape[2], gru_cfg()["encoder"])   # noqa: E731  (the trainer builds it AFTER seeding)
        bf = lambda idx: (torch.from_numpy(values[idx]), torch.from_numpy(mask[idx]))   # noqa: E731
        return train_multitask(factory, bf, mort, sep, split, cfg, **kw), bf, cfg

    def test_model_shapes_gradient_flow_and_patient_independence(self):
        import torch
        from src.three_modality.gru_encoder import make_gru_model
        v, m, mort, sep, split = self._task(n=40)
        model = make_gru_model(4, {"hidden_dim": 8, "num_layers": 2, "dropout": 0.0})
        e, ml, sl, aux = model(torch.from_numpy(v[:10]), torch.from_numpy(m[:10]))
        self.assertEqual((tuple(e.shape), tuple(ml.shape), tuple(sl.shape), aux), ((10, 8), (10,), (10,), None))
        (ml.mean() + sl.mean()).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name); self.assertGreater(float(p.grad.abs().sum()), 0, name)    # gradients reach the GRU AND both heads
        model.eval()
        with torch.no_grad():
            a = model(torch.from_numpy(v[:5]), torch.from_numpy(m[:5]))[0]
            b = model(torch.from_numpy(np.concatenate([v[:5], v[20:30]])), torch.from_numpy(np.concatenate([m[:5], m[20:30]])))[0][:5]
        torch.testing.assert_close(a, b)                                                                  # no interaction between patients in a batch
        with torch.no_grad():
            z = model(torch.zeros(3, 24, 4), torch.zeros(3, 24, 4))[0]                                    # a patient with NO observation at all
        self.assertTrue(torch.isfinite(z).all())

    def test_the_encoder_always_sees_the_fixed_full_length_never_one_derived_from_the_data(self):
        import torch
        from src.three_modality.gru_encoder import make_gru_model
        v, m, *_ = self._task(n=30)
        m[:15, 8:, :] = 0; v[:15, 8:, :] = 0                                                              # half the patients look like short stays: nothing observed after hour 8
        model = make_gru_model(4, {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}); model.eval()
        with torch.no_grad():
            got = model(torch.from_numpy(v), torch.from_numpy(m))[0]
            ref = model.encoder(torch.cat([torch.from_numpy(v), torch.from_numpy(m)], -1), torch.full((30,), 24, dtype=torch.long))
        torch.testing.assert_close(got, ref)                                                              # == the encoder run over ALL 24 steps for every patient

    def test_training_learns_the_planted_signal_and_is_deterministic(self):
        task = self._task()
        over = {"lr": 0.01}                                                                                # (measured: 0.92 / 0.95 val AUROC at 30 epochs; 0.003 x 10 epochs under-trains)
        (m1, h1), bf, cfg = self._run(task, over, epochs=30)
        self.assertLess(h1["epochs"][-1]["train_loss"], h1["epochs"][0]["train_loss"])
        self.assertGreater(max(e["val_auroc_mortality"] for e in h1["epochs"]), 0.85); self.assertGreater(max(e["val_auroc_sepsis"] for e in h1["epochs"]), 0.85)
        (m2, h2), _, _ = self._run(task, over, epochs=30)
        from src.three_modality.multitask_training import encode_all
        np.testing.assert_array_equal(encode_all(m1, bf, 500), encode_all(m2, bf, 500))
        self.assertEqual([e["train_loss"] for e in h1["epochs"]], [e["train_loss"] for e in h2["epochs"]])

    def test_results_do_not_depend_on_the_global_rng_state_at_call_time(self):
        """Regression: the model used to be built BEFORE the trainer seeded, so two fresh processes started from different initial weights."""
        import torch
        from src.three_modality.multitask_training import encode_all
        task = self._task(seed=7)
        outs = []
        for state in (123, 99999):
            torch.manual_seed(state); np.random.seed(state)                                              # simulate a different process / history
            (m, h), bf, _ = self._run(task, epochs=3)
            outs.append((encode_all(m, bf, 500), [e["train_loss"] for e in h["epochs"]]))
        np.testing.assert_array_equal(outs[0][0], outs[1][0]); self.assertEqual(outs[0][1], outs[1][1])

    def test_focal_parameters_and_weights_are_separate_per_task(self):
        task = self._task(seed=1)
        def first_loss(over):
            (_, h), _, _ = self._run(task, over, epochs=1); return h["epochs"][0]["train_loss"]
        base = copy.deepcopy(gru_cfg()["training"]); F = base["focal"]
        loss0 = first_loss({})
        for task_name in ("mortality", "sepsis"):
            self.assertNotEqual(loss0, first_loss({"focal": {**F, task_name: {**F[task_name], "gamma": 4.0}}}), task_name)      # each task's gamma drives the loss ...
            self.assertNotEqual(loss0, first_loss({"focal": {**F, task_name: {**F[task_name], "alpha": 0.7}}}), task_name)      # ... and each task's alpha
        # a task with weight 0 ignores ITS focal parameters entirely, but not the other task's
        w = {"task_weights": {"mortality": 1.0, "sepsis": 0.0}}
        self.assertEqual(first_loss(w), first_loss({**w, "focal": {**F, "sepsis": {"alpha": 0.9, "gamma": 5.0}}}))
        self.assertNotEqual(first_loss(w), first_loss({**w, "focal": {**F, "mortality": {"alpha": 0.9, "gamma": 5.0}}}))

    def test_a_zero_weight_task_head_receives_no_gradient(self):
        import torch
        from src.three_modality.gru_encoder import make_gru_model
        task = self._task(seed=2)
        torch.manual_seed(42); init = make_gru_model(4, gru_cfg()["encoder"]).sepsis_head.weight.detach().clone()
        (m, _), _, _ = self._run(task, {"task_weights": {"mortality": 1.0, "sepsis": 0.0}}, epochs=3)
        torch.testing.assert_close(m.sepsis_head.weight.detach(), init)

    def test_label_masks_are_per_task(self):
        v, m, mort, sep, split = self._task(seed=3)
        (_, h), _, _ = self._run((v, m, mort, np.full(len(sep), np.nan), split), epochs=3)               # sepsis has no labels at all: mortality still trains
        self.assertLess(h["epochs"][-1]["train_loss"], h["epochs"][0]["train_loss"]); self.assertIsNone(h["epochs"][0]["val_auroc_sepsis"])
        (_, h2), _, _ = self._run((v, m, np.full(len(mort), np.nan), sep, split), epochs=3)
        self.assertIsNone(h2["epochs"][0]["val_auroc_mortality"]); self.assertIsNotNone(h2["epochs"][0]["val_auroc_sepsis"])

    def test_validation_labels_select_the_model_but_never_train_it_and_best_weights_are_restored(self):
        import torch
        from src.three_modality.multitask_training import focal_multitask_loss
        v, m, mort, sep, split = self._task(seed=4)
        (_, a), _, _ = self._run((v, m, mort, sep, split), epochs=6)
        mort2 = mort.copy(); mort2[split == "val"] = 1 - mort2[split == "val"]
        (_, b), _, _ = self._run((v, m, mort2, sep, split), epochs=6)
        self.assertEqual(a["epochs"][0]["train_loss"], b["epochs"][0]["train_loss"]); self.assertNotEqual(a["epochs"][0]["val_loss"], b["epochs"][0]["val_loss"])
        # best-epoch restore: overfit-prone setup so the last epoch is NOT the best
        rs = np.random.default_rng(9); vv = rs.normal(size=(260, 24, 6)).astype(np.float32); mm = (rs.random((260, 24, 6)) < .5).astype(np.float32)
        mo = (rs.random(260) < .4).astype(float); se = np.full(260, np.nan); sp = np.array(["train"] * 90 + ["val"] * 170)
        (model, h), bf, cfg = self._run((vv, mm, mo, se, sp), {"lr": 0.02, "patience": 3}, epochs=40)
        self.assertLess(h["best_epoch"], len(h["epochs"]))
        model.eval()
        with torch.no_grad():
            _, ml, sl, _ = model(*bf(np.flatnonzero(sp == "val")))
            loss, _ = focal_multitask_loss(ml, sl, torch.from_numpy(mo[sp == "val"].astype(np.float32)), torch.from_numpy(se[sp == "val"].astype(np.float32)), cfg["focal"], cfg["task_weights"])
        self.assertAlmostEqual(float(loss), h["best_score"], places=5)

    def test_the_auxiliary_loss_hook_is_applied_only_with_a_positive_weight(self):
        import torch
        calls = []
        def aux(a, idx):
            calls.append(len(idx)); return torch.tensor(0.5, requires_grad=True)
        task = self._task(seed=5)
        self._run(task, epochs=1, aux_loss_fn=aux, aux_weight=0.0); self.assertEqual(calls, [])
        (_, h), _, _ = self._run(task, epochs=1, aux_loss_fn=aux, aux_weight=0.3)
        self.assertTrue(calls); self.assertEqual(h["aux_weight"], 0.3); self.assertIn("val_aux", h["epochs"][0])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestGRUEncoderStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.three_modality import gru_encoder as ge
        cls.ge = ge
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.pats, _ = make_dataset(cls.root / "data", n_pat=220, seed=6)
        cls.man = make_manifest(cls.pats); cls.tcfg = tcd_cfg(); cls.gcfg = gru_cfg()
        cls.src = {"data_root": str(cls.root / "data")}
        feat = tf.build_tcd_features(cls.src, cls.pats, cls.man, cls.tcfg, cls.root / "f")
        cls.inp = gi.build_gru_inputs(cls.src, feat["run_dir"], cls.gcfg, cls.tcfg, cls.root / "g")
        rs = np.random.default_rng(1); inc = pd.DataFrame({"patientunitstayid": cls.pats.patientunitstayid})
        inc["included"] = rs.random(len(inc)) < 0.6; inc["label"] = np.where(inc["included"], (rs.random(len(inc)) < 0.25).astype(float), np.nan)
        cls.inc_dir = cls.root / "inc"; cls.inc_dir.mkdir(); inc.to_csv(cls.inc_dir / "incident_sepsis_labels.csv", index=False)
        (cls.inc_dir / "manifest.json").write_text(json.dumps({"identity": "inc-A", "run_id": "incA", "files": {}}))
        cls.inc = inc

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, out, pats=None, gcfg=None, inc_dir=None, **kw):
        return self.ge.build_gru_encoder(self.inp["run_dir"], inc_dir or self.inc_dir, self.pats if pats is None else pats, gcfg or self.gcfg, self.tcfg, self.root / out, **kw)

    def _emb(self, out, **kw):
        return np.load(Path(self.build(out, force=True, **kw)["run_dir"]) / "gru_embeddings.npz")["embeddings"]

    def _flip(self, ids):
        out = self.pats.copy(); m = out["patientunitstayid"].isin(ids)
        out.loc[m, "hospitaldischargestatus"] = out.loc[m, "hospitaldischargestatus"].map({"Alive": "Expired", "Expired": "Alive"})
        return out

    def test_artifacts_and_fusion_compatible_embedding_file(self):
        from src.fusion.multimodal_fusion import align_embeddings_to_cohort
        r = self.build("e1", force=True); d = Path(r["run_dir"])
        for n in ("gru_embeddings.npz", "gru_encoder.pt", "training_log.json", "label_counts.json", "config_snapshot.json", "manifest.json"):
            self.assertTrue((d / n).is_file(), n)
        z = np.load(d / "gru_embeddings.npz")
        self.assertEqual(set(z.files), {"patient_ids", "embeddings"}); self.assertEqual(z["embeddings"].shape, (len(self.pats), 8)); self.assertEqual(z["embeddings"].dtype, np.float32)
        self.assertTrue(np.isfinite(z["embeddings"]).all()); self.assertEqual(z["patient_ids"].tolist(), sorted(self.pats.patientunitstayid.tolist()))
        np.testing.assert_array_equal(align_embeddings_to_cohort(z["patient_ids"].tolist(), z["embeddings"], z["patient_ids"][::-1].tolist()), z["embeddings"][::-1])
        m = json.loads((d / "manifest.json").read_text())
        self.assertEqual((m["embedding_dim"], m["inputs_identity"]), (8, self.inp["manifest"]["identity"]))
        ck = torch.load(d / "gru_encoder.pt", weights_only=False); self.assertEqual(ck["encoder_config"]["hidden_dim"], 8)

    def test_test_labels_are_unreachable_and_landmark_short_stay_labels_are_ignored(self):
        sp = dict(zip(self.inp_patients()["patientunitstayid"], self.inp_patients()["split"]))
        short = set(self.pats.loc[self.pats["unitdischargeoffset"] < 1440, "patientunitstayid"])
        test_ids = {i for i, s in sp.items() if s == "test"}; tr_short = {i for i in short if sp[i] == "train"}; tr_long = {i for i, s in sp.items() if s == "train" and i not in short}
        base = self._emb("e_base")
        np.testing.assert_array_equal(base, self._emb("e_t", pats=self._flip(test_ids)))                  # TEST labels: unreachable
        np.testing.assert_array_equal(base, self._emb("e_s", pats=self._flip(tr_short)))                  # short-stay TRAIN labels: unreachable (landmark_only)
        self.assertFalse(np.array_equal(base, self._emb("e_l", pats=self._flip(tr_long))))                # (sanity) landmark train labels matter

    def inp_patients(self):
        return pd.read_csv(Path(self.inp["run_dir"]) / "patients.csv")

    def test_population_cache_and_identity(self):
        r1 = self.build("e2"); self.assertFalse(r1["cache_hit"])
        pop = r1["manifest"]["training_population"]; self.assertTrue(pop["landmark_only"]); self.assertLess(pop["landmark_patients_by_split"]["train"], pop["patients_by_split"]["train"])
        self.assertTrue(self.build("e2")["cache_hit"])
        F = copy.deepcopy(self.gcfg["training"]["focal"]); F["sepsis"]["gamma"] = 3.0
        self.assertFalse(self.build("e2", gcfg=gru_cfg(**{"training.focal": F}))["cache_hit"])            # a per-task focal parameter is part of the identity
        self.assertFalse(self.build("e2", gcfg=gru_cfg(**{"training.landmark_only": False}))["cache_hit"])
        with self.assertRaises(FileExistsError):
            self.build("e2", force=True, run_id=r1["run_id"])
        h = json.loads((Path(r1["run_dir"]) / "training_log.json").read_text())
        for k in ("hyperparameters", "seed", "best_epoch", "epochs", "n_train", "n_val", "monitor", "training_population", "encoder"):
            self.assertIn(k, h)


if __name__ == "__main__":
    unittest.main()
