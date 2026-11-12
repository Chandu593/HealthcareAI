"""TCD encoder (src/three_modality/tcd_encoder.py): label alignment, focal multitask training, train/val/test discipline, determinism, caching, fusion compatibility.
Training tests need torch and are skipped without it (the label tests are torch-free)."""

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

from src.fusion.multimodal_fusion import align_embeddings_to_cohort  # noqa: E402
from src.three_modality import tcd_encoder as te  # noqa: E402
from src.three_modality import tcd_features as tf  # noqa: E402
from src.three_modality.config import THREE_MODALITY_DEFAULTS, validate_tcd_cfg  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False


def tcfg(**over):
    c = copy.deepcopy(THREE_MODALITY_DEFAULTS["tcd"])
    c["variables"]["chunksize"] = 4000
    c["audit"]["n_patients"] = 0
    c["encoder"].update({"hidden_dim": 24, "embedding_dim": 8})
    c["training"].update({"max_epochs": 12, "patience": 4, "batch_size": 64, "lr": 0.003})
    for k, v in over.items():
        a = c
        parts = k.split(".")
        for p in parts[:-1]:
            a = a[p]
        a[parts[-1]] = v
    return validate_tcd_cfg(c, 24)


class TestLandmark(unittest.TestCase):
    def test_landmark_membership_follows_the_strict_landmark_rule(self):
        pats = pd.DataFrame({"patientunitstayid": [1, 2, 3, 4, 5, 6]})
        table = pd.DataFrame({"patientunitstayid": [6, 5, 4, 3, 2, 1], "unitdischargeoffset": [np.nan, 5000, 1440, 1439, -10, 1441]})     # shuffled on purpose
        np.testing.assert_array_equal(te.landmark_mask(pats, table, 1440), [True, False, False, True, True, False])     # 1: 1441 yes, 2: negative no, 3: 1439 no, 4: exactly 1440 yes, 5: 5000 yes, 6: NaN no
        with self.assertRaises(ValueError):
            te.landmark_mask(pats, table.drop(columns="unitdischargeoffset"), 1440)

    def test_landmark_only_is_the_default_and_must_be_boolean(self):
        self.assertIs(validate_tcd_cfg(copy.deepcopy(THREE_MODALITY_DEFAULTS["tcd"]), 24)["training"]["landmark_only"], True)
        bad = copy.deepcopy(THREE_MODALITY_DEFAULTS["tcd"]); bad["training"]["landmark_only"] = "yes"
        with self.assertRaises(ValueError):
            validate_tcd_cfg(bad, 24)


class TestLabels(unittest.TestCase):
    def test_alignment_is_by_patient_id_not_row_order_and_sepsis_is_nan_outside_the_incident_cohort(self):
        pats = pd.DataFrame({"patientunitstayid": [30, 10, 20, 40]})
        table = pd.DataFrame({"patientunitstayid": [40, 20, 10, 30], "hospitaldischargestatus": ["Expired", "Alive", "Alive", "Expired"]})     # shuffled
        inc = pd.DataFrame({"patientunitstayid": [10, 20, 30, 40], "label": [1.0, 0.0, np.nan, np.nan], "included": [True, True, False, False]})
        mort, sep = te.build_task_labels(pats, table, inc)
        self.assertEqual(mort.tolist(), [1.0, 0.0, 0.0, 1.0])                              # 30, 10, 20, 40
        self.assertEqual(sep[1], 1.0); self.assertEqual(sep[2], 0.0)
        self.assertTrue(np.isnan(sep[0]) and np.isnan(sep[3]))                              # excluded patients carry NO sepsis label (never a fabricated 0)

    def test_an_unrecognised_discharge_status_is_unknown_not_negative(self):
        pats = pd.DataFrame({"patientunitstayid": [1, 2]})
        table = pd.DataFrame({"patientunitstayid": [1, 2], "hospitaldischargestatus": ["Alive", None]})
        mort, _ = te.build_task_labels(pats, table, pd.DataFrame({"patientunitstayid": [1, 2], "label": [0.0, 0.0], "included": [True, True]}))
        self.assertEqual(mort[0], 0.0); self.assertTrue(np.isnan(mort[1]))

    def test_label_counts_per_split(self):
        mort = np.array([1, 0, np.nan, 1.0]); sep = np.array([np.nan, 1, 0, np.nan]); split = np.array(["train", "train", "val", "test"])
        c = te.label_counts(mort, sep, split)
        self.assertEqual(c["train"]["mortality"], {"known": 2, "positive": 1, "negative": 1})
        self.assertEqual(c["val"]["sepsis_incident"], {"known": 1, "positive": 0, "negative": 1}); self.assertEqual(c["test"]["patients"], 1)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestModelAndTraining(unittest.TestCase):
    def test_architecture_reuses_the_existing_mlp_trunk_and_dimensions_are_configurable(self):
        from src.modalities.tabular.mlp_encoder import TabularMLPEncoder
        import torch
        for dim in (8, 64):
            m = te.make_model(30, {"hidden_dim": 16, "embedding_dim": dim, "dropout": 0.1})
            self.assertIsInstance(m.encoder, TabularMLPEncoder)
            e, ml, sl = m(torch.randn(5, 30))
            self.assertEqual((tuple(e.shape), tuple(ml.shape), tuple(sl.shape)), ((5, dim), (5,), (5,)))

    def _task(self, n=600, d=20, seed=0):
        rs = np.random.default_rng(seed)
        X = rs.normal(size=(n, d)).astype(np.float32)
        mort = (X[:, 0] + 0.3 * rs.normal(size=n) > 0.6).astype(float)               # learnable signal in feature 0
        sep = (X[:, 1] + 0.3 * rs.normal(size=n) > 0.6).astype(float); sep[rs.random(n) < 0.5] = np.nan
        split = np.array(["train"] * int(n * .7) + ["val"] * int(n * .15) + ["test"] * (n - int(n * .7) - int(n * .15)))
        return X, mort, sep, split

    def test_training_reduces_the_focal_loss_and_the_embedding_carries_the_planted_signal(self):
        X, mort, sep, split = self._task()
        cfg = tcfg()
        model, h = te.train_tcd_encoder(X, mort, sep, split, cfg["encoder"], cfg["training"])
        self.assertLess(h["epochs"][-1]["train_loss"], h["epochs"][0]["train_loss"])
        best = h["epochs"][h["best_epoch"] - 1]
        self.assertGreater(best["val_auroc_mortality"], 0.8); self.assertGreater(best["val_auroc_sepsis"], 0.8)
        self.assertEqual(h["hyperparameters"]["gamma"], 2.0); self.assertEqual(h["input_dim"], 20); self.assertEqual(h["seed"], 42)
        emb = te.encode_all(model, X)
        self.assertEqual((emb.shape, emb.dtype), ((600, 8), np.float32)); self.assertTrue(np.isfinite(emb).all())

    def test_deterministic_given_the_seed(self):
        X, mort, sep, split = self._task(seed=1)
        cfg = tcfg()
        m1, h1 = te.train_tcd_encoder(X, mort, sep, split, cfg["encoder"], cfg["training"])
        m2, h2 = te.train_tcd_encoder(X, mort, sep, split, cfg["encoder"], cfg["training"])
        np.testing.assert_array_equal(te.encode_all(m1, X), te.encode_all(m2, X))
        self.assertEqual([e["train_loss"] for e in h1["epochs"]], [e["train_loss"] for e in h2["epochs"]])
        m3, _ = te.train_tcd_encoder(X, mort, sep, split, cfg["encoder"], {**cfg["training"], "seed": 7})
        self.assertFalse(np.array_equal(te.encode_all(m1, X), te.encode_all(m3, X)))

    def test_focal_parameters_and_task_weights_really_drive_the_objective(self):
        X, mort, sep, split = self._task(seed=2)
        base = tcfg()
        _, ha = te.train_tcd_encoder(X, mort, sep, split, base["encoder"], base["training"])
        _, hb = te.train_tcd_encoder(X, mort, sep, split, base["encoder"], {**base["training"], "gamma": 4.0})
        _, hc = te.train_tcd_encoder(X, mort, sep, split, base["encoder"], {**base["training"], "alpha": 0.75})
        self.assertNotEqual(ha["epochs"][0]["train_loss"], hb["epochs"][0]["train_loss"]); self.assertNotEqual(ha["epochs"][0]["train_loss"], hc["epochs"][0]["train_loss"])
        # sepsis_weight = 0 -> the sepsis head receives no gradient (weight decay off) and stays at its initial value
        import torch
        c = {**base["training"], "sepsis_weight": 0.0, "weight_decay": 0.0}
        torch.manual_seed(c["seed"]); init = te.make_model(20, base["encoder"]).sepsis_head.weight.detach().clone()
        m, _ = te.train_tcd_encoder(X, mort, sep, split, base["encoder"], c)
        torch.testing.assert_close(m.sepsis_head.weight.detach(), init)
        self.assertGreater(float((m.mortality_head.weight.detach() - te.make_model(20, base["encoder"]).mortality_head.weight.detach()).abs().max()), 0)

    def test_a_task_with_no_labels_in_the_data_does_not_break_training(self):
        X, mort, sep, split = self._task(seed=3)
        cfg = tcfg()
        _, h = te.train_tcd_encoder(X, mort, np.full(len(sep), np.nan), split, cfg["encoder"], cfg["training"])
        self.assertLess(h["epochs"][-1]["train_loss"], h["epochs"][0]["train_loss"]); self.assertIsNone(h["epochs"][0]["val_auroc_sepsis"])

    def test_non_finite_inputs_fail_loudly(self):
        X, mort, sep, split = self._task(seed=4)
        X[0, 0] = np.inf
        cfg = tcfg()
        with self.assertRaises(FloatingPointError):
            te.train_tcd_encoder(X, mort, sep, split, cfg["encoder"], {**cfg["training"], "batch_size": 100000})

    def test_the_returned_model_is_the_best_validation_epoch_not_the_last(self):
        import torch
        rs = np.random.default_rng(9)
        X = rs.normal(size=(260, 60)).astype(np.float32)                                      # many features, few patients, noisy labels -> overfits quickly
        mort = (rs.random(260) < 0.4).astype(float); sep = np.full(260, np.nan)
        split = np.array(["train"] * 90 + ["val"] * 170)
        cfg = tcfg(**{"training.lr": 0.02, "training.max_epochs": 40, "training.patience": 3, "training.weight_decay": 0.0, "encoder.hidden_dim": 64})
        model, h = te.train_tcd_encoder(X, mort, sep, split, cfg["encoder"], cfg["training"])
        self.assertLess(h["best_epoch"], len(h["epochs"]))                                     # (teeth) the last epoch is NOT the best one here
        model.eval()
        with torch.no_grad():
            _, vm, vs = model(torch.from_numpy(X[split == "val"]))
            loss, _, _ = te._multitask_focal(vm, vs, torch.from_numpy(mort[split == "val"].astype(np.float32)), torch.from_numpy(sep[split == "val"].astype(np.float32)), cfg["training"])
        self.assertAlmostEqual(float(loss), h["best_score"], places=5)                         # the returned weights ARE the best-validation weights

    def test_validation_labels_only_select_the_model_they_never_train_it(self):
        X, mort, sep, split = self._task(seed=5)
        cfg = tcfg()
        _, a = te.train_tcd_encoder(X, mort, sep, split, cfg["encoder"], cfg["training"])
        mort2 = mort.copy(); mort2[split == "val"] = 1 - mort2[split == "val"]
        _, b = te.train_tcd_encoder(X, mort2, sep, split, cfg["encoder"], cfg["training"])
        self.assertEqual(a["epochs"][0]["train_loss"], b["epochs"][0]["train_loss"])           # epoch-1 training saw no validation label
        self.assertNotEqual(a["epochs"][0]["val_loss"], b["epochs"][0]["val_loss"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestOrchestration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.pats, _ = make_dataset(cls.root / "data", n_pat=220, seed=6)
        cls.man = make_manifest(cls.pats)
        cls.cfg = tcfg()
        cls.feat = tf.build_tcd_features({"data_root": str(cls.root / "data")}, cls.pats, cls.man, cls.cfg, cls.root / "f")
        rs = np.random.default_rng(1)
        inc = pd.DataFrame({"patientunitstayid": cls.pats.patientunitstayid})
        inc["included"] = rs.random(len(inc)) < 0.6; inc["label"] = np.where(inc["included"], (rs.random(len(inc)) < 0.25).astype(float), np.nan)
        cls.inc_dir = cls.root / "inc"; cls.inc_dir.mkdir()
        inc.to_csv(cls.inc_dir / "incident_sepsis_labels.csv", index=False)
        (cls.inc_dir / "manifest.json").write_text(json.dumps({"identity": "inc-identity-A", "run_id": "incA", "files": {}}))
        cls.inc = inc

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, out, pats=None, cfg=None, inc_dir=None, **kw):
        return te.build_tcd_encoder(self.feat["run_dir"], inc_dir or self.inc_dir, self.pats if pats is None else pats, cfg or self.cfg, self.root / out, **kw)

    def test_artifacts_and_fusion_compatible_embedding_file(self):
        r = self.build("e1"); d = Path(r["run_dir"])
        for n in ("tcd_embeddings.npz", "tcd_encoder.pt", "training_log.json", "label_counts.json", "config_snapshot.json", "manifest.json"):
            self.assertTrue((d / n).is_file(), n)
        z = np.load(d / "tcd_embeddings.npz")
        self.assertEqual(set(z.files), {"patient_ids", "embeddings"})                           # the same keys the GRU / autoencoder files use
        self.assertEqual(z["embeddings"].shape, (len(self.pats), 8)); self.assertEqual(z["embeddings"].dtype, np.float32); self.assertTrue(np.isfinite(z["embeddings"]).all())
        self.assertEqual(z["patient_ids"].tolist(), sorted(self.pats.patientunitstayid.tolist()))
        target = z["patient_ids"][::-1].tolist()                                                # the fusion code re-orders by explicit ID lookup
        aligned = align_embeddings_to_cohort(z["patient_ids"].tolist(), z["embeddings"], target)
        np.testing.assert_array_equal(aligned, z["embeddings"][::-1])
        m = json.loads((d / "manifest.json").read_text())
        self.assertEqual((m["embedding_dim"], m["feature_identity"]), (8, self.feat["manifest"]["identity"]))
        ck = torch.load(d / "tcd_encoder.pt", weights_only=False)
        self.assertEqual(ck["input_dim"], len(ck["transformed_feature_names"])); self.assertEqual(ck["encoder_config"]["embedding_dim"], 8)

    def test_test_patients_labels_never_influence_the_trained_encoder(self):
        sp = self.feat_split()
        pats2 = self.pats.copy()
        te_ids = {i for i, s in sp.items() if s == "test"}
        flip = pats2["patientunitstayid"].isin(te_ids)
        pats2.loc[flip, "hospitaldischargestatus"] = pats2.loc[flip, "hospitaldischargestatus"].map({"Alive": "Expired", "Expired": "Alive"})
        inc2 = self.inc.copy(); m = inc2["patientunitstayid"].isin(te_ids) & inc2["included"]; inc2.loc[m, "label"] = 1 - inc2.loc[m, "label"]
        d2 = self.root / "inc2"; d2.mkdir(exist_ok=True); inc2.to_csv(d2 / "incident_sepsis_labels.csv", index=False)
        (d2 / "manifest.json").write_text(json.dumps({"identity": "inc-identity-B", "run_id": "incB", "files": {}}))
        a = np.load(Path(self.build("e_test_a", force=True)["run_dir"]) / "tcd_embeddings.npz")["embeddings"]
        b = np.load(Path(self.build("e_test_b", pats=pats2, inc_dir=d2, force=True)["run_dir"]) / "tcd_embeddings.npz")["embeddings"]
        np.testing.assert_array_equal(a, b)                                                     # bit-identical: test labels were unreachable
        # (sanity) the same flip on TRAIN patients does change the encoder
        tr_ids = {i for i, s in sp.items() if s == "train"}
        pats3 = self.pats.copy(); f3 = pats3["patientunitstayid"].isin(tr_ids)
        pats3.loc[f3, "hospitaldischargestatus"] = pats3.loc[f3, "hospitaldischargestatus"].map({"Alive": "Expired", "Expired": "Alive"})
        c = np.load(Path(self.build("e_test_c", pats=pats3, force=True)["run_dir"]) / "tcd_embeddings.npz")["embeddings"]
        self.assertFalse(np.array_equal(a, c))

    def feat_split(self):
        p = tf.load_tcd_features(self.feat["run_dir"])["patients"]
        return dict(zip(p["patientunitstayid"], p["split"]))

    def test_cache_reuse_and_identity(self):
        r1 = self.build("e2"); self.assertFalse(r1["cache_hit"])
        r2 = self.build("e2"); self.assertTrue(r2["cache_hit"]); self.assertEqual(r2["run_id"], r1["run_id"])
        r3 = self.build("e2", cfg=tcfg(**{"training.gamma": 3.0})); self.assertFalse(r3["cache_hit"])          # focal gamma is part of the identity
        r4 = self.build("e2", cfg=tcfg(**{"encoder.embedding_dim": 4})); self.assertFalse(r4["cache_hit"])
        d5 = self.root / "inc5"; d5.mkdir(exist_ok=True); (d5 / "incident_sepsis_labels.csv").write_text((self.inc_dir / "incident_sepsis_labels.csv").read_text())
        (d5 / "manifest.json").write_text(json.dumps({"identity": "inc-identity-C", "run_id": "incC", "files": {}}))
        r5 = self.build("e2", inc_dir=d5); self.assertFalse(r5["cache_hit"])                                    # a different incident label run is a different encoder
        self.assertEqual(len({r1["run_id"], r3["run_id"], r4["run_id"], r5["run_id"]}), 4)
        with self.assertRaises(FileExistsError):
            self.build("e2", force=True, run_id=r1["run_id"])

    def _flip(self, pats, ids):
        out = pats.copy(); m = out["patientunitstayid"].isin(ids)
        out.loc[m, "hospitaldischargestatus"] = out.loc[m, "hospitaldischargestatus"].map({"Alive": "Expired", "Expired": "Alive"})
        return out

    def _emb(self, out, **kw):
        return np.load(Path(self.build(out, force=True, **kw)["run_dir"]) / "tcd_embeddings.npz")["embeddings"]

    def test_landmark_only_training_ignores_short_stay_patients_labels_in_training_and_early_stopping(self):
        sp = self.feat_split()
        short = set(self.pats.loc[self.pats["unitdischargeoffset"] < 1440, "patientunitstayid"])
        self.assertGreater(len(short), 10)
        tr_short = {i for i in short if sp[i] == "train"}; va_short = {i for i in short if sp[i] == "val"}
        tr_long = {i for i, s_ in sp.items() if s_ == "train" and i not in short}
        va_long = {i for i, s_ in sp.items() if s_ == "val" and i not in short}
        def run(out, **kw):
            d = Path(self.build(out, force=True, **kw)["run_dir"])
            return np.load(d / "tcd_embeddings.npz")["embeddings"], json.loads((d / "training_log.json").read_text())
        base, log0 = run("e_lm_base")
        ea, _ = run("e_lm_a", pats=self._flip(self.pats, tr_short))
        np.testing.assert_array_equal(base, ea)                                                                       # short-stay TRAIN labels: unreachable
        eb, logb = run("e_lm_b", pats=self._flip(self.pats, va_short))
        np.testing.assert_array_equal(base, eb)                                                                       # short-stay VAL labels: unreachable ...
        self.assertEqual([e["val_loss"] for e in log0["epochs"]], [e["val_loss"] for e in logb["epochs"]])             # ... in the validation LOSS itself (not just the chosen epoch)
        self.assertEqual(log0["best_score"], logb["best_score"])
        ec, logc = run("e_lm_c", pats=self._flip(self.pats, tr_long))
        self.assertFalse(np.array_equal(base, ec))                                                                    # (sanity) landmark train labels DO matter
        _, logd = run("e_lm_f", pats=self._flip(self.pats, va_long))
        self.assertNotEqual([e["val_loss"] for e in log0["epochs"]], [e["val_loss"] for e in logd["epochs"]])         # (sanity) landmark val labels DO drive the validation loss
        off = tcfg(**{"training.landmark_only": False})
        self.assertFalse(np.array_equal(self._emb("e_lm_d", cfg=off), self._emb("e_lm_e", cfg=off, pats=self._flip(self.pats, tr_short))))   # (sanity) the flag has an effect

    def test_every_patient_is_still_embedded_and_the_population_is_recorded(self):
        r = self.build("e_pop", force=True); d = Path(r["run_dir"])
        z = np.load(d / "tcd_embeddings.npz"); self.assertEqual(z["embeddings"].shape[0], len(self.pats))
        m = json.loads((d / "manifest.json").read_text()); pop = m["training_population"]
        sp = self.feat_split(); ex = dict(zip(self.pats["patientunitstayid"], self.pats["unitdischargeoffset"]))
        for part in ("train", "val", "test"):
            ids = [i for i, s_ in sp.items() if s_ == part]
            self.assertEqual(pop["patients_by_split"][part], len(ids))
            self.assertEqual(pop["landmark_patients_by_split"][part], sum(ex[i] >= 1440 for i in ids))
        self.assertEqual(pop["excluded_from_training_and_early_stopping"]["train"], pop["patients_by_split"]["train"] - pop["landmark_patients_by_split"]["train"])
        self.assertTrue(pop["landmark_only"]); self.assertIn(">= 1440", pop["rule"])
        h = json.loads((d / "training_log.json").read_text())
        self.assertEqual(h["n_train"], pop["landmark_patients_by_split"]["train"]); self.assertEqual(h["n_val"], pop["landmark_patients_by_split"]["val"])
        self.assertLess(h["n_train"], pop["patients_by_split"]["train"])

    def test_landmark_flag_is_part_of_the_encoder_identity(self):
        r1 = self.build("e_id"); r2 = self.build("e_id", cfg=tcfg(**{"training.landmark_only": False}))
        self.assertFalse(r2["cache_hit"]); self.assertNotEqual(r1["run_id"], r2["run_id"])

    def test_a_patient_table_without_the_exit_time_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            self.build("e_noexit", force=True, pats=self.pats.drop(columns="unitdischargeoffset"))
        self.build("e_noexit_ok", force=True, pats=self.pats.drop(columns="unitdischargeoffset"), cfg=tcfg(**{"training.landmark_only": False}))     # not needed when the flag is off

    def test_overlap_audit_is_recorded_in_the_encoder_manifest(self):
        m = json.loads((Path(self.build("e_ov", force=True)["run_dir"]) / "manifest.json").read_text())
        self.assertIn("creatinine", m["overlap_audit"]["lab_ingredients_of_the_label"]); self.assertNotIn("sodium", m["overlap_audit"]["lab_ingredients_of_the_label"])
        self.assertIn("ventilation_support_settings", m["overlap_audit"])

    def test_training_log_records_hyperparameters_seed_and_label_counts(self):
        r = self.build("e3", force=True)
        h = json.loads((Path(r["run_dir"]) / "training_log.json").read_text())
        for k in ("hyperparameters", "encoder", "seed", "best_epoch", "epochs", "input_dim", "n_train", "n_val", "monitor"):
            self.assertIn(k, h)
        lc = json.loads((Path(r["run_dir"]) / "label_counts.json").read_text())
        self.assertEqual(sum(lc[s]["patients"] for s in ("train", "val", "test")), len(self.pats))
        self.assertGreater(lc["train"]["sepsis_incident"]["known"], 0)


if __name__ == "__main__":
    unittest.main()
