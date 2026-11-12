"""Stage 3b -- tabular autoencoder inputs (tabular_inputs.py) and the autoencoder stage (tabular_ae.py). Torch-dependent tests are skipped without torch."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from _tcd_synth import make_manifest  # noqa: E402
from generate_mock_eicu import generate  # noqa: E402

from src.three_modality import artifacts, tabular_inputs as ti, tcd_features as tf  # noqa: E402
from src.three_modality.config import THREE_MODALITY_DEFAULTS, get_tabular_cfg, validate_tabular_cfg, validate_tcd_cfg  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:       # pragma: no cover
    HAVE_TORCH = False

ALLOWED_GROUPS = {"demographics", "apache_comorbidity", "diagnosis", "admission_drug", "missing_indicator"}


def cfgs(**tab_over):
    t = copy.deepcopy(THREE_MODALITY_DEFAULTS["tcd"]); t["variables"].update({"chunksize": 5000, "min_train_coverage": 0.01}); t["audit"]["n_patients"] = 0
    a = copy.deepcopy(THREE_MODALITY_DEFAULTS["tabular"]); a["bundle_overrides"] = {"min_vocab_patients": 1}
    a["encoder"].update({"hidden_dim": 16, "embedding_dim": 8}); a["training"].update({"max_epochs": 6, "patience": 3, "batch_size": 64, "lr": 0.005})
    for k, v in tab_over.items():
        x = a
        parts = k.split(".")
        for p in parts[:-1]:
            x = x[p]
        x[parts[-1]] = v
    return validate_tcd_cfg(t, 24), validate_tabular_cfg(a)


class TestTypedPreprocessing(unittest.TestCase):
    def _data(self):
        rs = np.random.default_rng(0); n = 300
        split = np.array(["train"] * 200 + ["val"] * 50 + ["test"] * 50)
        raw = np.column_stack([rs.normal(5, 2, n), (rs.random(n) < 0.3).astype(float), np.full(n, 1.0), rs.normal(0, 1, n), (rs.random(n) < 0.5).astype(float)]).astype(np.float32)
        miss = rs.random(n) < 0.2; raw[miss, 3] = np.nan; raw[7, 4] = np.nan; raw[5, 0] = 1e6
        return raw, ["cont", "bin", "const_bin", "cont_missing", "bin_nan"], ["g"] * 5, split

    def test_types_binary_columns_stay_zero_one_constants_dropped_indicators_binary(self):
        raw, names, groups, split = self._data()
        r = ti.build_inputs_matrix(raw, names, groups, split == "train", THREE_MODALITY_DEFAULTS["tabular"]["preprocessing"])
        self.assertEqual(r["n_continuous"], 2); self.assertEqual(r["binary_kept"], ["bin", "bin_nan"])
        self.assertEqual([d["feature"] for d in r["dropped"]], ["const_bin"]); self.assertEqual(r["dropped"][0]["code"], "constant")
        self.assertEqual(r["kinds"], ["continuous"] * 2 + ["binary"] * 2 + ["missing_indicator"] * r["n_indicators"])
        self.assertEqual(r["is_binary"].tolist(), [k != "continuous" for k in r["kinds"]])
        X = r["X"]; self.assertEqual(X.shape, (300, len(r["names"]))); self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.isin(X[:, r["is_binary"]], [0, 1]).all())                       # every binary / indicator column is exactly 0/1 (valid BCE target)
        self.assertLess(np.abs(X[5, 0]), 50)                                               # winsorised outlier
        self.assertEqual(r["binary_nan_imputed"]["bin_nan"], 1)                            # NaN imputed to 0 and COUNTED

    def test_everything_is_fitted_on_training_rows_only(self):
        raw, names, groups, split = self._data()
        pp = THREE_MODALITY_DEFAULTS["tabular"]["preprocessing"]
        a = ti.build_inputs_matrix(raw, names, groups, split == "train", pp)
        bad = raw.copy(); bad[split != "train", 0] = 1e9; bad[split != "train", 3] = -1e9; bad[split != "train", 1] = 1 - bad[split != "train", 1]
        b = ti.build_inputs_matrix(bad, names, groups, split == "train", pp)
        for k in ("lo", "hi", "median", "mean", "std"):
            np.testing.assert_array_equal(a["continuous_state"][k], b["continuous_state"][k], err_msg=k)
        self.assertEqual((a["names"], a["binary_kept"]), (b["names"], b["binary_kept"]))
        np.testing.assert_array_equal(a["X"][split == "train"], b["X"][split == "train"])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        generate(str(cls.root / "d"), n_hospitals=4, n_patients_per_hospital=60, seed=1)
        cls.pats = pd.read_csv(cls.root / "d" / "patient.csv"); cls.man = make_manifest(cls.pats)
        cls.src = {"data_root": str(cls.root / "d")}; cls.tcfg, cls.acfg = cfgs()
        cls.feat = tf.build_tcd_features(cls.src, cls.pats, cls.man, cls.tcfg, cls.root / "f")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, out, src=None, acfg=None, feat=None, **kw):
        return ti.build_tabular_inputs(src or self.src, self.pats, (feat or self.feat)["run_dir"], acfg or self.acfg, self.tcfg, self.root / out, **kw)

    def test_clean_static_set_only_in_the_tcd_patient_order(self):
        r = self.build("t1", force=True); x = ti.load_tabular_inputs(r["run_dir"]); tcd = tf.load_tcd_features(self.feat["run_dir"])
        pd.testing.assert_frame_equal(x["patients"], tcd["patients"])                      # one patient ordering for every modality
        self.assertEqual(x["X"].shape[0], len(self.pats)); self.assertTrue(np.isfinite(np.asarray(x["X"])).all())
        self.assertTrue(set(x["schema"]["groups"]) <= ALLOWED_GROUPS, set(x["schema"]["groups"]) - ALLOWED_GROUPS)
        for n in x["schema"]["names"]:
            self.assertIsNone(ti.FORBIDDEN_NAME.search(n), n)                              # no labs, APACHE scores, outcome-like names
        rep = r["report"]; self.assertEqual(rep["forbidden_name_check"], "passed")
        self.assertTrue({"apache_score", "lab", "lab_sofa", "treatment", "medication", "hospital"} <= set(rep["excluded_groups"]))
        self.assertEqual(sum(rep["patients"][k] for k in ("train", "val", "test")), rep["patients"]["total"])

    def test_events_after_the_window_cannot_change_the_inputs(self):
        late = self.root / "late"; late.mkdir(exist_ok=True)
        for f in (self.root / "d").glob("*.csv"):
            df = pd.read_csv(f)
            if f.stem in ("diagnosis", "admissionDrug", "treatment", "medication"):
                extra = df.head(40).copy()
                for c in [c for c in extra.columns if c.lower().endswith("offset")]:
                    extra[c] = 5000                                                         # far beyond the 1440-minute window
                df = pd.concat([df, extra], ignore_index=True)
            df.to_csv(late / f.name, index=False)
        a = ti.load_tabular_inputs(self.build("t2", force=True)["run_dir"])
        b = ti.load_tabular_inputs(ti.build_tabular_inputs({"data_root": str(late)}, self.pats, self.feat["run_dir"], self.acfg, self.tcfg, self.root / "t2late")["run_dir"])
        self.assertEqual(a["schema"]["names"], b["schema"]["names"]); np.testing.assert_array_equal(np.asarray(a["X"]), np.asarray(b["X"]))

    def test_cache_force_tamper_and_identity(self):
        r1 = self.build("t3"); self.assertFalse(r1["cache_hit"]); d1 = Path(r1["run_dir"]); h1 = artifacts.sha256_file(d1 / "tab_inputs.npy")
        self.assertTrue(self.build("t3")["cache_hit"])
        r3 = self.build("t3", force=True, run_id="forced"); self.assertEqual(artifacts.sha256_file(Path(r3["run_dir"]) / "tab_inputs.npy"), h1)     # deterministic
        with self.assertRaises(FileExistsError):
            self.build("t3", force=True, run_id="forced")
        _, a2 = cfgs(); a2["feature_set"]["exclude_groups"] = a2["feature_set"]["exclude_groups"] + ["diagnosis"]
        self.assertFalse(self.build("t3", acfg=a2)["cache_hit"])                           # the exclusion list is part of the identity
        _, a3 = cfgs(**{"encoder.embedding_dim": 4}); self.assertTrue(self.build("t3", acfg=a3)["cache_hit"])                                     # encoder settings are not
        for d in (d1, Path(r3["run_dir"])):
            b = bytearray((d / "tab_inputs.npy").read_bytes()); b[len(b) // 2] ^= 1; (d / "tab_inputs.npy").write_bytes(bytes(b))
        self.assertFalse(self.build("t3")["cache_hit"])                                    # every identical run tampered -> recompute


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestAutoencoderStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.three_modality import tabular_ae as ta
        cls.ta = ta
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        generate(str(cls.root / "d"), n_hospitals=4, n_patients_per_hospital=80, seed=3)
        cls.pats = pd.read_csv(cls.root / "d" / "patient.csv"); cls.man = make_manifest(cls.pats)
        cls.src = {"data_root": str(cls.root / "d")}; cls.tcfg, cls.acfg = cfgs()
        feat = tf.build_tcd_features(cls.src, cls.pats, cls.man, cls.tcfg, cls.root / "f")
        cls.inp = ti.build_tabular_inputs(cls.src, cls.pats, feat["run_dir"], cls.acfg, cls.tcfg, cls.root / "t")
        rs = np.random.default_rng(1); inc = pd.DataFrame({"patientunitstayid": cls.pats.patientunitstayid})
        inc["included"] = rs.random(len(inc)) < 0.6; inc["label"] = np.where(inc["included"], (rs.random(len(inc)) < 0.25).astype(float), np.nan)
        cls.inc_dir = cls.root / "inc"; cls.inc_dir.mkdir(); inc.to_csv(cls.inc_dir / "incident_sepsis_labels.csv", index=False)
        (cls.inc_dir / "manifest.json").write_text(json.dumps({"identity": "inc-A", "run_id": "incA", "files": {}}))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, out, pats=None, acfg=None, **kw):
        return self.ta.build_tabular_ae(self.inp["run_dir"], self.inc_dir, self.pats if pats is None else pats, acfg or self.acfg, self.tcfg, self.root / out, **kw)

    def test_reconstruction_loss_is_typed_and_gradients_reach_every_part(self):
        import torch
        from src.three_modality.tabular_ae import make_reconstruction_loss, make_tabular_ae_model
        is_bin = np.array([False, False, True, True])
        x = torch.tensor([[1.0, 2.0, 1.0, 0.0]]); recon = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        lf = make_reconstruction_loss(is_bin, {"continuous_weight": 1.0, "binary_weight": 1.0})
        mse = (1.0 + 4.0) / 2; bce = (np.log(2) + np.log(2)) / 2                              # logit 0 -> p = 0.5 for both binary targets
        self.assertAlmostEqual(float(lf((recon, x), None)), mse + bce, places=5)
        self.assertAlmostEqual(float(make_reconstruction_loss(is_bin, {"continuous_weight": 0.0, "binary_weight": 1.0})((recon, x), None)), bce, places=5)
        m = make_tabular_ae_model(6, {"hidden_dim": 8, "embedding_dim": 4, "dropout": 0.0}); xb = torch.randn(10, 6)
        e, ml, sl, aux = m(xb)
        (ml.mean() + sl.mean() + make_reconstruction_loss(np.array([False] * 3 + [True] * 3), {"continuous_weight": 1.0, "binary_weight": 1.0})(aux, None)).backward()
        for n, p in m.named_parameters():
            self.assertIsNotNone(p.grad, n); self.assertGreater(float(p.grad.abs().sum()), 0, n)          # encoder, decoder AND both focal heads

    def test_stage_artifacts_objectives_and_split_discipline(self):
        r = self.build("e1", force=True); d = Path(r["run_dir"]); h = r["history"]
        z = np.load(d / "tab_ae_embeddings.npz"); self.assertEqual(set(z.files), {"patient_ids", "embeddings"})
        self.assertEqual((z["embeddings"].shape, z["embeddings"].dtype), ((len(self.pats), 8), np.float32)); self.assertTrue(np.isfinite(z["embeddings"]).all())
        self.assertTrue(all("val_aux" in e and "val_focal" in e for e in h["epochs"]))                    # reconstruction AND focal are both active and logged separately
        self.assertEqual(h["aux_weight"], 1.0); self.assertGreater(h["n_train"], 10)
        self.assertLess(h["epochs"][-1]["val_aux"], h["epochs"][0]["val_aux"])                            # the autoencoder actually reconstructs better
        pop = r["manifest"]["training_population"]; self.assertTrue(pop["landmark_only"]); self.assertLess(pop["landmark_patients_by_split"]["train"], pop["patients_by_split"]["train"])
        sp = dict(zip(self.pats["patientunitstayid"], pd.read_csv(Path(self.inp["run_dir"]) / "patients.csv")["split"]))
        flip = self.pats.copy(); m = flip["patientunitstayid"].map(lambda i: sp[i] == "test")
        flip.loc[m, "hospitaldischargestatus"] = flip.loc[m, "hospitaldischargestatus"].map({"Alive": "Expired", "Expired": "Alive"})
        z2 = np.load(Path(self.build("e2", pats=flip, force=True)["run_dir"]) / "tab_ae_embeddings.npz")["embeddings"]
        np.testing.assert_array_equal(z["embeddings"], z2)                                              # TEST labels are unreachable (bit-identical)
        r_again = self.build("e1"); self.assertTrue(r_again["cache_hit"]); self.assertEqual(r_again["run_id"], r["run_id"])

    def test_reconstruction_only_and_supervised_only_are_both_valid_settings(self):
        a_rec = cfgs(**{"training.task_weights": {"mortality": 0.0, "sepsis": 0.0}})[1]
        h = self.build("e3", acfg=a_rec, force=True)["history"]; self.assertLess(h["epochs"][-1]["train_loss"], h["epochs"][0]["train_loss"])
        a_sup = cfgs(**{"training.reconstruction_weight": 0.0})[1]
        h2 = self.build("e4", acfg=a_sup, force=True)["history"]; self.assertEqual(h2["aux_weight"], 0.0); self.assertNotIn("val_aux", h2["epochs"][0])
        with self.assertRaises(ValueError):
            cfgs(**{"training.reconstruction_weight": 0.0, "training.task_weights": {"mortality": 0.0, "sepsis": 0.0}})


if __name__ == "__main__":
    unittest.main()
