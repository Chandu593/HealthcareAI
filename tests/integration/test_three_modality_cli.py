"""run_three_modality.py end to end through the real CLI (subprocess): label build, verified-run reuse, --inspect-cache, --force-recompute,
and the guarantees that the permanent split manifest is only LOADED (never built) and that bad parameters fail before anything is written."""

import json
import numpy as np
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.splitting.patient_split import SplitManifest, save_split_manifest

ROOT = Path(__file__).resolve().parents[2]


class TestThreeModalityCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        d = self.data = self.root / "eicu"; d.mkdir()
        pd.DataFrame([(1, 1, 7000, "Alive"), (2, 1, 7000, "Alive"), (3, 1, 3000, "Alive")],
                     columns=["patientunitstayid", "hospitalid", "unitdischargeoffset", "unitdischargestatus"]).to_csv(d / "patient.csv", index=False)
        med = pd.DataFrame([(1, "Vancomycin", 1600), (2, "Acetaminophen", 100), (3, "Acetaminophen", 100), (1, "Acetaminophen", 100)], columns=["patientunitstayid", "drugname", "drugstartoffset"])
        for t in ("medication", "medication_imputed", "medication_imputed1"):
            (med if t == "medication" else med.iloc[:0]).to_csv(d / f"{t}.csv", index=False)
        pd.DataFrame([(1, 1700, "creatinine", 2.5)], columns=["patientunitstayid", "labresultoffset", "labname", "labresult"]).to_csv(d / "lab.csv", index=False)
        pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname"]).to_csv(d / "infusionDrug.csv", index=False)
        pd.DataFrame(columns=["patientunitstayid", "treatmentoffset", "treatmentstring"]).to_csv(d / "treatment.csv", index=False)
        self.manifest_path = self.root / "split_manifest.json"
        save_split_manifest(SplitManifest(seed=1, train_frac=.7, val_frac=.15, test_frac=.15, stratify_on="none", sepsis_cohort=None,
                                          patient_to_split={"1": "train", "2": "val", "3": "test"}), self.manifest_path)
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *extra, manifest=True):
        sets = ["--set", f"data.root={self.data}", "--set", f"run.output_dir={self.out}"]
        if manifest:
            sets += ["--set", f"ablation.split.manifest_path={self.manifest_path}"]
        else:
            sets += ["--set", f"ablation.split.manifest_path={self.root / 'missing.json'}"]
        return subprocess.run([sys.executable, "run_three_modality.py", "--config", "configs/three_modality.yaml", "--stage", "incident_label", *sets, *extra],
                              cwd=ROOT, capture_output=True, text=True)

    @staticmethod
    def summary(stdout: str) -> dict:
        """The CLI prints its JSON summary last, starting at the beginning of a line; log lines above may contain braces of their own."""
        start = stdout.rindex("\n{") + 1 if "\n{" in stdout else stdout.index("{")
        return json.loads(stdout[start: stdout.rindex("}") + 1])

    def test_build_reuse_inspect_and_force(self):
        r1 = self.cli(); self.assertEqual(r1.returncode, 0, r1.stderr + r1.stdout)
        s1 = self.summary(r1.stdout)
        self.assertEqual((s1["cache_hit"], s1["initial_eligible_cohort"], s1["incident_positive"], s1["incident_negative"], s1["censored_or_excluded_followup"]),
                         (False, 3, 1, 2, 0))     # DEFAULT = alive_discharge_as_negative: patient 3 (at risk at hour 24, alive ICU discharge at 50 h, no event) is a negative
        rc = self.cli("--set", "three_modality.incident_sepsis.followup.censoring_policy=exclude_censored"); sc = self.summary(rc.stdout)
        self.assertEqual((sc["incident_positive"], sc["incident_negative"], sc["censored_or_excluded_followup"]), (1, 1, 1))   # sensitivity policy: the same patient is censored
        self.assertNotEqual(sc["run_id"], s1["run_id"])
        d = Path(s1["run_dir"]); self.assertEqual(d.parent, self.out / "three_modality" / "incident_sepsis")
        r2 = self.cli(); s2 = self.summary(r2.stdout); self.assertTrue(s2["cache_hit"]); self.assertEqual(s2["run_id"], s1["run_id"])
        ins = self.cli("--inspect-cache"); self.assertEqual(ins.returncode, 0); self.assertIn("matches-current=YES", ins.stdout); self.assertIn(s1["run_id"], ins.stdout)
        r3 = self.cli("--force-recompute"); s3 = self.summary(r3.stdout); self.assertFalse(s3["cache_hit"]); self.assertNotEqual(s3["run_id"], s1["run_id"])
        self.assertEqual(len(list((self.out / "three_modality" / "incident_sepsis").glob("*/manifest.json"))), 3)
        self.assertTrue((self.out / "three_modality" / "logs" / "pipeline.log").is_file())                 # own log directory, not outputs/logs

    def test_changed_parameter_is_not_a_cache_hit_and_inspect_says_so(self):
        s1 = self.summary(self.cli().stdout)
        r = self.cli("--set", "three_modality.incident_sepsis.horizon_end_hours=72"); self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(self.summary(r.stdout)["cache_hit"])
        ins = self.cli("--inspect-cache", "--set", "three_modality.incident_sepsis.horizon_end_hours=48")
        self.assertEqual(ins.stdout.count("matches-current=no"), 2); self.assertNotIn("matches-current=YES", ins.stdout)

    def test_missing_manifest_fails_loudly_and_never_builds_one(self):
        r = self.cli(manifest=False)
        self.assertNotEqual(r.returncode, 0); self.assertIn("Split manifest not found", r.stderr)
        self.assertFalse((self.root / "missing.json").exists())
        self.assertFalse((self.out / "three_modality" / "incident_sepsis").exists() and any((self.out / "three_modality" / "incident_sepsis").iterdir()))

    def test_invalid_parameters_fail_before_anything_is_written(self):
        r = self.cli("--set", "three_modality.incident_sepsis.horizon_end_hours=20")
        self.assertEqual(r.returncode, 2); self.assertIn("horizon_end_hours", r.stderr)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()


class TestTCDStagesCLI(unittest.TestCase):
    """incident_label -> tcd_features -> tcd_encoder through the real CLI on a synthetic eICU-like data root."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tests" / "unit"))
        from _tcd_synth import make_dataset, make_manifest
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        d = cls.data = cls.root / "eicu"
        pats, lab = make_dataset(d, n_pat=180, seed=11)
        meds = pd.DataFrame({"patientunitstayid": pats.patientunitstayid.tolist(), "drugname": "Acetaminophen", "drugstartoffset": 100})
        for t in ("medication", "medication_imputed", "medication_imputed1"):
            (meds if t == "medication" else meds.iloc[:0]).to_csv(d / f"{t}.csv", index=False)
        pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname"]).to_csv(d / "infusionDrug.csv", index=False)
        pd.DataFrame(columns=["patientunitstayid", "treatmentoffset", "treatmentstring"]).to_csv(d / "treatment.csv", index=False)
        cls.manifest_path = cls.root / "split_manifest.json"
        save_split_manifest(make_manifest(pats), cls.manifest_path)
        cls.out = cls.root / "out"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cli(self, stage, *extra):
        sets = ["--set", f"data.root={self.data}", "--set", f"run.output_dir={self.out}", "--set", f"ablation.split.manifest_path={self.manifest_path}",
                "--set", "three_modality.tcd.audit.n_patients=40", "--set", "three_modality.tcd.variables.chunksize=3000", "--set", "three_modality.tcd.encoder.embedding_dim=8",
                "--set", "three_modality.tcd.encoder.hidden_dim=16", "--set", "three_modality.tcd.training.max_epochs=4", "--set", "three_modality.tcd.training.batch_size=64"]
        return subprocess.run([sys.executable, "run_three_modality.py", "--config", "configs/three_modality.yaml", "--stage", stage, *sets, *extra], cwd=ROOT, capture_output=True, text=True)

    summary = staticmethod(TestThreeModalityCLI.summary)

    def test_chain_reuse_inspect_and_error_guidance(self):
        r0 = self.cli("tcd_encoder")                                                    # nothing built yet -> a clear instruction, not a crash
        self.assertNotEqual(r0.returncode, 0); self.assertIn("Run --stage tcd_features first", r0.stderr)
        rf = self.cli("tcd_features"); self.assertEqual(rf.returncode, 0, rf.stderr + rf.stdout)
        sf = self.summary(rf.stdout)
        self.assertFalse(sf["cache_hit"]); self.assertTrue(sf["cutoff_audit_passed"]); self.assertEqual(sf["patients"]["total"], 180)
        self.assertEqual(Path(sf["run_dir"]).parent, self.out / "three_modality" / "tcd" / "features")
        r1 = self.cli("tcd_encoder"); self.assertNotEqual(r1.returncode, 0); self.assertIn("Run --stage incident_label first", r1.stderr)
        ri = self.cli("incident_label"); self.assertEqual(ri.returncode, 0, ri.stderr)
        re_ = self.cli("tcd_encoder"); self.assertEqual(re_.returncode, 0, re_.stderr + re_.stdout)
        se = self.summary(re_.stdout)
        self.assertEqual((se["cache_hit"], se["embedding_shape"]), (False, [180, 8])); self.assertEqual(se["feature_run"], sf["run_id"])
        self.assertTrue(self.summary(self.cli("tcd_features").stdout)["cache_hit"]); self.assertTrue(self.summary(self.cli("tcd_encoder").stdout)["cache_hit"])
        ins = self.cli("tcd_features", "--inspect-cache"); self.assertIn("matches-current=YES", ins.stdout)
        ine = self.cli("tcd_encoder", "--inspect-cache"); self.assertIn("matches-current=YES", ine.stdout)
        z = np.load(Path(se["run_dir"]) / "tcd_embeddings.npz"); self.assertEqual(set(z.files), {"patient_ids", "embeddings"})
        # a changed feature parameter makes the encoder stage ask for new features instead of silently using stale ones
        rx = self.cli("tcd_encoder", "--set", "three_modality.tcd.half_split_minutes=600"); self.assertNotEqual(rx.returncode, 0); self.assertIn("Run --stage tcd_features first", rx.stderr)

    def test_gru_stages_chain_reuse_inspect_and_error_guidance(self):
        small = ["--set", "three_modality.gru.encoder.hidden_dim=8", "--set", "three_modality.gru.encoder.num_layers=1", "--set", "three_modality.gru.training.max_epochs=3",
                 "--set", "three_modality.gru.training.batch_size=64", "--set", "three_modality.gru.audit.n_patients=40"]
        r0 = self.cli("gru_inputs", *small)                                                  # TCD features may not exist yet -> clear instruction
        if r0.returncode != 0:
            self.assertIn("Run --stage tcd_features first", r0.stderr)
            self.assertEqual(self.cli("tcd_features").returncode, 0)
        rg = self.cli("gru_inputs", *small); self.assertEqual(rg.returncode, 0, rg.stderr + rg.stdout)
        sg = self.summary(rg.stdout)
        self.assertTrue(sg["cutoff_audit_passed"]); self.assertEqual(sg["patients"]["total"], 180); self.assertEqual(sg["shapes"]["bins"], 24)
        self.assertEqual(Path(sg["run_dir"]).parent, self.out / "three_modality" / "gru" / "inputs")
        self.assertEqual(self.cli("incident_label").returncode, 0)
        re_ = self.cli("gru_encoder", *small); self.assertEqual(re_.returncode, 0, re_.stderr + re_.stdout)
        se = self.summary(re_.stdout)
        self.assertEqual((se["cache_hit"], se["embedding_shape"]), (False, [180, 8])); self.assertEqual(se["inputs_run"], sg["run_id"])
        self.assertTrue(self.summary(self.cli("gru_inputs", *small).stdout)["cache_hit"]); self.assertTrue(self.summary(self.cli("gru_encoder", *small).stdout)["cache_hit"])
        self.assertIn("matches-current=YES", self.cli("gru_inputs", "--inspect-cache", *small).stdout); self.assertIn("matches-current=YES", self.cli("gru_encoder", "--inspect-cache", *small).stdout)
        # a changed input definition asks for new inputs instead of silently reusing stale ones
        rx = self.cli("gru_encoder", "--set", "three_modality.gru.scaling.winsor_pct=[1.0,99.0]", *small); self.assertNotEqual(rx.returncode, 0); self.assertIn("Run --stage gru_inputs first", rx.stderr)
        # the sequence grid must cover exactly the window
        rb = self.cli("gru_inputs", "--set", "three_modality.gru.n_bins=48"); self.assertEqual(rb.returncode, 2); self.assertIn("n_bins", rb.stderr)

    def test_a_window_that_disagrees_with_the_incident_cutoff_is_rejected_before_anything_runs(self):
        r = self.cli("tcd_features", "--set", "three_modality.incident_sepsis.cutoff_hours=48")
        self.assertEqual(r.returncode, 2); self.assertIn("window_minutes", r.stderr)

    def test_the_split_manifest_is_never_modified(self):
        before = self.manifest_path.read_bytes()
        self.cli("tcd_features"); self.assertEqual(self.manifest_path.read_bytes(), before)
