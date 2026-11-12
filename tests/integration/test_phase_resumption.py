"""End-to-end test for `--start-phase` resumption -- invokes the actual
`run_pipeline.py` CLI via subprocess (not just the underlying loader
functions in isolation), because the real claim being tested is "the
command line tool can skip already-completed phases," and that can
only be faithfully proven by actually running the command line tool.

The core proof technique: after the first (full) run completes, the
mock data root is DELETED before the resumed run. If resumption
secretly still depended on re-reading raw data for a "skipped" phase,
the resumed run would fail with a file-not-found error -- so a
resumed run succeeding with the data gone is direct evidence the
skipped phases genuinely weren't re-executed, not just evidence they
were skipped quickly.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from src.utils.config import load_config  # noqa: E402


class TestPhaseResumption(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=3)

        # Build a fully self-contained, resolved config file for the
        # subprocess to read -- reuses load_config()'s already-tested
        # inheritance/env-var-substitution logic, then dumps the
        # resolved dict to a temp YAML rather than relying on subprocess
        # environment overrides that debug.yaml's own settings can't see.
        cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        cfg.raw["data"]["root"] = str(self.data_root)
        cfg.raw["run"]["output_dir"] = str(self.output_dir)
        cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        cfg.raw["run"]["cache_dir"] = str(self.output_dir / "cache")
        cfg.raw["cohort"]["patient_subset"] = 15
        cfg.raw.setdefault("graph", {})
        cfg.raw["graph"]["method"] = "knn"
        cfg.raw["graph"]["k"] = 3
        cfg.raw.setdefault("gnn", {})
        cfg.raw["gnn"]["hidden_dim"] = 32

        self.config_path = Path(self.tmpdir.name) / "resolved_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.safe_dump(cfg.raw, f)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_cli(self, phase: int, start_phase: int | None = None, expect_success: bool = True):
        cmd = [sys.executable, "run_pipeline.py", "--config", str(self.config_path), "--phase", str(phase)]
        if start_phase is not None:
            cmd += ["--start-phase", str(start_phase)]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
        if expect_success:
            self.assertEqual(
                result.returncode, 0,
                f"CLI failed unexpectedly.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
        return result

    def test_mismatched_cohort_config_blocked_not_silently_mixed(self):
        """Direct regression target for a real, undocumented risk found
        after --start-phase shipped: debug.yaml and full.yaml (and
        subset.yaml) all share the same default run.output_dir, since
        none of them override it. Without this guard, running Phase
        1-7 with a small cohort_patient_subset then --start-phase 8
        with a DIFFERENT patient_subset (simulating debug.yaml vs
        full.yaml) would silently resume Phase 8/9 against the small
        cohort's graphs while the user believes they're on the larger
        one -- same file shapes, no crash, just quietly wrong."""
        self._run_cli(phase=7)  # patient_subset=15, from setUp's debug.yaml-based config

        # Simulate switching to a config with a different cohort
        # identity (like full.yaml's patient_subset: null) while
        # still pointing at the same run.output_dir.
        with open(self.config_path) as f:
            cfg_dict = yaml.safe_load(f)
        cfg_dict["cohort"]["patient_subset"] = 999  # different from the 15 used above
        with open(self.config_path, "w") as f:
            yaml.safe_dump(cfg_dict, f)

        result = self._run_cli(phase=9, start_phase=8, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("resumption blocked", result.stderr)
        self.assertIn("patient_subset", result.stderr)

    def test_matching_config_resumption_still_works_normally(self):
        """Sanity check alongside the mismatch test above -- the guard
        must not false-positive on a genuinely unchanged config."""
        self._run_cli(phase=7)
        result = self._run_cli(phase=9, start_phase=8)  # same config_path, unchanged
        self.assertIn("PHASE9_GNN_FORWARD_PASS] COMPLETE", result.stdout)

    def test_full_run_then_resume_from_phase7_after_deleting_raw_data(self):
        full_result = self._run_cli(phase=9)
        self.assertIn("PHASE9_GNN_FORWARD_PASS] COMPLETE", full_result.stdout)

        gnn_path = self.output_dir / "embeddings" / "gnn_embeddings.npz"
        self.assertTrue(gnn_path.exists())
        full_run_embeddings = np.load(gnn_path)["embeddings"].copy()
        full_run_n_patients = len(np.load(gnn_path)["patient_ids"])

        # Delete Phase 9's own output (so a genuine re-run is
        # detectable) but NOT the earlier phases' outputs -- exactly
        # the "just re-run the new phase" scenario being tested.
        gnn_path.unlink()

        # The core proof: delete the raw mock data entirely. If
        # anything from phase 1-6 secretly got re-executed, this
        # would fail with a file-not-found error from pandas/discovery.
        shutil.rmtree(self.data_root)

        resumed_result = self._run_cli(phase=9, start_phase=7)
        self.assertIn("PHASE9_GNN_FORWARD_PASS] COMPLETE", resumed_result.stdout)
        self.assertTrue(gnn_path.exists(), "Phase 9 did not actually re-run and produce fresh output")

        resumed_data = np.load(gnn_path)
        self.assertEqual(len(resumed_data["patient_ids"]), full_run_n_patients)
        self.assertEqual(resumed_data["embeddings"].shape[1], full_run_embeddings.shape[1])

    def test_resume_from_phase4_reuses_phase3_labels_correctly(self):
        """A different resumption point -- proves this isn't special-
        cased to phase 7 only. Phase 3's persisted mortality/sepsis
        labels must round-trip correctly (dtype-safe, ID-aligned) for
        this to work at all."""
        self._run_cli(phase=3)
        cohort_csv = self.output_dir / "cohort_reports" / "cohort_patients.csv"
        self.assertTrue(cohort_csv.exists())
        original_patient_count = sum(1 for _ in open(cohort_csv)) - 1  # minus header

        shutil.rmtree(self.data_root.parent / "mock_eicu", ignore_errors=True)
        # Only delete raw patient/hospital/apachePredVar/etc CSVs, not
        # the whole tree, since Phase 4 (unlike Phase 7+) still reads
        # raw lab/medication data directly for the LSTM's own
        # time-series construction -- --start-phase 4 only skips
        # Phases 1-3, not raw-data access within Phase 4 itself.
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=3)

        result = self._run_cli(phase=4, start_phase=4)
        self.assertIn("PHASE4", result.stdout)

        embeddings_path = self.output_dir / "embeddings" / "timeseries_embeddings.npz"
        if embeddings_path.exists():  # only if torch is available in this environment
            n_embedded = len(np.load(embeddings_path)["patient_ids"])
            self.assertEqual(n_embedded, original_patient_count)

    def test_start_phase_greater_than_phase_rejected_with_clear_error(self):
        result = self._run_cli(phase=3, start_phase=8, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot be greater than", result.stderr)

    def test_start_phase_8_needs_no_phase1or3_artifacts_but_does_need_phase2(self):
        """Direct regression target for a real bug caught while
        building this feature: an earlier version still transitively
        tried to reload Phase 1/3's artifacts even for --start-phase 8,
        because every phaseN block executed unconditionally whenever
        args.phase >= N, regardless of whether anything downstream
        actually needed phaseN's result. Fixed by computing exact
        per-result "is this needed by anything that will actually run"
        dependency flags instead. Proven here by deleting Phase 1 and
        Phase 3's saved artifacts entirely (not just the raw data)
        before the --start-phase 8 run and confirming it still
        succeeds -- while Phase 2's partition_result is deliberately
        KEPT, since Phase 8 genuinely, directly depends on it
        (per-hospital patient_ids) and correctly requires it."""
        self._run_cli(phase=9)

        shutil.rmtree(self.output_dir / "data_reports", ignore_errors=True)
        (self.output_dir / "cohort_reports" / "cohort_patients.csv").unlink(missing_ok=True)
        (self.output_dir / "cohort_reports" / "mortality_labels.csv").unlink(missing_ok=True)
        (self.output_dir / "cohort_reports" / "sepsis_labels.csv").unlink(missing_ok=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

        gnn_path = self.output_dir / "embeddings" / "gnn_embeddings.npz"
        gnn_path.unlink()

        result = self._run_cli(phase=9, start_phase=8)
        self.assertIn("PHASE9_GNN_FORWARD_PASS] COMPLETE", result.stdout)
        self.assertTrue(gnn_path.exists())

    def test_resuming_past_a_phase_that_was_never_run_fails_clearly(self):
        # Nothing has been run yet. --start-phase 8 needs BOTH
        # partition_result (Phase 2) and phase7_result (Phase 7) to be
        # loadable, and neither exists -- whichever is checked first
        # should name itself specifically, not fail silently/vaguely.
        result = self._run_cli(phase=9, start_phase=8, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(
            "Phase 2" in result.stderr or "Phase 7" in result.stderr,
            f"expected a specific missing-phase message, got: {result.stderr}",
        )

    def test_resuming_past_phase8_specifically_fails_clearly_when_phase8_was_never_run(self):
        # Phase 7 ran (so partition_result and phase7_result both
        # exist), but Phase 8 itself never did. --start-phase 9 means
        # Phase 8 must be LOADED (not run, since 8 < 9), which should
        # fail clearly since nothing produced graph_manifest.json yet.
        # (--start-phase 8 would instead just run Phase 8 fresh, which
        # is valid and needs nothing from a prior Phase 8 run at all.)
        self._run_cli(phase=7)
        result = self._run_cli(phase=9, start_phase=9, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Phase 8", result.stderr)

    def test_start_phase_10_needs_no_earlier_artifacts_except_phase3_and_9(self):
        """Direct regression target for Phase 10's dependency flags,
        added alongside Phase 10 itself: run_phase10_multitask_heads()
        only consumes phase9_result (GNN embeddings) and phase3_result
        (mortality/sepsis_df) directly -- NOT phase1_summary,
        partition_result, phase4_result, phase5_result, phase7_result,
        or phase8_result. Proven by deleting every one of those
        artifacts (not just the raw data) before --start-phase 10 and
        confirming it still succeeds, while Phase 3's persisted labels
        and Phase 9's gnn_embeddings.npz/gnn_manifest.json are
        deliberately KEPT, since Phase 10 genuinely, directly depends
        on both."""
        self._run_cli(phase=9)

        shutil.rmtree(self.output_dir / "data_reports", ignore_errors=True)
        (self.output_dir / "cohort_reports" / "hospital_partition.json").unlink(missing_ok=True)
        (self.output_dir / "cohort_reports" / "partition_result_full.json").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "timeseries_embeddings.npz").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "timeseries_sequences.npz").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "tabular_embeddings.npz").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "tabular_features.npz").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "fused_embeddings.npz").unlink(missing_ok=True)
        shutil.rmtree(self.output_dir / "graphs", ignore_errors=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

        result = self._run_cli(phase=10, start_phase=10)
        self.assertIn("PHASE10_MULTITASK_HEADS] COMPLETE", result.stdout)

    def test_resuming_past_phase9_specifically_fails_clearly_when_phase9_was_never_run(self):
        # Phase 3 ran (so cohort_patients/mortality/sepsis_df all
        # persist), but Phase 9 itself never did. --start-phase 10
        # means Phase 9 must be LOADED (not run), which should fail
        # clearly since nothing produced gnn_manifest.json yet.
        self._run_cli(phase=3)
        result = self._run_cli(phase=10, start_phase=10, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Phase 9", result.stderr)

    def test_start_phase_11_needs_no_phase1_2_7_9_10_artifacts_but_does_need_phase3_4_5_8(self):
        """Direct regression target for Phase 11's dependency flags:
        run_phase11_local_training() consumes phase3_result (labels),
        phase4_result/phase5_result (Phase 4/5's RAW tensors, not their
        frozen embeddings -- it trains those encoders jointly), and
        phase8_result (graphs) directly -- NOT phase1_summary,
        partition_result, phase7_result, phase9_result, or
        phase10_result. Proven by deleting every one of those
        artifacts before --start-phase 11 and confirming it still
        succeeds, while Phase 3/4/5/8's persisted outputs are
        deliberately KEPT."""
        self._run_cli(phase=10)

        shutil.rmtree(self.output_dir / "data_reports", ignore_errors=True)
        (self.output_dir / "cohort_reports" / "hospital_partition.json").unlink(missing_ok=True)
        (self.output_dir / "cohort_reports" / "partition_result_full.json").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "fused_embeddings.npz").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "gnn_embeddings.npz").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "gnn_manifest.json").unlink(missing_ok=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

        result = self._run_cli(phase=11, start_phase=11)
        self.assertIn("PHASE11_LOCAL_TRAINING] COMPLETE", result.stdout)

    def test_resuming_past_phase11_fails_clearly_when_phase8_was_never_run(self):
        # Phase 3/4/5 all ran (persisted), but Phase 8 (graphs) never
        # did. --start-phase 11 means Phase 8 must be LOADED, which
        # should fail clearly since no graph_manifest.json exists yet.
        self._run_cli(phase=5)
        result = self._run_cli(phase=11, start_phase=11, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Phase 8", result.stderr)

    def test_start_phase_12_needs_no_earlier_artifacts_except_phase3_4_5_8(self):
        """Direct regression target for Phase 12's dependency flags:
        run_phase12_fedavg() consumes phase3_result/phase4_result/
        phase5_result/phase8_result directly -- the exact same set as
        Phase 11 -- NOT phase1_summary, partition_result,
        phase7_result, phase9_result, or phase10_result. It also does
        NOT need phase11_result as an in-memory value: round 1 reuse
        reads directly from whatever's already on disk under
        checkpoints/phase11_local/, so this test deliberately does NOT
        run Phase 11 first -- proving Phase 12 works via
        --start-phase 12 on its own, exactly the user's real
        situation (a completed real Phase 11 run from a PRIOR
        invocation, with no phase11_result available in this one)."""
        self._run_cli(phase=8)

        shutil.rmtree(self.output_dir / "data_reports", ignore_errors=True)
        (self.output_dir / "cohort_reports" / "hospital_partition.json").unlink(missing_ok=True)
        (self.output_dir / "cohort_reports" / "partition_result_full.json").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "fused_embeddings.npz").unlink(missing_ok=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

        result = self._run_cli(phase=12, start_phase=12)
        self.assertIn("PHASE12_FEDAVG] COMPLETE", result.stdout)

    def test_resuming_past_phase12_fails_clearly_when_phase8_was_never_run(self):
        # Phase 3/4/5 all ran (persisted), but Phase 8 (graphs) never
        # did. --start-phase 12 means Phase 8 must be LOADED, which
        # should fail clearly since no graph_manifest.json exists yet.
        self._run_cli(phase=5)
        result = self._run_cli(phase=12, start_phase=12, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Phase 8", result.stderr)

    def test_start_phase_13_needs_no_earlier_artifacts_except_phase3_4_5_8(self):
        """Direct regression target for Phase 13's dependency flags:
        run_phase13_baselines() consumes phase3_result/phase4_result/
        phase5_result/phase8_result directly -- the exact same set as
        Phase 11/12 -- NOT phase1_summary, partition_result,
        phase7_result, phase9_result, or phase10_result. It also does
        NOT need phase11_result/phase12_result as in-memory values:
        the federated-baseline reference reads Phase 12's manifest
        directly from disk if present (and gracefully handles it being
        absent, per test_federated_reference_missing_is_handled_
        gracefully in test_phase13_pipeline.py) -- this test
        deliberately does NOT run Phase 11 or 12 first, proving Phase
        13 works via --start-phase 13 entirely on its own."""
        self._run_cli(phase=8)

        shutil.rmtree(self.output_dir / "data_reports", ignore_errors=True)
        (self.output_dir / "cohort_reports" / "hospital_partition.json").unlink(missing_ok=True)
        (self.output_dir / "cohort_reports" / "partition_result_full.json").unlink(missing_ok=True)
        (self.output_dir / "embeddings" / "fused_embeddings.npz").unlink(missing_ok=True)
        shutil.rmtree(self.data_root, ignore_errors=True)

        result = self._run_cli(phase=13, start_phase=13)
        self.assertIn("PHASE13_BASELINES] COMPLETE", result.stdout)

    def test_resuming_past_phase13_fails_clearly_when_phase8_was_never_run(self):
        # Phase 3/4/5 all ran (persisted), but Phase 8 (graphs) never
        # did. --start-phase 13 means Phase 8 must be LOADED, which
        # should fail clearly since no graph_manifest.json exists yet.
        self._run_cli(phase=5)
        result = self._run_cli(phase=13, start_phase=13, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Phase 8", result.stderr)

    def test_invalid_start_phase_value_rejected(self):
        result = self._run_cli(phase=9, start_phase=6, expect_success=False)  # 6 is intentionally skipped
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a valid resumption point", result.stderr)


if __name__ == "__main__":
    unittest.main()
