"""Integration tests for the Phase 4/5 multi-architecture capability
added 2026-09-06 (user request): `modalities.timeseries.encoder`
(lstm | rnn | gru | transformer) and `modalities.tabular.encoder`
(autoencoder | mlp).

The single most important property under test: selecting a
NON-DEFAULT architecture must NEVER touch the canonical
timeseries_embeddings.npz / tabular_embeddings.npz files that Phase
7+ depend on -- those files are (and must remain) exclusively LSTM's
and the Autoencoder's own output, byte-for-byte unchanged from before
this capability existed. See tests/integration/test_phase9_pipeline.py
and test_phase10_pipeline.py, re-run completely unmodified after this
change, for the complementary proof that the DEFAULT path's behavior
is unaffected all the way through the pipeline.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_mock_eicu import generate  # noqa: E402
from run_pipeline import (  # noqa: E402
    run_phase1_ingestion,
    run_phase2_hospital_partition,
    run_phase3_cohort_construction,
    run_phase4_timeseries_modality,
    run_phase5_tabular_modality,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import close_logging  # noqa: E402


class TestModalityArchitectureOptions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tmpdir.name) / "mock_eicu"
        self.output_dir = Path(self.tmpdir.name) / "outputs"
        generate(str(self.data_root), n_hospitals=4, n_patients_per_hospital=25, seed=3)

        self._old_env = os.environ.get("EICU_DATA_ROOT")
        os.environ["EICU_DATA_ROOT"] = str(self.data_root)

        self.cfg = load_config(str(REPO_ROOT / "configs" / "debug.yaml"))
        self.cfg.raw["run"]["output_dir"] = str(self.output_dir)
        self.cfg.raw["run"]["log_dir"] = str(self.output_dir / "logs")
        self.cfg.raw["run"]["cache_dir"] = str(self.output_dir / "cache")
        self.cfg.raw["cohort"]["patient_subset"] = 15

    def tearDown(self):
        close_logging()
        self.tmpdir.cleanup()
        if self._old_env is not None:
            os.environ["EICU_DATA_ROOT"] = self._old_env
        else:
            os.environ.pop("EICU_DATA_ROOT", None)

    def _get_to_phase3(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)
        return phase3_result, partition_result

    def _skip_if_no_torch(self, phase_result):
        if not phase_result.get("torch_available"):
            self.skipTest("torch not available in this environment")

    # ---- Timeseries (Phase 4) ----

    def test_default_lstm_still_writes_canonical_filename(self):
        phase3_result, partition_result = self._get_to_phase3()
        result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        self._skip_if_no_torch(result)

        self.assertEqual(result["encoder"], "lstm")
        self.assertTrue(result["embeddings_path"].endswith("timeseries_embeddings.npz"))
        self.assertTrue(Path(result["embeddings_path"]).exists())

    def test_rnn_gru_transformer_each_write_their_own_distinctly_named_file(self):
        phase3_result, partition_result = self._get_to_phase3()

        for encoder in ("rnn", "gru", "transformer"):
            self.cfg.raw["modalities"]["timeseries"]["encoder"] = encoder
            result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
            self._skip_if_no_torch(result)

            self.assertEqual(result["encoder"], encoder)
            expected_name = f"timeseries_embeddings_{encoder}.npz"
            self.assertTrue(
                result["embeddings_path"].endswith(expected_name),
                f"encoder={encoder} wrote to {result['embeddings_path']!r}, expected it to end with {expected_name!r}",
            )
            self.assertTrue(Path(result["embeddings_path"]).exists())

    def test_canonical_lstm_file_is_byte_for_byte_untouched_by_alternate_architecture_runs(self):
        """THE core safety guarantee under test: running Phase 4 with
        a non-default encoder must never modify timeseries_embeddings.npz
        in any way -- not overwrite it, not touch its mtime, nothing."""
        phase3_result, partition_result = self._get_to_phase3()

        lstm_result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        self._skip_if_no_torch(lstm_result)
        canonical_path = Path(lstm_result["embeddings_path"])
        original_bytes = canonical_path.read_bytes()
        original_mtime = canonical_path.stat().st_mtime_ns

        for encoder in ("rnn", "gru", "transformer"):
            self.cfg.raw["modalities"]["timeseries"]["encoder"] = encoder
            run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)

            self.assertEqual(
                canonical_path.read_bytes(), original_bytes,
                f"timeseries_embeddings.npz's CONTENT changed after running encoder={encoder}.",
            )
            self.assertEqual(
                canonical_path.stat().st_mtime_ns, original_mtime,
                f"timeseries_embeddings.npz's mtime changed after running encoder={encoder} -- "
                f"it was rewritten even if the content happened to match.",
            )

    def test_all_timeseries_encoders_produce_the_configured_hidden_dim(self):
        """Confirms the shape-contract each new encoder promises to
        match LSTM's own output width -- this is what keeps Phase 7's
        fusion (and everything after it) oblivious to which
        architecture produced the embeddings it's fusing."""
        phase3_result, partition_result = self._get_to_phase3()
        configured_hidden_dim = self.cfg.get("modalities.timeseries.hidden_dim", 128)

        for encoder in ("lstm", "rnn", "gru", "transformer"):
            self.cfg.raw["modalities"]["timeseries"]["encoder"] = encoder
            result = run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
            self._skip_if_no_torch(result)

            data = np.load(result["embeddings_path"])
            self.assertEqual(
                data["embeddings"].shape[1], configured_hidden_dim,
                f"encoder={encoder} produced embedding width {data['embeddings'].shape[1]}, "
                f"expected {configured_hidden_dim}.",
            )

    def test_raises_clearly_on_invalid_timeseries_encoder(self):
        phase3_result, partition_result = self._get_to_phase3()
        self.cfg.raw["modalities"]["timeseries"]["encoder"] = "bilstm"
        with self.assertRaises(ValueError) as ctx:
            run_phase4_timeseries_modality(self.cfg, phase3_result, partition_result)
        self.assertIn("bilstm", str(ctx.exception))

    # ---- Tabular (Phase 5) ----

    def test_default_autoencoder_still_writes_canonical_filename(self):
        phase1_summary, _ = None, None
        phase1_summary = run_phase1_ingestion(self.cfg)
        _, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)

        result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        self._skip_if_no_torch(result)

        self.assertEqual(result["encoder"], "autoencoder")
        self.assertTrue(result["embeddings_path"].endswith("tabular_embeddings.npz"))
        self.assertTrue(Path(result["embeddings_path"]).exists())

    def test_mlp_writes_its_own_distinctly_named_file(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)

        self.cfg.raw["modalities"]["tabular"]["encoder"] = "mlp"
        result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        self._skip_if_no_torch(result)

        self.assertEqual(result["encoder"], "mlp")
        self.assertTrue(result["embeddings_path"].endswith("tabular_embeddings_mlp.npz"))
        self.assertTrue(Path(result["embeddings_path"]).exists())

    def test_canonical_autoencoder_file_is_byte_for_byte_untouched_by_mlp_run(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)

        ae_result = run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        self._skip_if_no_torch(ae_result)
        canonical_path = Path(ae_result["embeddings_path"])
        original_bytes = canonical_path.read_bytes()
        original_mtime = canonical_path.stat().st_mtime_ns

        self.cfg.raw["modalities"]["tabular"]["encoder"] = "mlp"
        run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)

        self.assertEqual(canonical_path.read_bytes(), original_bytes)
        self.assertEqual(canonical_path.stat().st_mtime_ns, original_mtime)

    def test_raises_clearly_on_invalid_tabular_encoder(self):
        phase1_summary = run_phase1_ingestion(self.cfg)
        partition_result, patient_df, _ = run_phase2_hospital_partition(self.cfg, phase1_summary)
        phase3_result = run_phase3_cohort_construction(self.cfg, phase1_summary, partition_result, patient_df)

        self.cfg.raw["modalities"]["tabular"]["encoder"] = "vae"
        with self.assertRaises(ValueError) as ctx:
            run_phase5_tabular_modality(self.cfg, phase1_summary, phase3_result, partition_result)
        self.assertIn("vae", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
