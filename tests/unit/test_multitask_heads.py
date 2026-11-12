"""Tests for src/heads/multitask_heads.py (Phase 10).

Covers: correct output shapes (specifically the (N,) vs (N,1) squeeze
bug -- an extremely common, easy-to-miss shape-confusion class), the
single-patient edge case (where squeeze() without an explicit axis
would silently also collapse the batch dimension), architectural
independence of the two heads (no shared parameters beyond the input),
and align_labels_to_patient_ids()'s ID-based join -- including the
same "storage-order mismatch" regression style Phase 7's
test_multimodal_fusion.py established, plus the sepsis_cohort
validation and legitimate-NaN-vs-genuinely-missing-patient distinction.
"""

import unittest

import numpy as np
import pandas as pd
import torch

from src.heads.multitask_heads import MultiTaskHeads, align_labels_to_patient_ids


def _fixed_seed_model(in_dim: int = 16, head_hidden_dim: int = 8, **kwargs) -> MultiTaskHeads:
    torch.manual_seed(1)
    model = MultiTaskHeads(in_dim, head_hidden_dim=head_hidden_dim, **kwargs)
    model.eval()
    return model


class TestMultiTaskHeadsConstruction(unittest.TestCase):
    def test_raises_on_zero_in_dim(self):
        with self.assertRaises(ValueError):
            MultiTaskHeads(0, head_hidden_dim=8)

    def test_raises_on_zero_head_hidden_dim(self):
        with self.assertRaises(ValueError):
            MultiTaskHeads(16, head_hidden_dim=0)


class TestMultiTaskHeadsForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n, self.in_dim, self.head_hidden_dim = 12, 16, 8
        self.x = torch.randn(self.n, self.in_dim)

    def test_output_keys_present(self):
        model = _fixed_seed_model(self.in_dim, self.head_hidden_dim)
        with torch.no_grad():
            out = model(self.x)
        self.assertIn("mortality_logits", out)
        self.assertIn("sepsis_logits", out)

    def test_output_shape_is_1d_not_2d(self):
        """Direct regression target: Linear(hidden,1) naturally produces
        (N,1). Forgetting the explicit squeeze(-1) would silently
        return (N,1) here instead of (N,) -- a shape mismatch that
        broadcasts incorrectly (to (N,N)) rather than crashing when
        later compared against an (N,)-shaped label vector via e.g.
        BCEWithLogitsLoss. Proven by hand: removing the squeeze(-1)
        call from forward() and rerunning this test fails with
        out['mortality_logits'].shape == (12, 1), confirming the test
        actually catches the bug it targets."""
        model = _fixed_seed_model(self.in_dim, self.head_hidden_dim)
        with torch.no_grad():
            out = model(self.x)
        self.assertEqual(out["mortality_logits"].shape, (self.n,))
        self.assertEqual(out["sepsis_logits"].shape, (self.n,))
        self.assertEqual(out["mortality_logits"].dim(), 1)
        self.assertEqual(out["sepsis_logits"].dim(), 1)

    def test_single_patient_shape_not_collapsed(self):
        """squeeze() (no axis) on a single patient's (1,1) output would
        ALSO collapse the batch dimension down to a 0-d scalar --
        squeeze(-1) must not do this. A 0-d tensor would silently break
        any downstream code assuming an indexable (N,) vector."""
        model = _fixed_seed_model(self.in_dim, self.head_hidden_dim)
        x_single = torch.randn(1, self.in_dim)
        with torch.no_grad():
            out = model(x_single)
        self.assertEqual(out["mortality_logits"].shape, (1,))
        self.assertEqual(out["sepsis_logits"].shape, (1,))

    def test_empty_input_does_not_crash(self):
        """Confirmed to occur in real data -- at least one real
        hospital has zero patients (see Phase 8/9's real-data
        verification). Phase 10 runs one global (non-per-hospital)
        forward pass, but the overall cohort-level input could still
        legitimately be empty in a degenerate config (e.g. an
        all-hospitals-excluded subset)."""
        model = _fixed_seed_model(self.in_dim, self.head_hidden_dim)
        x_empty = torch.empty((0, self.in_dim))
        with torch.no_grad():
            out = model(x_empty)
        self.assertEqual(out["mortality_logits"].shape, (0,))
        self.assertEqual(out["sepsis_logits"].shape, (0,))

    def test_output_has_no_nan_or_inf(self):
        model = _fixed_seed_model(self.in_dim, self.head_hidden_dim)
        with torch.no_grad():
            out = model(self.x)
        for key in ("mortality_logits", "sepsis_logits"):
            self.assertFalse(torch.isnan(out[key]).any())
            self.assertFalse(torch.isinf(out[key]).any())


class TestHeadsAreArchitecturallyIndependent(unittest.TestCase):
    """Direct regression target for the design decision confirmed with
    the user: independent MLP heads, NOT a shared bottleneck. Proves
    the two heads genuinely don't share parameters -- editing one
    head's weights must leave the other head's output completely
    unchanged."""

    def setUp(self):
        torch.manual_seed(0)
        self.n, self.in_dim, self.head_hidden_dim = 10, 16, 8
        self.x = torch.randn(self.n, self.in_dim)

    def test_no_shared_parameter_tensors(self):
        model = _fixed_seed_model(self.in_dim, self.head_hidden_dim)
        mortality_param_ids = {id(p) for p in model.mortality_head.parameters()}
        sepsis_param_ids = {id(p) for p in model.sepsis_head.parameters()}
        self.assertTrue(
            mortality_param_ids.isdisjoint(sepsis_param_ids),
            "mortality_head and sepsis_head share at least one parameter tensor -- "
            "they should be fully independent per the confirmed design.",
        )

    def test_editing_mortality_head_does_not_change_sepsis_output(self):
        model = _fixed_seed_model(self.in_dim, self.head_hidden_dim)
        with torch.no_grad():
            sepsis_before = model(self.x)["sepsis_logits"].clone()

        with torch.no_grad():
            for p in model.mortality_head.parameters():
                p.add_(1.0)  # deliberately corrupt mortality_head's weights

        with torch.no_grad():
            sepsis_after = model(self.x)["sepsis_logits"]

        self.assertTrue(
            torch.allclose(sepsis_before, sepsis_after),
            "sepsis_logits changed after only mortality_head's weights were edited -- "
            "the heads are not architecturally independent.",
        )


class TestAlignLabelsToPatientIds(unittest.TestCase):
    def setUp(self):
        self.cohort_patients = pd.DataFrame({"patientunitstayid": [10, 20, 30]})
        self.mortality = pd.Series([1.0, 0.0, np.nan])  # patient 30's outcome unknown
        self.sepsis_df = pd.DataFrame({
            "patientunitstayid": [10, 20, 30],
            "sepsis_broad": [True, False, True],
            "sepsis_strict": [False, False, True],
        })

    def test_basic_alignment_broad(self):
        result = align_labels_to_patient_ids(
            [10, 20, 30], self.cohort_patients, self.mortality, self.sepsis_df, "broad",
        )
        np.testing.assert_array_equal(result["mortality"], [1.0, 0.0, np.nan])
        np.testing.assert_array_equal(result["sepsis"], [True, False, True])
        self.assertEqual(result["n_mortality_known"], 2)
        self.assertEqual(result["n_mortality_positive"], 1)
        self.assertEqual(result["n_mortality_negative"], 1)
        self.assertEqual(result["n_mortality_unknown"], 1)
        self.assertEqual(result["n_sepsis_positive"], 2)
        self.assertEqual(result["n_sepsis_negative"], 1)

    def test_strict_cohort_selects_different_column(self):
        result = align_labels_to_patient_ids(
            [10, 20, 30], self.cohort_patients, self.mortality, self.sepsis_df, "strict",
        )
        np.testing.assert_array_equal(result["sepsis"], [False, False, True])
        self.assertEqual(result["n_sepsis_positive"], 1)

    def test_reorders_to_match_patient_ids_order_not_storage_order(self):
        """Mirrors test_multimodal_fusion.py's
        test_misaligned_storage_order_still_fuses_correctly: the GNN
        embeddings' patient order need not match cohort_patients'/
        sepsis_df's own storage order (Phase 9 groups by hospital, not
        Phase 3's cohort order). A positional (non-ID-based) join
        would silently attach the wrong patient's label."""
        result = align_labels_to_patient_ids(
            [30, 10, 20], self.cohort_patients, self.mortality, self.sepsis_df, "broad",
        )
        np.testing.assert_array_equal(result["mortality"], [np.nan, 1.0, 0.0])
        np.testing.assert_array_equal(result["sepsis"], [True, True, False])

    def test_raises_on_invalid_sepsis_cohort(self):
        with self.assertRaises(ValueError) as ctx:
            align_labels_to_patient_ids(
                [10, 20, 30], self.cohort_patients, self.mortality, self.sepsis_df, "moderate",
            )
        self.assertIn("moderate", str(ctx.exception))

    def test_raises_on_none_sepsis_cohort(self):
        """Covers an unset config key (cfg.get() returning None) --
        must fail the same way as any other invalid value, never
        silently pick a default here."""
        with self.assertRaises(ValueError):
            align_labels_to_patient_ids(
                [10, 20, 30], self.cohort_patients, self.mortality, self.sepsis_df, None,
            )

    def test_raises_on_patient_missing_from_cohort_patients(self):
        """cohort_patients is authoritative -- absence there is a real
        problem (ID corruption upstream), unlike absence from
        sepsis_df (see test_absent_from_sepsis_df_becomes_unknown_not_an_error)."""
        with self.assertRaises(ValueError) as ctx:
            align_labels_to_patient_ids(
                [10, 20, 999], self.cohort_patients, self.mortality, self.sepsis_df, "broad",
            )
        self.assertIn("999", str(ctx.exception))

    def test_absent_from_sepsis_df_becomes_unknown_not_an_error(self):
        """Root-caused via this phase's own integration test against
        mock data (not assumed): compute_infection_proxy()'s output
        only covers patients with >=1 medication/microLab row, and
        compute_sofa_like_scores()'s output only covers patients with
        an apachePredVar row -- a patient with literally none of those
        (confirmed: mock patient 141029, zero rows in all three) is
        entirely absent from sepsis_df. This is the SAME phenomenon
        compute_sepsis_labels()'s own inline comment already documents
        for the one-sided case ("a real, common situation in eICU"),
        just on both sides at once -- so it gets the same treatment as
        mortality's own unrecognized-status case: NaN (unknown), not
        an error and not silently coded as "no sepsis"."""
        sepsis_df_missing_30 = self.sepsis_df[self.sepsis_df["patientunitstayid"] != 30]
        result = align_labels_to_patient_ids(
            [10, 20, 30], self.cohort_patients, self.mortality, sepsis_df_missing_30, "broad",
        )
        self.assertTrue(np.isnan(result["sepsis"][2]))
        self.assertEqual(result["n_sepsis_known"], 2)
        self.assertEqual(result["n_sepsis_unknown"], 1)
        # patients 10 and 20 are unaffected
        np.testing.assert_array_equal(result["sepsis"][:2], [1.0, 0.0])

    def test_does_not_raise_on_legitimate_unknown_mortality(self):
        """Patient 30's mortality is NaN (legitimately unknown, e.g. an
        unrecognized hospitaldischargestatus value) but IS present in
        both tables -- must NOT be treated as a missing-patient error."""
        result = align_labels_to_patient_ids(
            [10, 20, 30], self.cohort_patients, self.mortality, self.sepsis_df, "broad",
        )
        self.assertTrue(np.isnan(result["mortality"][2]))

    def test_raises_on_duplicate_patient_id_in_cohort_patients(self):
        dup_cohort = pd.DataFrame({"patientunitstayid": [10, 10, 30]})
        dup_mortality = pd.Series([1.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            align_labels_to_patient_ids([10, 30], dup_cohort, dup_mortality, self.sepsis_df, "broad")

    def test_mortality_aligned_by_index_not_position(self):
        """cohort_patients and mortality are expected to share a
        pandas INDEX (per run_phase3_cohort_construction()'s own
        docstring), not necessarily be handed in the same row order as
        each other by every caller. Construct them with a shuffled,
        non-default index and confirm the join still lands on the
        right patient."""
        cohort_shuffled = pd.DataFrame(
            {"patientunitstayid": [30, 10, 20]}, index=[2, 0, 1],
        )
        mortality_by_index = pd.Series({0: 1.0, 1: 0.0, 2: np.nan})  # index 0->patient10, 1->patient20, 2->patient30
        result = align_labels_to_patient_ids(
            [10, 20, 30], cohort_shuffled, mortality_by_index, self.sepsis_df, "broad",
        )
        np.testing.assert_array_equal(result["mortality"], [1.0, 0.0, np.nan])


if __name__ == "__main__":
    unittest.main()
