"""Tests for Fed-THOR_results_package/reproduce/final_comparison.py (Table 14: accuracy, AUROC, F1 in the format of FLICS's Tables IV and V)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Fed-THOR_results_package" / "reproduce"))
import final_comparison as fc  # noqa: E402

EXP = fc.M


def rec(arm, seed, acc, auc, f1, lo=0.80, hi=0.90, n=1000, prev=0.1):
    return {"arm": arm, "seed": seed, "test": {"accuracy": acc, "auroc": auc, "f1": f1, "n": n, "prevalence": prev}, "test_ci": {"auroc": {"point": auc, "lo": lo, "hi": hi}}}


def world(spec, arms, root=None):
    """results.json dicts per run folder for one block spec. arms: arm -> list of (seed, acc, auc, f1)."""
    runs = {}
    for a, vals in arms.items():
        run = spec["over"].get(a, spec["base"])
        runs.setdefault(run, {"experiments": {spec["exp"]: {"records": []}}})["experiments"][spec["exp"]]["records"] += [rec(a, s, acc, auc, f1) for s, acc, auc, f1 in vals]
    return runs


def loader(runs):
    def load(run):
        if run not in runs:
            raise FileNotFoundError(run)
        return runs[run]
    return load


def three(acc, auc, f1, wob=0.0):
    return [(42, acc, auc, f1), (43, acc + wob, auc + wob, f1 + wob), (44, acc - wob, auc - wob, f1 - wob)]


BASE_ARMS = {  # strict-like numbers; federated best on AUROC = GBDT, on accuracy = MLP, on F1 = LR; central best on AUROC = LightGBM, accuracy = LR, F1 = MLP
    "fedavg_logreg": three(0.880, 0.850, 0.470), "fedavg_mlp": three(0.890, 0.860, 0.440), "fed_gbdt": three(0.885, 0.875, 0.450),
    "logreg_central": [(42, 0.900, 0.855, 0.400)], "mlp_central": three(0.895, 0.865, 0.480), "lightgbm_central": three(0.870, 0.880, 0.460), "gbdt_central": three(0.875, 0.874, 0.465),
}


def block(key="t14a", arms=None):
    spec = fc.BLOCKS[key]
    h, rows, cap, lab, meta = fc.build_block(loader(world(spec, arms or BASE_ARMS)), spec)
    return rows, cap, meta


def cell(rows, name, col):
    return next(r for r in rows if r[0].startswith(name))[col]


class TestMarks(unittest.TestCase):
    def test_best_federated_is_bold_per_metric_independently(self):
        rows, _, _ = block()
        self.assertTrue(cell(rows, "Federated MLP", 1).startswith("**"))          # best federated accuracy
        self.assertTrue(cell(rows, "Federated GBDT", 2).startswith("**"))         # best federated AUROC
        self.assertTrue(cell(rows, "Federated LR", 3).startswith("**"))           # best federated F1
        for name, col in (("Federated LR", 1), ("Federated LR", 2), ("Federated MLP", 2), ("Federated MLP", 3), ("Federated GBDT", 1), ("Federated GBDT", 3)):
            self.assertFalse(cell(rows, name, col).startswith("**"), (name, col))

    def test_best_central_is_underlined_per_metric_and_never_bold(self):
        rows, _, _ = block()
        self.assertTrue(cell(rows, "Central LR", 1).startswith("<u>"))            # best central accuracy
        self.assertTrue(cell(rows, "Central LightGBM", 2).startswith("<u>"))      # best central AUROC
        self.assertTrue(cell(rows, "Central MLP", 3).startswith("<u>"))           # best central F1
        for r in rows:
            if r[0].startswith("Central"):
                self.assertFalse(any(c.startswith("**") for c in r[1:]), r)
            if r[0].startswith("Federated"):
                self.assertFalse(any(c.startswith("<u>") for c in r[1:]), r)

    def test_the_best_central_model_never_takes_the_federated_mark_even_when_it_beats_every_federated_model(self):
        rows, _, meta = block()
        self.assertEqual(meta["bold"]["auroc"], {"fed_gbdt"})                      # central LightGBM (0.880) is higher but belongs to the other group
        self.assertEqual(meta["underline"]["auroc"], {"lightgbm_central"})

    def test_ties_are_marked_on_every_tied_row(self):
        arms = dict(BASE_ARMS, fedavg_logreg=three(0.880, 0.875, 0.470))          # identical AUROC to the federated GBDT
        rows, _, meta = block(arms=arms)
        self.assertEqual(meta["bold"]["auroc"], {"fedavg_logreg", "fed_gbdt"})
        self.assertTrue(cell(rows, "Federated LR", 2).startswith("**") and cell(rows, "Federated GBDT", 2).startswith("**"))

    def test_a_difference_below_the_printed_precision_counts_as_a_tie(self):
        arms = dict(BASE_ARMS, fedavg_logreg=three(0.880, 0.87504, 0.470))        # 0.87504 prints as 0.8750 = the GBDT's 0.875
        self.assertEqual(block(arms=arms)[2]["bold"]["auroc"], {"fedavg_logreg", "fed_gbdt"})

    def test_best_arms_helper(self):
        self.assertEqual(fc.best_arms({"a": 0.5, "b": 0.7, "c": 0.7}), {"b", "c"})
        self.assertEqual(fc.best_arms({}), set())
        self.assertEqual(fc.best_arms({"a": 0.12344, "b": 0.12341}), {"a", "b"})

    def test_a_cell_cannot_be_both_marks(self):
        with self.assertRaises(ValueError):
            fc.wrap("x", bold=True, underline=True)

    def test_flics_rows_are_never_marked(self):
        rows, _, _ = block("t14c")
        flics = [r for r in rows if r[0].startswith("FLICS")]
        self.assertEqual(len(flics), 4)
        self.assertFalse(any(c.startswith(("**", "<u>")) for r in flics for c in r))


class TestCells(unittest.TestCase):
    def test_mean_and_sd_are_population_statistics_over_the_seeds(self):
        arms = dict(BASE_ARMS, fedavg_mlp=[(42, 0.80, 0.80, 0.30), (43, 0.82, 0.82, 0.32), (44, 0.85, 0.85, 0.35)])
        rows, _, _ = block(arms=arms)
        v = [0.80, 0.82, 0.85]
        self.assertIn(f"{np.mean(v):.4f} ± {np.std(v):.4f}", cell(rows, "Federated MLP", 1).replace("**", ""))     # ddof = 0, as results.json's own aggregate
        self.assertNotIn(f"{np.std(v, ddof=1):.4f}", cell(rows, "Federated MLP", 1))

    def test_single_seed_arm_says_so_instead_of_printing_a_zero_sd(self):
        rows, _, _ = block()
        c = cell(rows, "Central LR", 1)
        self.assertIn("(1 seed)", c); self.assertNotIn("±", c)
        self.assertIn("(1 seed)", cell(rows, "Central LR", 2))

    def test_auroc_interval_comes_from_the_seed_42_record_not_another_seed(self):
        spec = fc.BLOCKS["t14a"]
        runs = world(spec, BASE_ARMS)
        recs = runs[spec["over"]["fed_gbdt"]]["experiments"][EXP]["records"]
        for r in recs:
            r["test_ci"]["auroc"] = {"point": 0.875, "lo": 0.1 * (r["seed"] - 41), "hi": 0.2 * (r["seed"] - 41)}   # seed 42 -> [0.1, 0.2]
        rows = fc.build_block(loader(runs), spec)[1]
        self.assertIn("[0.1000, 0.2000]", cell(rows, "Federated GBDT", 2))

    def test_caption_reports_test_n_prevalence_and_the_all_negative_accuracy(self):
        _, cap, meta = block()
        self.assertEqual(meta["n_test"], 1000)
        self.assertIn("test n = 1,000", cap); self.assertIn("10.0%", cap); self.assertIn("accuracy 0.900", cap)
        self.assertIn("validation", cap); self.assertIn("FLICS tunes its threshold on its test set", cap)

    def test_flics_reference_rows_only_in_the_two_flics_scale_blocks(self):
        for k, n in (("t14a", 0), ("t14b", 0), ("t14c", 4), ("t14d", 4), ("t14e", 0), ("t14f", 0)):
            spec = fc.BLOCKS[k]
            arms = {a: v for a, v in BASE_ARMS.items()}
            rows = fc.build_block(loader(world(spec, arms)), spec)[1]
            self.assertEqual(sum(r[0].startswith("FLICS") for r in rows), n, k)

    def test_flics_transcription_matches_its_tables_iv_and_v(self):
        self.assertEqual(fc.FLICS["top3"][3], ("Centralized XGBoost", "0.91 (0.89–0.93)", "0.90 (0.87–0.93)", "0.65"))
        self.assertEqual(fc.FLICS["top10"][0], ("FedAvg LR", "0.89 (0.87–0.91)", "0.83 (0.79–0.87)", "0.59"))
        self.assertEqual(fc.FLICS["top10"][3][1:], ("0.90 (0.88–0.91)", "0.88 (0.85–0.91)", "0.63"))


class TestMissingAndMismatched(unittest.TestCase):
    def test_missing_optional_gbdt_arm_is_skipped_with_a_warning(self):
        arms = {a: v for a, v in BASE_ARMS.items() if a != "gbdt_central"}
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rows, _, meta = block(arms=arms)
        self.assertNotIn("gbdt_central", meta["arms"]); self.assertFalse(any(r[0].startswith("Central GBDT") for r in rows))
        self.assertIn("WARNING", err.getvalue())

    def test_missing_required_arm_is_an_error_not_a_silent_skip(self):
        arms = {a: v for a, v in BASE_ARMS.items() if a != "mlp_central"}
        spec = fc.BLOCKS["t14a"]
        with self.assertRaises(ValueError):
            fc.build_block(loader(world(spec, arms)), spec)

    def test_arms_scored_on_different_test_sets_are_refused(self):
        spec = fc.BLOCKS["t14a"]
        runs = world(spec, BASE_ARMS)
        for r in runs[spec["over"]["gbdt_central"]]["experiments"][EXP]["records"]:
            r["test"]["n"] = 999
        with self.assertRaises(ValueError):
            fc.build_block(loader(runs), spec)

    def test_tables_skips_a_block_whose_base_run_is_missing_and_keeps_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            spec = fc.BLOCKS["t14a"]
            for run, js in world(spec, BASE_ARMS).items():
                (Path(d) / run).mkdir(); (Path(d) / run / "results.json").write_text(json.dumps(js), encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                out = fc.tables(d)
            self.assertEqual(list(out), ["t14a"]); self.assertIn("skipped", err.getvalue())


class TestTex(unittest.TestCase):
    def test_marks_become_latex_commands(self):
        self.assertEqual(fc.to_tex("**0.8746 \\pm 0.0005**"), "\\textbf{0.8746 \\pm 0.0005}")
        self.assertEqual(fc.to_tex("<u>0.88</u> and plain"), "\\underline{0.88} and plain")
        self.assertEqual(fc.to_tex("no marks 0.5"), "no marks 0.5")


if __name__ == "__main__":
    unittest.main()
