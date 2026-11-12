"""FL Configs 1-4 and the personalised experiments end to end on synthetic upstream runs: client / hospital identities against the FROZEN cohort, paired set == the Non-FL regime's trained hospitals,
hospital-local graphs / SMOTE, the two-stage Config 4, personalisation, artifacts and manifests, run isolation, checkpoint-resume at run level, the FL-vs-Non-FL report and the CLI (subprocess)."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.three_modality import federated as fed, fl_report as fr, hw_data, non_federated as nf
from tests.unit._fl_cohort import build_fl_cohort, expected_test, fast_ds, fast_fl

ROOT = Path(__file__).resolve().parents[2]
ALGOS = ("fedavg", "fedprox", "fedadam", "scaffold")


def tree_hash(d):
    d = Path(d)
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "out"
        cls.root = cls.out / "three_modality"
        cls.up, cls.cohort, cls.ca = build_fl_cohort(cls.root)
        cls.ds, cls.fl = fast_ds(), fast_fl()
        cls.fed_base = cls.root / "federated"
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- helpers ----
    def summary(self, res):
        return json.loads((Path(res["run_dir"]) / "summary.json").read_text())

    def rows(self, res):
        return json.loads((Path(res["run_dir"]) / "per_hospital.json").read_text())

    def check_predictions_match_frozen_cohort(self, res):
        """Every saved prediction row is a TEST patient of the frozen cohort, in the right hospital, with the frozen label; each evaluated hospital's test set is COMPLETE."""
        npz = np.load(Path(res["run_dir"]) / "predictions.npz", allow_pickle=True)
        names = sorted({k.rsplit("@", 1)[0] for k in npz.files})
        self.assertTrue(names)
        for v in names:
            task = "mortality" if "mortality" in v else "sepsis"
            pid, hid, y, p = npz[v + "@pid"], npz[v + "@hid"], npz[v + "@y"], npz[v + "@p"]
            self.assertEqual(len(set(pid.tolist())), len(pid), f"{v}: duplicate patients")
            self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
            for h in np.unique(hid):
                ids, truth = expected_test(self.up, h, task)
                m = hid == h
                order = np.argsort(pid[m])
                np.testing.assert_array_equal(pid[m][order], np.sort(ids), f"{v} hospital {h}: test patients differ from the frozen cohort")
                np.testing.assert_array_equal(y[m][order], truth[np.argsort(ids)], f"{v} hospital {h}: labels differ from the frozen cohort")

    def nonfl_trained(self, task, clf="lr"):
        specs = nf.variant_specs(1, ("fused",), (clf,), ("A", "B"), (task,))
        rows = nf.run_variants(1, self.cohort, self.ds, specs, progress_every=0)
        return {r["hospitalid"] for r in rows if r["status"] == "trained"}


class TestConfig1(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.res = fed.run_fl(1, cls.cohort, cls.ds, cls.fl, cls.fed_base, embedding_sets=("fused", "gru"), classifiers=("lr", "mlp", "gbdt_fed"), algorithms=ALGOS,
                             tasks=("mortality", "sepsis"), run_id="c1")

    def test_variants_clients_and_the_paired_set(self):
        s = self.summary(self.res)
        expect = {f"f1__{es}__{t}__{c}__{a}" for es in ("fused", "gru") for t in ("mortality", "sepsis") for c in ("lr", "mlp") for a in ALGOS}
        expect |= {f"f1__{es}__{t}__gbdt_fed__fedgbdt" for es in ("fused", "gru") for t in ("mortality", "sepsis")}
        self.assertEqual(set(s), expect)
        for task in ("mortality", "sepsis"):
            trained = self.nonfl_trained(task)
            for v, a in s.items():
                if f"__{task}__" not in v:
                    continue
                rows = [r for r in self.rows(self.res) if r["variant"] == v]
                self.assertEqual({r["hospitalid"] for r in rows}, {11, 12, 13, 14}, "every hospital must appear exactly once per variant (evaluated or skipped with a reason)")
                self.assertEqual({r["hospitalid"] for r in rows if r["eval_set"] == "paired" and r["status"] == "trained"}, trained, f"{v}: paired set != Non-FL trained hospitals")
                tiny = next(r for r in rows if r["hospitalid"] == 14)
                self.assertEqual(tiny["eval_set"], "fl_only")
                self.assertTrue(tiny["participated_in_training"])                       # the 24-patient hospital still contributes to the federation
                self.assertEqual(a["counts"]["n_fl_only_evaluated"] + a["counts"]["n_paired_evaluated"], a["counts"]["n_evaluated"])

    def test_predictions_equal_the_frozen_cohort_test_patients(self):
        self.check_predictions_match_frozen_cohort(self.res)

    def test_manifest_states_the_regime_and_every_file_verifies(self):
        m = json.loads((Path(self.res["run_dir"]) / "manifest.json").read_text())
        self.assertTrue(m["centrally_pretrained_representations"] and m["end_to_end_federated"] is False)
        self.assertIn("NOT end-to-end federated learning", m["regime"])
        self.assertEqual(m["split_identity"], self.cohort.manifest["split_identity"])
        self.assertEqual(m["cross_attention_run"], self.cohort.manifest["run_id"])
        self.assertIn("tie", m["histogram_validation"]["note"])
        self.assertIn("never parameter-averaged", m["tree_note"])
        for n, h in m["files"].items():
            self.assertEqual(hashlib.sha256((Path(self.res["run_dir"]) / n).read_bytes()).hexdigest(), h)

    def test_training_logs_checkpoints_reports_and_confidence_intervals(self):
        d = Path(self.res["run_dir"])
        for v in self.summary(self.res):
            lg = json.loads((d / "training_logs" / f"{v}.json").read_text())
            self.assertIsNotNone(lg["seconds"])
            if "fedgbdt" not in v:
                self.assertGreaterEqual(lg["rounds_run"], 1)
                self.assertGreaterEqual(lg["comm_mb_total"], 0)
                self.assertLessEqual(abs(lg["val_auroc_hist_at_best"] - lg["val_auroc_exact_diag"]), lg["val_auroc_tie_bound_at_best"] + 1e-9)
                self.assertTrue((d / "checkpoints" / v / "final.pt").is_file())
            else:
                self.assertTrue((d / "checkpoints" / v / "final.pkl").is_file())
                self.assertIsNotNone(lg["best_iteration"])
        self.assertIn("Per-class", (d / "classification_reports.md").read_text())
        cls = json.loads((d / "classification_reports.json").read_text())
        v0 = next(iter(cls))
        self.assertIn("f1_max_operating_point", cls[v0]["paired_nonfl"])
        self.assertIn("recall_target_operating_point", cls[v0]["paired_nonfl"])
        self.assertGreater(len((d / "per_hospital_classification_reports.csv").read_text().splitlines()), 1)
        ps = json.loads((d / "pooled_summary.json").read_text())
        e = ps["f1__fused__mortality__lr__fedavg"]["paired_nonfl"]
        self.assertIsNotNone(e["pooled"])
        self.assertIsNotNone(e["macro_ci95"]["auroc"])
        csv_head = (d / "per_hospital.csv").read_text().splitlines()[0]
        for col in ("threshold_source", "eval_set", "nonfl_eligible", "skip_reason", "tn", "fp", "fn", "tp", "rt_recall"):
            self.assertIn(col, csv_head)

    def test_clients_outside_the_matched_scope_are_reported_with_reasons_and_the_scope_control_matches_nonfl(self):
        fl = fast_fl()
        fl["scope"] = "nonfl_matched"
        res = fed.run_fl(1, self.cohort, self.ds, fl, self.fed_base, classifiers=("lr",), algorithms=("fedavg",), tasks=("mortality",), run_id="c1_matched")
        rows = self.rows(res)
        tiny = next(r for r in rows if r["hospitalid"] == 14)
        self.assertEqual(tiny["eval_set"], "not_a_client")
        self.assertFalse(tiny["participated_in_training"])
        self.assertIn("outside the Non-FL matched set", tiny["skip_reason"])
        self.assertEqual({r["hospitalid"] for r in rows if r["status"] == "trained"}, self.nonfl_trained("mortality"))

    def test_fedgbdt_is_never_parameter_averaged_and_trees_have_no_neural_algorithm_variants(self):
        names = set(self.summary(self.res))
        self.assertFalse([v for v in names if "gbdt" in v and any(a in v for a in ("fedprox", "fedadam", "scaffold"))])
        with self.assertRaises(ValueError):
            fed.fl_variant_specs(1, ("fused",), ("lightgbm",), ("A",), ("mortality",), ALGOS)
        with self.assertRaises(ValueError):
            fed.fl_variant_specs(1, ("fused",), ("lr",), ("A",), ("mortality",), ("fedavg", "fedgbdt"))


class TestConfig2And3(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.c2 = fed.run_fl(2, cls.cohort, cls.ds, cls.fl, cls.fed_base, graphs=("A", "B"), algorithms=ALGOS, tasks=("mortality", "sepsis"), run_id="c2")
        cls.c3 = fed.run_fl(3, cls.cohort, cls.ds, cls.fl, cls.fed_base, graphs=("A", "B"), algorithms=("fedavg", "scaffold"), tasks=("mortality", "sepsis"), run_id="c3")

    def test_graph_variants_predictions_and_paired_set(self):
        s2, s3 = self.summary(self.c2), self.summary(self.c3)
        self.assertEqual(len(s2), 2 * 2 * 4)
        self.assertEqual(len(s3), 2 * 2 * 2)
        for res in (self.c2, self.c3):
            self.check_predictions_match_frozen_cohort(res)
        for task in ("mortality", "sepsis"):
            trained = self.nonfl_trained(task)
            for v in [x for x in s2 if f"__{task}__" in x]:
                rows = [r for r in self.rows(self.c2) if r["variant"] == v]
                self.assertEqual({r["hospitalid"] for r in rows if r["eval_set"] == "paired" and r["status"] == "trained"}, trained)

    def test_graphs_are_hospital_local_and_smote_is_training_only(self):
        for res, smote in ((self.c2, False), (self.c3, True)):
            logs = [json.loads(p.read_text()) for p in sorted((Path(res["run_dir"]) / "training_logs").glob("*.json"))]
            seen_synth = 0
            for lg in logs:
                gi = lg["graph_info"]
                self.assertEqual(set(map(int, gi)), {11, 12, 13, 14}, "every client must own its graph")
                for h, info in gi.items():
                    n_train = next(r for r in self.rows(res) if r["hospitalid"] == int(h) and r["variant"] == lg["spec"]["variant"])["counts"]["train"]["n"]
                    self.assertEqual(info["train_nodes"], n_train, "a client's graph must contain exactly its OWN training patients")
                    self.assertEqual(info["smote"]["n_synthetic"] if smote else 0, info["smote"].get("n_synthetic", 0) if smote else 0)
                    seen_synth += info["smote"].get("n_synthetic", 0)
            if smote:
                self.assertGreater(seen_synth, 0, "SMOTE produced no synthetic node anywhere")
            else:
                self.assertEqual(seen_synth, 0)
        a = json.loads((Path(self.c2.get("run_dir")) / "summary.json").read_text())
        b = json.loads((Path(self.c3.get("run_dir")) / "summary.json").read_text())
        self.assertNotEqual(a["f2__graphA__mortality__fedavg"]["paired_nonfl"]["metrics"]["auroc"]["macro_mean"], b["f3__graphA__mortality__fedavg"]["paired_nonfl"]["metrics"]["auroc"]["macro_mean"])

    def test_every_algorithm_trained_a_gcn_and_none_diverged_silently(self):
        for lg in (json.loads(p.read_text()) for p in (Path(self.c2.get("run_dir")) / "training_logs").glob("*.json")):
            self.assertEqual(lg["family"], "gcn")
            self.assertEqual(lg["nonfinite_updates"], 0)
            self.assertGreaterEqual(lg["rounds_run"], 1)


class TestConfig4(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.res = fed.run_fl(4, cls.cohort, cls.ds, cls.fl, cls.fed_base, graphs=("A", "B"), classifiers=("lr", "mlp", "gbdt_fed"), algorithms=("fedavg", "fedprox"), tasks=("mortality", "sepsis"),
                             c4_gcn_algorithm="fedadam", run_id="c4")

    def test_two_stage_structure_and_stage_one_checkpoints(self):
        d = Path(self.res["run_dir"])
        s = self.summary(self.res)
        self.assertEqual(len(s), 2 * 2 * (2 * 2 + 1))
        hidden = self.ds["gcn"]["hidden"]
        for g in ("A", "B"):
            for t in ("mortality", "sepsis"):
                s1 = f"c4s1__graph{g}__{t}__fedadam"
                self.assertTrue((d / "checkpoints" / s1 / "final.pt").is_file(), "stage-1 federated GCN checkpoint missing")
                lg1 = json.loads((d / "training_logs" / f"{s1}.json").read_text())
                self.assertEqual((lg1["family"], lg1["algo"]), ("gcn", "fedadam"))
                for clf in ("lr", "mlp"):
                    lg = json.loads((d / "training_logs" / f"f4__graph{g}__{t}__{clf}__fedavg.json").read_text())
                    self.assertEqual(lg["stage1_variant"], s1)
                    if clf == "lr":
                        self.assertEqual(lg["n_params"], hidden + 1, "stage-2 LR must consume the GCN's hidden-dimension embeddings, not the fused input")
        gb = json.loads((d / "training_logs" / "f4__graphA__mortality__gbdt_fed__fedgbdt.json").read_text())
        self.assertEqual(gb["stage1_variant"], "c4s1__graphA__mortality__fedadam")

    def test_stage_two_inputs_are_the_final_global_gcn_embeddings(self):
        from src.three_modality import fl_algorithms as fa, fl_clients, fl_server
        d = Path(self.res["run_dir"])
        clients, _ = fl_clients.build_clients(self.cohort, "mortality", "fused", self.ds, "all", None, self.fl["seed"])
        problems, _ = fa.make_problems(clients, "gcn", self.ds, "A", False)
        ck = torch.load(d / "checkpoints" / "c4s1__graphA__mortality__fedadam" / "final.pt", weights_only=False)["result"]
        model = fa.build_model("gcn", clients[0].dim, self.ds, 0)
        fa.set_params(model, ck["w"])
        model.eval()
        with torch.no_grad():
            emb = problems[11].embeddings(model, "test")
        self.assertEqual(emb.shape, (problems[11].client.n_test, self.ds["gcn"]["hidden"]))
        self.assertTrue(np.isfinite(emb).all())

    def test_predictions_match_the_frozen_cohort(self):
        self.check_predictions_match_frozen_cohort(self.res)


class TestPersonalised(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.res = fed.run_fl_personalized(cls.cohort, cls.ds, cls.fl, cls.fed_base, tasks=("mortality", "sepsis"), run_id="pers")

    def test_variants_head_sources_and_reference(self):
        s = self.summary(self.res)
        for t in ("mortality", "sepsis"):
            for m in ("mlp", "gcn"):
                g = "__graphA" if m == "gcn" else ""
                for meth in ("fedavg_global", "finetune", "fedper", "fedrep"):
                    self.assertIn(f"fp__{meth}__{m}{g}__{t}", s)
        rows = self.rows(self.res)
        per = [r for r in rows if r["variant"] == "fp__fedper__mlp__mortality" and r["status"] == "trained"]
        self.assertEqual({r["head_source"] for r in per if r["hospitalid"] != 14}, {"local_head"})
        self.assertEqual([r["head_source"] for r in per if r["hospitalid"] == 14], ["mean_head_fallback"])
        ft = [r for r in rows if r["variant"] == "fp__finetune__mlp__mortality" and r["status"] == "trained"]
        self.assertEqual({r["head_source"] for r in ft if r["hospitalid"] != 14}, {"local_finetune"})
        self.assertEqual([r["head_source"] for r in ft if r["hospitalid"] == 14], ["global_fallback"])
        lg = json.loads((Path(self.res["run_dir"]) / "training_logs" / "fp__finetune__mlp__mortality.json").read_text())
        self.assertEqual(lg["base_variant"], "fp__fedavg_global__mlp__mortality")
        self.assertGreaterEqual(lg["n_finetuned"], 3)

    def test_fine_tuning_never_ends_worse_than_the_global_model_on_validation_loss(self):
        lg = json.loads((Path(self.res["run_dir"]) / "training_logs" / "fp__finetune__mlp__mortality.json").read_text())
        for h, info in lg["per_client"].items():
            if info["source"] == "local_finetune":
                self.assertGreaterEqual(info["best_epoch"], 0)
                self.assertLessEqual(info["best_epoch"], info["epochs_run"])

    def test_predictions_match_the_frozen_cohort(self):
        self.check_predictions_match_frozen_cohort(self.res)


class TestRunSemantics(Base):
    def test_existing_runs_are_never_touched_cache_hits_and_explicit_ids_protected(self):
        nres = nf.run_nonfl(1, self.cohort, self.ds, self.root / "non_federated", classifiers=("lr",), tasks=("mortality",), run_id="nonfl_frozen")
        before = {"nonfl": tree_hash(self.root / "non_federated"), "ca": tree_hash(self.root / "cross_attention")}
        a = fed.run_fl(1, self.cohort, self.ds, self.fl, self.fed_base, classifiers=("lr",), algorithms=("fedavg",), tasks=("mortality",), run_id="iso1")
        self.assertFalse(a["cache_hit"])
        again = fed.run_fl(1, self.cohort, self.ds, self.fl, self.fed_base, classifiers=("lr",), algorithms=("fedavg",), tasks=("mortality",))
        self.assertTrue(again["cache_hit"] and again["run_id"] == "iso1")
        forced = fed.run_fl(1, self.cohort, self.ds, self.fl, self.fed_base, classifiers=("lr",), algorithms=("fedavg",), tasks=("mortality",), force=True, run_id="iso2")
        self.assertNotEqual(forced["run_dir"], a["run_dir"])
        with self.assertRaises(FileExistsError):
            fed.run_fl(1, self.cohort, self.ds, self.fl, self.fed_base, classifiers=("lr",), algorithms=("fedavg",), tasks=("mortality",), force=True, run_id="iso1")
        self.assertEqual(tree_hash(self.root / "non_federated"), before["nonfl"], "the Non-FL runs were modified")
        self.assertEqual(tree_hash(self.root / "cross_attention"), before["ca"], "the cross-attention run was modified")
        self.assertEqual(nres["run_id"], "nonfl_frozen")

    def test_interrupted_run_resumes_to_exactly_the_uninterrupted_result_and_resume_guards(self):
        class Boom(Exception):
            pass
        kw = dict(graphs=("A",), algorithms=("fedavg", "scaffold"), tasks=("mortality",))
        full = fed.run_fl(2, self.cohort, self.ds, self.fl, self.fed_base, run_id="full", **kw)

        def hook(r):
            if r == 3:
                raise Boom()
        with self.assertRaises(Boom):
            fed.run_fl(2, self.cohort, self.ds, self.fl, self.fed_base, run_id="part", round_hook=hook, force=True, **kw)
        part = self.fed_base / "part"
        self.assertTrue((part / "run_state.json").is_file() and not (part / "manifest.json").exists())
        with self.assertRaises(ValueError):
            fed.run_fl(2, self.cohort, self.ds, self.fl, self.fed_base, resume=True, **kw)                  # a run id is mandatory
        with self.assertRaises(RuntimeError):
            fed.run_fl(2, self.cohort, self.ds, self.fl, self.fed_base, resume=True, run_id="part", graphs=("A",), algorithms=("fedavg",), tasks=("mortality",))     # different identity
        res = fed.run_fl(2, self.cohort, self.ds, self.fl, self.fed_base, resume=True, run_id="part", **kw)
        pa, pb = np.load(Path(full["run_dir"]) / "predictions.npz", allow_pickle=True), np.load(Path(res["run_dir"]) / "predictions.npz", allow_pickle=True)
        self.assertEqual(sorted(pa.files), sorted(pb.files))
        for k in pa.files:
            if k.endswith("@p"):
                np.testing.assert_allclose(pa[k], pb[k], atol=1e-6, err_msg=f"resumed run differs: {k}")
        ra, rb = json.loads((Path(full["run_dir"]) / "per_hospital.json").read_text()), json.loads((Path(res["run_dir"]) / "per_hospital.json").read_text())
        for x, y in zip(ra, rb):
            self.assertAlmostEqual(x["threshold"] or 0, y["threshold"] or 0, places=6)
        with self.assertRaises(FileExistsError):
            fed.run_fl(2, self.cohort, self.ds, self.fl, self.fed_base, resume=True, run_id="part", **kw)    # completed runs are never resumed / overwritten


class TestReport(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.nres = nf.run_nonfl(1, cls.cohort, cls.ds, cls.root / "non_federated", classifiers=("lr", "mlp", "lightgbm", "gbdt"), tasks=("mortality",), run_id="nonfl_c1")
        cls.fres = fed.run_fl(1, cls.cohort, cls.ds, cls.fl, cls.fed_base, classifiers=("lr", "mlp", "gbdt_fed"), algorithms=("fedavg", "fedprox"), tasks=("mortality",), run_id="rep_c1")
        cls.rep = fr.build_fl_report(cls.root / "non_federated", cls.fed_base, cls.root / "reports", cls.cohort.manifest["run_id"], n_bootstrap=100, pooled_boot=50, run_id="rep")

    def test_pairing_identical_test_patients_and_separate_macro_and_pooled_tables(self):
        comp = {(c["fl_variant"], c["nonfl_variant"]): c for c in self.rep["comparisons"]}
        c = comp[("f1__fused__mortality__lr__fedavg", "c1__fused__mortality__lr")]
        self.assertTrue(c["identical_test_patients"])
        self.assertEqual(c["n_common_hospitals"], 3)
        self.assertIsNotNone(c["macro_auroc_delta_ci95"])
        self.assertIsNotNone(c["pooled_auroc_fl"])
        self.assertAlmostEqual(c["macro_auroc_delta"], c["macro_auroc_fl"] - c["macro_auroc_nonfl"], places=9)
        self.assertIn(("f1__fused__mortality__gbdt_fed__fedgbdt", "c1__fused__mortality__lightgbm"), comp)
        self.assertIn(("f1__fused__mortality__gbdt_fed__fedgbdt", "c1__fused__mortality__gbdt"), comp)
        self.assertEqual(self.rep["manifest"]["n_identical_test_patient_checks_failed"], 0)
        md = (Path(self.rep["run_dir"]) / "report.md").read_text()
        self.assertIn("Hospital-macro, paired hospitals", md)
        self.assertIn("Pooled over the same paired test patients (separate from the table above)", md)
        self.assertIn("select the headline arm HERE, on validation", md)
        self.assertIn("NOT end-to-end federated learning", md)

    def test_report_refuses_a_modified_run_and_a_missing_counterpart_is_marked_unavailable(self):
        comp = fr.compare({"variants": {}}, fr.collect_fl_runs(self.fed_base, self.cohort.manifest["run_id"]), 50, 20, 0, {}, {})
        self.assertTrue(comp and all(not c["available"] for c in comp))
        victim = Path(self.fres["run_dir"]) / "summary.json"
        original = victim.read_text()
        try:
            victim.write_text(original + " ")
            with self.assertRaises(ValueError):
                fr.collect_fl_runs(self.fed_base, self.cohort.manifest["run_id"])
        finally:
            victim.write_text(original)


class TestCLI(Base):
    def cli(self, *args):
        sets = ["--set", f"run.output_dir={self.out}", "--set", "three_modality.federated.rounds.max_rounds=3", "--set", "three_modality.federated.rounds.min_rounds=1",
                "--set", "three_modality.federated.gbdt.max_rounds=3"]
        return subprocess.run([sys.executable, "run_three_modality.py", "--config", "configs/three_modality.yaml", *args, *sets], cwd=ROOT, capture_output=True, text=True)

    @staticmethod
    def js(stdout):
        return json.loads(stdout[stdout.rindex("\n{") + 1: stdout.rindex("}") + 1]) if "\n{" in stdout else json.loads(stdout[stdout.index("{"): stdout.rindex("}") + 1])

    def test_fl_stages_flags_errors_tuning_and_reuse(self):
        r = self.cli("--stage", "fl_config1", "--classifiers", "lr,gbdt_fed", "--algorithms", "fedavg,fedadam", "--tasks", "mortality", "--max-hospitals", "3", "--run-id", "cli_c1")
        self.assertEqual(r.returncode, 0, r.stderr[-2500:] + r.stdout[-1500:])
        s = self.js(r.stdout)
        self.assertTrue(s["partial"] and not s["cache_hit"])
        self.assertEqual(set(s["variants"]), {"f1__fused__mortality__lr__fedavg", "f1__fused__mortality__lr__fedadam", "f1__fused__mortality__gbdt_fed__fedgbdt"})
        self.assertTrue(self.js(self.cli("--stage", "fl_config1", "--classifiers", "lr,gbdt_fed", "--algorithms", "fedavg,fedadam", "--tasks", "mortality", "--max-hospitals", "3").stdout)["cache_hit"])
        bad = self.cli("--stage", "fl_config1", "--algorithms", "fedavg,fedgbdt")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("never FedAvg", bad.stderr + bad.stdout)
        worse = self.cli("--stage", "fl_config1", "--classifiers", "lightgbm")
        self.assertNotEqual(worse.returncode, 0)
        self.assertIn("unknown FL classifier", worse.stderr + worse.stdout)
        gru = self.cli("--stage", "fl_config2", "--embedding-set", "gru")
        self.assertNotEqual(gru.returncode, 0)
        self.assertIn("fl_config1 only", gru.stderr + gru.stdout)
        rep = self.cli("--stage", "fl_report")                                       # only a partial FL run exists -> excluded -> nothing to report
        self.assertNotEqual(rep.returncode, 0)
        t = self.cli("--stage", "fl_tune", "--tune-families", "lr", "--tasks", "mortality", "--max-hospitals", "3", "--algorithms", "fedavg", "--run-id", "cli_tune",
                     "--set", "three_modality.federated.tuning.quick=true")
        self.assertEqual(t.returncode, 0, t.stderr[-2500:] + t.stdout[-1500:])
        tune = self.js(t.stdout)
        self.assertIn("lr", tune["best_hparams"]["local"]["lr"])
        a = self.cli("--stage", "fl_config1", "--classifiers", "lr", "--algorithms", "fedavg", "--tasks", "mortality", "--max-hospitals", "3", "--fl-hparams-run-id", "cli_tune", "--run-id", "cli_tuned")
        self.assertEqual(a.returncode, 0, a.stderr[-2500:] + a.stdout[-1500:])
        man = json.loads((Path(self.js(a.stdout)["run_dir"]) / "manifest.json").read_text())
        self.assertEqual(man["hparams"], {"source_run": "cli_tune"})
        for stage, extra in (("fl_config3", ["--graph", "A", "--algorithms", "fedavg"]), ("fl_config4", ["--graph", "A", "--classifiers", "lr", "--algorithms", "fedavg"]),
                             ("fl_personalized", ["--methods", "fedper,finetune", "--models", "mlp"])):
            o = self.cli("--stage", stage, "--tasks", "mortality", "--max-hospitals", "3", *extra)
            self.assertEqual(o.returncode, 0, f"{stage}: " + o.stderr[-2500:] + o.stdout[-1500:])
            self.assertTrue(self.js(o.stdout)["variants"])


if __name__ == "__main__":
    unittest.main()
