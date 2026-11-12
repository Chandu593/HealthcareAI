"""FL core: aggregation mathematics (FedAvg, FedProx, FedAdam, SCAFFOLD), local updates, federated histogram validation, the round loop, personalisation structure, checkpoint/resume
exactness, non-finite handling and tiny-client graph construction. Pure arrays -- no cohort needed."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.paper import fed_gbdt as fg
from src.three_modality import fl_algorithms as fa
from src.three_modality import fl_gbdt_adapter as fga
from src.three_modality.downstream_config import DOWNSTREAM_DEFAULTS
from src.three_modality.fl_config import FL_DEFAULTS, apply_fl_hparams, validate_fl_cfg
from src.three_modality.fl_server import HistAccumulator, train_federated
from tests.unit._fl_synth import make_client, make_clients

DS = copy.deepcopy(DOWNSTREAM_DEFAULTS)
DS["gcn"].update(max_epochs=8, patience=4)


def fl_cfg(**rounds):
    c = validate_fl_cfg(copy.deepcopy(FL_DEFAULTS))
    c["rounds"].update(max_rounds=6, patience=4, min_rounds=2, local_steps=2)
    c["rounds"].update(rounds)
    c["checkpoint"]["every_rounds"] = 1
    return c


def problems_for(clients, family="lr", graph=None, smote=False):
    p, failed = fa.make_problems(clients, family, DS, graph, smote)
    assert not failed, failed
    return p


def T(*v):
    return torch.tensor(v, dtype=torch.float64)


class TestAggregationMath(unittest.TestCase):
    def test_fedavg_is_the_sample_weighted_mean(self):
        w = [T(1.0, 0.0)]
        locs = [[T(3.0, 1.0)], [T(7.0, 5.0)]]
        p = fa.client_weights([100, 300])                                   # 0.25, 0.75
        new = fa.apply_server_update("fedavg", w, fa.weighted_delta(w, locs, p), {}, {})
        np.testing.assert_allclose(new[0].numpy(), [0.25 * 3 + 0.75 * 7, 0.25 * 1 + 0.75 * 5])
        np.testing.assert_allclose(new[0].numpy(), [6.0, 4.0])              # hand value

    def test_identical_clients_leave_the_mean_unchanged_and_weights_sum_to_one(self):
        w = [T(2.0)]
        new = fa.apply_server_update("fedprox", w, fa.weighted_delta(w, [[T(5.0)]] * 3, fa.client_weights([1, 2, 3])), {}, {})
        np.testing.assert_allclose(new[0].numpy(), [5.0])
        self.assertAlmostEqual(float(fa.client_weights([5, 15, 80]).sum()), 1.0)
        for bad in ([], [0, 0], [-1, 2]):
            with self.assertRaises(ValueError):
                fa.client_weights(bad)

    def test_fedadam_first_and_second_step_match_hand_computation(self):
        cfg = {"server_lr": 0.1, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3}
        w, state = [T(1.0)], {}
        d1 = [T(0.5)]
        n1 = fa.apply_server_update("fedadam", w, d1, state, cfg)
        m1, v1 = 0.1 * 0.5, 0.01 * 0.25
        np.testing.assert_allclose(n1[0].numpy(), [1.0 + 0.1 * m1 / (np.sqrt(v1) + 1e-3)], rtol=1e-12)
        d2 = [T(-0.2)]
        n2 = fa.apply_server_update("fedadam", n1, d2, state, cfg)               # moments persist across rounds
        m2, v2 = 0.9 * m1 + 0.1 * -0.2, 0.99 * v1 + 0.01 * 0.04
        np.testing.assert_allclose(n2[0].numpy(), [n1[0].item() + 0.1 * m2 / (np.sqrt(v2) + 1e-3)], rtol=1e-12)

    def test_scaffold_option_two_control_update_and_weighted_server_invariant(self):
        ci, c, wg, wl = [T(0.2)], [T(0.05)], [T(1.0)], [T(0.4)]
        new = fa.scaffold_client_control(ci, c, wg, wl, K=3, lr=0.5)
        np.testing.assert_allclose(new[0].numpy(), [0.2 - 0.05 + (1.0 - 0.4) / (3 * 0.5)])
        p = fa.client_weights([100, 300])
        c_clients = [[T(0.0)], [T(0.0)]]
        new_ci = [[T(0.4)], [T(-0.2)]]
        c_new = fa.scaffold_server_control([T(0.0)], [[n[0] - o[0]] for n, o in zip(new_ci, c_clients)], p)
        np.testing.assert_allclose(c_new[0].numpy(), [float(p[0] * 0.4 + p[1] * -0.2)])      # c = sum p_i c_i is preserved when it held before


class TestLocalUpdates(unittest.TestCase):
    def setUp(self):
        self.client = make_client(11, n_tr=80, seed=1)
        self.prob = fa.ClientProblem(self.client, "lr", DS)
        self.model = fa.build_model("lr", self.client.dim, DS, 0)
        self.w0 = fa.get_params(self.model)

    def run_local(self, **kw):
        base = dict(algo="fedavg", optimizer="sgd", lr=0.1, wd=0.0, phases=[(1, list(range(len(self.w0))))], seed=3)
        base.update(kw)
        return fa.local_update(self.model, self.prob, kw.pop("w_start", self.w0), 0.5, 2.0, **{k: v for k, v in base.items() if k != "w_start"})

    def test_fedprox_one_step_differs_from_fedavg_by_exactly_minus_lr_mu_times_the_displacement(self):
        anchor = [t + 0.3 for t in self.w0]                                      # start != anchor so the proximal gradient is non-zero
        a, _ = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="fedavg", optimizer="sgd", lr=0.1, wd=0.0, phases=[(1, [0, 1])], seed=3, w_anchor=anchor)
        b, _ = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="fedprox", optimizer="sgd", lr=0.1, wd=0.0, phases=[(1, [0, 1])], mu=0.7, seed=3, w_anchor=anchor)
        for j in range(2):
            np.testing.assert_allclose((b[j] - a[j]).numpy(), (-0.1 * 0.7 * (self.w0[j] - anchor[j])).numpy(), atol=1e-7)

    def test_scaffold_with_zero_controls_equals_fedavg_and_controls_shift_the_step_exactly(self):
        zeros = [torch.zeros_like(t) for t in self.w0]
        a, _ = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="fedavg", optimizer="sgd", lr=0.1, wd=0.0, phases=[(3, [0, 1])], seed=3)
        s, _ = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="scaffold", optimizer="sgd", lr=0.1, wd=0.0, phases=[(3, [0, 1])], seed=3, c_global=zeros, c_local=zeros)
        for j in range(2):
            np.testing.assert_allclose(s[j].numpy(), a[j].numpy(), atol=1e-7)
        c_loc = [torch.full_like(t, 0.2) for t in self.w0]
        c_glo = [torch.full_like(t, 0.05) for t in self.w0]
        s1, _ = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="scaffold", optimizer="sgd", lr=0.1, wd=0.0, phases=[(1, [0, 1])], seed=3, c_global=c_glo, c_local=c_loc)
        a1, _ = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="fedavg", optimizer="sgd", lr=0.1, wd=0.0, phases=[(1, [0, 1])], seed=3)
        for j in range(2):                                                       # first step: only the correction differs: -lr * (-c_i + c)
            np.testing.assert_allclose((s1[j] - a1[j]).numpy(), (-0.1 * (-c_loc[j] + c_glo[j])).numpy(), atol=1e-7)

    def test_scaffold_requires_plain_sgd_and_single_phase(self):
        z = [torch.zeros_like(t) for t in self.w0]
        with self.assertRaises(ValueError):
            fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="scaffold", optimizer="adam", lr=0.1, wd=0.0, phases=[(1, [0, 1])], c_global=z, c_local=z)
        with self.assertRaises(ValueError):
            fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="scaffold", optimizer="sgd", lr=0.1, wd=0.0, phases=[(1, [0]), (1, [1])], c_global=z, c_local=z)

    def test_adam_control_path_trains_and_nonfinite_loss_excludes_the_client(self):
        w, st = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="fedavg", optimizer="adam", lr=0.05, wd=1e-3, phases=[(3, [0, 1])], seed=1)
        self.assertIsNotNone(w)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(w, self.w0)))
        with mock.patch.object(fa, "focal_loss_torch", lambda lg, y, a, g: lg.sum() * float("nan")):
            w2, st2 = fa.local_update(self.model, self.prob, self.w0, 0.5, 2.0, algo="fedavg", optimizer="sgd", lr=0.1, wd=0.0, phases=[(1, [0, 1])], seed=1)
        self.assertIsNone(w2)
        self.assertTrue(st2["nonfinite"])


class TestHistogramValidation(unittest.TestCase):
    def test_histogram_auroc_is_within_its_tie_bound_of_the_exact_auroc(self):
        from sklearn.metrics import roc_auc_score
        rs = np.random.default_rng(0)
        y = (rs.random(3000) < 0.2).astype(int)
        s = rs.normal(0, 2, 3000) + 1.5 * y
        for bins in (20, 200, 3000):
            h = HistAccumulator(bins, 15.0)
            h.add(s, y)
            self.assertLessEqual(abs(h.auroc() - roc_auc_score(y, s)), h.tie_bound() + 1e-12)
        fine = HistAccumulator(3000, 15.0)
        fine.add(s, y)
        self.assertLess(abs(fine.auroc() - roc_auc_score(y, s)), 2e-3)
        coarse = HistAccumulator(20, 15.0)
        coarse.add(s, y)
        self.assertGreater(coarse.tie_bound(), fine.tie_bound())            # the documented approximation error shrinks with the bin count

    def test_merging_client_histograms_equals_histogramming_the_union_and_saturated_logits_are_clipped(self):
        rs = np.random.default_rng(1)
        y, s = (rs.random(600) < 0.3).astype(int), rs.normal(0, 3, 600)
        s[:5] = [99, -99, 99, -99, 50]
        whole, a, b = HistAccumulator(500, 15.0), HistAccumulator(500, 15.0), HistAccumulator(500, 15.0)
        whole.add(s, y)
        a.add(s[:250], y[:250])
        b.add(s[250:], y[250:])
        a.merge(b)
        np.testing.assert_array_equal(a.pos, whole.pos)
        np.testing.assert_array_equal(a.neg, whole.neg)
        self.assertEqual(int(whole.pos.sum() + whole.neg.sum()), 600)

    def test_global_threshold_maximises_f1_over_the_pooled_histogram(self):
        rs = np.random.default_rng(2)
        y = (rs.random(2000) < 0.25).astype(int)
        s = rs.normal(0, 1.5, 2000) + 2.0 * y
        h = HistAccumulator(3000, 15.0)
        h.add(s, y)
        thr = h.best_f1_probability_threshold()
        logit_thr = np.log(thr / (1 - thr))

        def f1_at(t):
            pred = s >= t
            tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
            return 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(f1_at(t) for t in np.unique(s))
        self.assertGreater(f1_at(logit_thr), best - 0.02)                    # bin-resolution optimum is within 0.02 F1 of the exact optimum
        self.assertTrue(np.isnan(HistAccumulator(10, 5.0).best_f1_probability_threshold()))


class TestRoundLoop(unittest.TestCase):
    def setUp(self):
        self.clients = make_clients()

    def train(self, family="lr", algo="fedavg", method="global", cfg=None, ckpt=None, resume=False, hook=None, clients=None, ident="t"):
        cfg = cfg or fl_cfg()
        pr = problems_for(clients or self.clients, family)
        alphas = fa.make_alphas(pr, DS, cfg, False)
        return train_federated(pr, family, DS, cfg, algo=algo, method=method, alphas=alphas, seed=5, ckpt_dir=ckpt, resume=resume, ident=ident, round_hook=hook, tag="t")

    def test_every_algorithm_runs_is_deterministic_and_differs_from_the_others(self):
        outs = {}
        for algo in ("fedavg", "fedprox", "fedadam", "scaffold"):
            a, b = self.train(algo=algo), self.train(algo=algo)
            for x, y in zip(a["w"], b["w"]):
                self.assertTrue(torch.equal(x, y), f"{algo} is not deterministic")
            self.assertEqual(a["best_round"], b["best_round"])
            self.assertEqual(a["nonfinite_updates"], 0)
            outs[algo] = a
        flat = {k: torch.cat([t.flatten() for t in v["w"]]) for k, v in outs.items()}
        for i, a in enumerate(flat):
            for b in list(flat)[i + 1:]:
                self.assertFalse(torch.allclose(flat[a], flat[b]), f"{a} and {b} gave identical models")

    def test_first_round_fedavg_equals_the_hand_computed_weighted_mean_of_local_models(self):
        cfg = fl_cfg(max_rounds=1, min_rounds=1)
        res = self.train(cfg=cfg)
        pr = problems_for(self.clients)
        alphas = fa.make_alphas(pr, DS, cfg, False)
        model = fa.build_model("lr", self.clients[0].dim, DS, 5)
        w0 = fa.get_params(model)
        ids = sorted(pr)
        locs = [fa.local_update(model, pr[h], w0, alphas[h], 2.0, algo="fedavg", optimizer="sgd", lr=cfg["local"]["lr"]["lr"], wd=cfg["local"]["weight_decay"]["lr"],
                               phases=[(2, [0, 1])], seed=fa.round_seed(5, 1, h))[0] for h in ids]
        p = fa.client_weights([pr[h].client.n_train for h in ids])
        for j in range(2):
            np.testing.assert_allclose(res["w"][j].numpy(), sum(float(p[i]) * locs[i][j] for i in range(len(ids))).numpy(), atol=1e-6)

    def test_selected_round_is_the_validation_best_and_history_is_complete(self):
        res = self.train(cfg=fl_cfg(max_rounds=8, patience=3))
        scores = [h["val_auroc_hist"] for h in res["history"]]
        self.assertEqual(res["best_round"], int(np.argmax(scores)) + 1 if scores.count(max(scores)) == 1 else res["best_round"])
        self.assertAlmostEqual(res["best_val_score"], max(scores), places=6)
        self.assertEqual(res["rounds_run"], len(res["history"]))
        self.assertIn(res["stop_reason"].split(" ")[0], ("early", "max_rounds"))
        self.assertTrue(all(h["val_auroc_tie_bound"] is not None for h in res["history"]))
        self.assertGreater(res["comm_mb_total"], 0)

    def test_resume_after_interruption_reproduces_the_uninterrupted_run_for_every_algorithm_and_method(self):
        class Boom(Exception):
            pass
        cases = [("lr", "fedavg", "global"), ("mlp", "fedadam", "global"), ("mlp", "scaffold", "global"), ("mlp", "fedprox", "global"), ("mlp", "fedavg", "fedper"), ("mlp", "fedavg", "fedrep")]
        for family, algo, method in cases:
            with self.subTest(family=family, algo=algo, method=method), tempfile.TemporaryDirectory() as d:
                cfg = fl_cfg(max_rounds=5, patience=9)
                full = self.train(family, algo, method, cfg, ckpt=Path(d) / "full")

                def hook(r):
                    if r == 4:
                        raise Boom()
                with self.assertRaises(Boom):
                    self.train(family, algo, method, cfg, ckpt=Path(d) / "part", hook=hook)
                self.assertTrue((Path(d) / "part" / "state.pt").is_file())
                again = self.train(family, algo, method, cfg, ckpt=Path(d) / "part", resume=True)
                self.assertEqual(again["resumed_from"], "state")
                for x, y in zip(full["w"], again["w"]):
                    self.assertTrue(torch.allclose(x, y, atol=1e-7), "resumed model differs")
                self.assertEqual(full["best_round"], again["best_round"])
                self.assertEqual([h["val_auroc_hist"] for h in full["history"]], [h["val_auroc_hist"] for h in again["history"]])
                if method != "global":
                    for h in full["heads"]:
                        for x, y in zip(full["heads"][h], again["heads"][h]):
                            self.assertTrue(torch.allclose(x, y, atol=1e-7))
                self.assertFalse((Path(d) / "part" / "state.pt").exists())             # a finished run leaves final.pt only
                done = self.train(family, algo, method, cfg, ckpt=Path(d) / "part", resume=True)
                self.assertEqual(done["resumed_from"], "final")                      # a completed variant is reloaded, never retrained

    def test_resume_refuses_a_checkpoint_from_a_different_identity(self):
        with tempfile.TemporaryDirectory() as d:
            self.train(cfg=fl_cfg(max_rounds=2, min_rounds=1), ckpt=Path(d), ident="run-A")
            with self.assertRaises(RuntimeError):
                self.train(cfg=fl_cfg(max_rounds=2, min_rounds=1), ckpt=Path(d), resume=True, ident="run-B")

    def test_personalised_heads_stay_local_and_only_the_body_is_aggregated(self):
        res = self.train("mlp", "fedavg", "fedper", fl_cfg(max_rounds=4))
        model = fa.build_model("mlp", self.clients[0].dim, DS, 5)
        init = fa.get_params(model)
        hidx, bidx = res["hidx"], res["bidx"]
        self.assertEqual(sorted(hidx + bidx), list(range(len(init))))
        self.assertTrue(set(hidx).isdisjoint(bidx))
        for j in hidx:                                                          # the SHARED head parameters never move: heads are not aggregated
            self.assertTrue(torch.equal(res["w"][j], init[j]))
        self.assertTrue(any(not torch.equal(res["w"][j], init[j]) for j in bidx))
        eligible = {c.hospitalid for c in self.clients if c.nonfl_eligible}
        self.assertEqual(set(res["heads"]), eligible)                           # the 25-patient hospital is not head-eligible -> mean-head fallback
        hs = [torch.cat([t.flatten() for t in res["heads"][h]]) for h in sorted(res["heads"])]
        self.assertTrue(any(not torch.allclose(hs[0], x) for x in hs[1:]))      # heads genuinely differ between hospitals

    def test_a_client_with_non_finite_updates_is_skipped_counted_and_never_averaged_in(self):
        bad = self.clients[0].hospitalid
        real = fa.local_update

        def flaky(model, prob, w_start, *a, **k):
            if prob.client.hospitalid == bad:
                return None, {"nonfinite": True, "steps": 0}
            return real(model, prob, w_start, *a, **k)
        with mock.patch("src.three_modality.fl_server.local_update", flaky):
            res = self.train(cfg=fl_cfg(max_rounds=3, min_rounds=1))
        self.assertEqual(res["nonfinite_updates"], 3)
        self.assertTrue(all(h["clients_updated"] == len(self.clients) - 1 for h in res["history"]))

    def test_fedprox_with_a_huge_mu_stays_near_the_round_start_and_fedavg_moves_further(self):
        cfg = fl_cfg(max_rounds=1, min_rounds=1, local_steps=5)
        base = self.train(cfg=cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2["algorithms"]["fedprox"]["mu"] = 5.0                                  # lr * mu = 1.5 < 2: strongly regularised but stable
        prox = self.train(algo="fedprox", cfg=cfg2)
        cfg3 = copy.deepcopy(cfg)
        cfg3["algorithms"]["fedprox"]["mu"] = 50.0                                 # lr * mu = 15 >= 2: explicit SGD would diverge -> refused up front, never silently run
        with self.assertRaises(ValueError):
            self.train(algo="fedprox", cfg=cfg3)
        init = fa.get_params(fa.build_model("lr", self.clients[0].dim, DS, 5))
        mv = lambda r: float(sum((a - b).norm() for a, b in zip(r["w"], init)))
        self.assertLess(mv(prox), mv(base))


class TestTinyClientGraphs(unittest.TestCase):
    def test_hospital_local_graphs_build_for_tiny_clients_or_fail_visibly(self):
        for n in (1, 2, 3, 5):
            for g in ("A", "B"):
                with self.subTest(n=n, graph=g):
                    c = make_client(50 + n, n_tr=n, n_va=6, n_te=6, min_pos=1)
                    problems, failed = fa.make_problems([c], "gcn", DS, g, False)
                    if problems:                                                # built: must also train and score without error
                        cfg = fl_cfg(max_rounds=2, min_rounds=1)
                        res = train_federated(problems, "gcn", DS, cfg, algo="fedavg", alphas=fa.make_alphas(problems, DS, cfg, False), seed=1, tag="tiny")
                        self.assertGreaterEqual(res["rounds_run"], 1)
                    else:
                        self.assertTrue(failed and "graph construction failed" in failed[0]["reason"])       # reported, never dropped silently


class TestHospitalLocalGraphAndSmote(unittest.TestCase):
    def test_smote_nodes_stay_inside_their_own_hospital_and_attach_edges_only_from_real_training_nodes(self):
        a = make_client(11, n_tr=60, seed=1, shift=0.0)
        b = make_client(12, n_tr=60, seed=2, shift=100.0)                         # hospital B lives 100 units away
        for g in ("A", "B"):
            pa = fa.ClientProblem(a, "gcn", DS, g, smote=True)
            self.assertGreater(pa.n_synth, 0)
            synth = pa.x_graph_train[pa.n_tr:].numpy()
            self.assertLess(np.abs(a.scaler.inverse_transform(synth)).max(), 50.0, "a synthetic node of hospital A leaked toward hospital B's feature space")
            ei = pa.ei_tr.numpy()
            self.assertTrue((ei < pa.n_tr + pa.n_synth).all(), "edge references a node outside the owning client's graph")
            into_synth = ei[:, ei[1] >= pa.n_tr]
            self.assertTrue((into_synth[0] < pa.n_tr).all(), "synthetic nodes may only receive edges FROM real training nodes")
            out_of_synth = ei[:, ei[0] >= pa.n_tr]
            self.assertEqual(out_of_synth.shape[1], 0, "no edge may leave a synthetic node")
            pb = fa.ClientProblem(b, "gcn", DS, g, smote=True)
            self.assertGreater(b.scaler.inverse_transform(pb.x_graph_train[pb.n_tr:].numpy()).min(), 50.0)      # B's synthetic nodes live in B's own feature space

    def test_query_graphs_have_only_train_to_query_edges_and_never_connect_val_to_test(self):
        c = make_client(11, n_tr=80, n_va=30, n_te=30, seed=3)
        pr = fa.ClientProblem(c, "gcn", DS, "A", smote=False)
        for split in ("val", "test"):
            x, ei, ew, n_att = pr._query_graph(split)
            ei = ei.numpy()
            att = ei[:, -n_att:] if n_att else np.empty((2, 0), int)
            self.assertTrue((att[0] < pr.n_tr).all() and (att[1] >= pr.n_tr).all(), "attach edges must run train -> query only")
            self.assertEqual(x.shape[0], pr.n_tr + (c.n_val if split == "val" else c.n_test), "a query graph must contain the train nodes plus ONLY that split's patients")

    def test_train_node_embeddings_do_not_depend_on_the_attached_query_nodes(self):
        c = make_client(11, n_tr=80, n_va=30, n_te=30, seed=4)
        pr = fa.ClientProblem(c, "gcn", DS, "A", smote=False)
        model = fa.build_model("gcn", c.dim, DS, 0)
        model.eval()
        with torch.no_grad():
            h_train_only, _ = model(pr.x_graph_train, pr.ei_tr, pr.ew_tr)
            emb = pr.embeddings(model, "train")
        np.testing.assert_allclose(emb, h_train_only[: pr.n_tr].numpy(), atol=1e-5)


class TestGBDTAdapter(unittest.TestCase):
    def setUp(self):
        self.clients = make_clients(sizes=(120, 80, 40))
        self.cfg = fl_cfg()
        self.gcfg = copy.deepcopy(self.cfg["gbdt"])
        self.gcfg.update(max_rounds=6, early_stopping_rounds=3)

    def test_validation_metric_is_swapped_for_the_histogram_auroc_and_always_restored(self):
        original = fg._auroc
        with fga.federated_validation_metric(100, 15.0):
            self.assertIsNot(fg._auroc, original)
        self.assertIs(fg._auroc, original)
        with self.assertRaises(RuntimeError):
            with fga.federated_validation_metric(100, 15.0):
                raise RuntimeError("boom")
        self.assertIs(fg._auroc, original)

    def test_test_labels_are_masked_in_the_assembled_matrix_and_models_train_and_resume(self):
        asm = fga.assemble(self.clients)
        for h in asm["ids"]:
            self.assertTrue((asm["y"][asm["rows"][h]["test"]] == -1).all(), "test label slots must hold the mask value")
            self.assertTrue((asm["y"][asm["rows"][h]["train"]] >= 0).all())
        with tempfile.TemporaryDirectory() as d:
            r = fga.train_fedgbdt(self.clients, self.gcfg, self.cfg["validation"], 3, ckpt_dir=d, ident="x")
            self.assertLessEqual(abs(r["val_auroc_hist_at_best"] - r["val_auroc_exact_diag"]), r["val_auroc_tie_bound_at_best"] + 1e-9)
            again = fga.train_fedgbdt(self.clients, self.gcfg, self.cfg["validation"], 3, ckpt_dir=d, resume=True, ident="x")
            self.assertEqual(again["resumed_from"], "final")
            for h in r["margins"]:
                np.testing.assert_allclose(r["margins"][h]["test"], again["margins"][h]["test"])
            with self.assertRaises(RuntimeError):
                fga.train_fedgbdt(self.clients, self.gcfg, self.cfg["validation"], 3, ckpt_dir=d, resume=True, ident="other")


class TestConfig(unittest.TestCase):
    def test_defaults_validate_and_unsafe_overrides_are_rejected(self):
        c = validate_fl_cfg(copy.deepcopy(FL_DEFAULTS))
        self.assertEqual(c["scaler"], "per_hospital")
        bad = copy.deepcopy(FL_DEFAULTS)
        bad["scaler"] = "global"
        with self.assertRaises(ValueError):
            validate_fl_cfg(bad)
        with self.assertRaises(ValueError):
            apply_fl_hparams(c, {"seed": 1})                                       # a tuning file can never change seeds / scope / protocol
        with self.assertRaises(ValueError):
            apply_fl_hparams(c, {"scope": "nonfl_matched"})
        ok = apply_fl_hparams(c, {"local": {"lr": {"mlp": 0.5}}, "algorithms": {"fedprox": {"mu": 1.0}}, "gbdt": {"params": {"max_depth": 6}}})
        self.assertEqual((ok["local"]["lr"]["mlp"], ok["local"]["lr"]["lr"], ok["algorithms"]["fedprox"]["mu"], ok["gbdt"]["params"]["max_depth"]), (0.5, c["local"]["lr"]["lr"], 1.0, 6))
        self.assertEqual(c["algorithms"]["fedprox"]["mu"], FL_DEFAULTS["algorithms"]["fedprox"]["mu"])          # the input config is not mutated


if __name__ == "__main__":
    unittest.main()
