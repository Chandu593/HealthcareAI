import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import src.paper.trainers as T  # noqa: E402
from src.paper.config import get_paper_cfg  # noqa: E402

P = get_paper_cfg({"paper": {"neural": {"hidden": [8], "emb_dim": 4, "seq_hidden": 4, "batch_size": 16, "patience": 50},
                             "federated": {"rounds": 1, "local_epochs": 1, "patience": 50, "fedprox_mu": 0.0},
                             "graph": {"k": 3, "cosine_threshold": 0.0, "n_clusters": 2, "hidden": 4, "rounds": 2, "local_epochs": 1}}})


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed); X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0.5).astype(np.float32)
    hosp = np.repeat([1, 2, 3], n // 3)
    return T.NeuralData(torch.from_numpy(X), None, y, hosp), hosp


class TestFedAvg(unittest.TestCase):
    def test_server_update_is_n_weighted_mean_of_client_updates(self):
        data, hosp = toy()
        train_by_hosp = {1: np.arange(0, 30), 2: np.arange(40, 50)}      # 30 vs 10 patients -> weights 0.75 / 0.25
        val = np.arange(80, 120)
        deltas = iter([1.0, 5.0]); orig = T._train_epochs

        def fake(model, *a, **k):
            d = next(deltas)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(d)
            return None
        T._train_epochs = fake
        try:
            T.set_seed(7); ref = T.make_model(5, 0, P["neural"]); init = {k: v.clone() for k, v in ref.state_dict().items()}
            g, _ = T.run_fedavg(data, train_by_hosp, val, 5, 0, P["neural"], P["federated"], 7)
        finally:
            T._train_epochs = orig
        for (k, v), (_, v0) in zip(g.state_dict().items(), init.items()):
            if v.is_floating_point():
                self.assertTrue(torch.allclose(v, v0 + 0.75 * 1.0 + 0.25 * 5.0, atol=1e-5), f"{k} is not the n-weighted mean")

    def test_clients_only_see_their_own_rows(self):
        data, hosp = toy(); seen = []
        orig = T._train_epochs

        def spy(model, data_, idx, *a, **k):
            seen.append(np.unique(hosp[idx])); return orig(model, data_, idx, *a, **k)
        T._train_epochs = spy
        try:
            train_by_hosp = {h: np.where(hosp == h)[0][:20] for h in (1, 2, 3)}
            T.run_fedavg(data, train_by_hosp, np.arange(100, 120), 5, 0, P["neural"], P["federated"], 3)
        finally:
            T._train_epochs = orig
        self.assertTrue(all(len(s) == 1 for s in seen), "a local update touched rows from more than one hospital")

    def test_fedavg_and_central_actually_learn(self):
        data, hosp = toy(300, 1); tr = np.arange(0, 200); va = np.arange(200, 300)
        ncfg = dict(P["neural"], hidden=[16], lr=1e-2, central_epochs=30, patience=30)
        m, h = T.train_central(data, tr, va, 5, 0, ncfg, 0)
        self.assertGreater(h["best_val"], 0.8)
        fcfg = dict(P["federated"], rounds=30, local_epochs=2, patience=30)
        g, h2 = T.run_fedavg(data, {k: tr[hosp[tr] == k] for k in (1, 2, 3)}, va, 5, 0, ncfg, fcfg, 0)
        self.assertGreater(h2["best_val"], 0.8)

    def test_pos_weight_is_real(self):
        y = np.array([0] * 90 + [1] * 10, dtype=np.float32)
        self.assertAlmostEqual(T.pos_weight_value(y, "full"), 9.0); self.assertAlmostEqual(T.pos_weight_value(y, "sqrt"), 3.0)
        self.assertIsNone(T.pos_weight_value(y, "none"))


class TestGraphs(unittest.TestCase):
    def test_no_cross_hospital_edges_and_train_subgraph_only(self):
        rng = np.random.default_rng(0); n = 90
        emb = rng.normal(size=(n, 6)).astype(np.float32); hosp = np.repeat([1, 2, 3], 30); pids = np.arange(n) + 1000
        split = np.array((["train"] * 20 + ["val"] * 5 + ["test"] * 5) * 3)
        for method in ("knn_and_threshold", "minibatch_kmeans_and_threshold"):
            gr = T.build_graphs(emb, hosp, pids, split, P["graph"], method)
            for h, g in gr.items():
                ei = g["ei"].numpy(); self.assertTrue(np.all(hosp[g["rows"][ei[0]]] == h) and np.all(hosp[g["rows"][ei[1]]] == h))
                tr_rows = g["rows"][g["train_local"]]; self.assertTrue(np.all(split[tr_rows] == "train"))
                tei = g["train_ei"].numpy()
                if tei.shape[1]:
                    self.assertLess(tei.max(), len(tr_rows))
        none = T.build_graphs(emb, hosp, pids, split, P["graph"], "none")
        self.assertTrue(all(g["n_edges"] == 0 for g in none.values()))

    def test_graph_head_starts_at_base_logit(self):
        from src.paper.models import GraphHead
        m = GraphHead(6, 4, 2, 0.0, True).eval(); x = torch.randn(10, 6); base = torch.randn(10)
        self.assertTrue(torch.allclose(m(x, torch.zeros((2, 0), dtype=torch.long), base), base))


if __name__ == "__main__":
    unittest.main()


class TestFederatedStabilisers(unittest.TestCase):
    def test_lr_decay_and_noop_defaults(self):
        self.assertAlmostEqual(T.lr_for_round(0.01, 0.5, 3), 0.00125); self.assertEqual(T.lr_for_round(0.01, 1.0, 7), 0.01)

    def test_server_momentum_math_and_fedavg_passthrough(self):
        g = {"w": torch.tensor([1.0, 1.0])}; avg1 = {"w": torch.tensor([2.0, 3.0])}
        same, v0 = T.server_update(g, avg1, None, 0.0, 1.0)
        self.assertTrue(same is avg1 and v0 is None)                                   # plain FedAvg is untouched
        n1, v1 = T.server_update(g, avg1, None, 0.5, 1.0)                              # round 1: v = delta
        self.assertTrue(torch.allclose(n1["w"], avg1["w"]) and torch.allclose(v1["w"], torch.tensor([1.0, 2.0])))
        avg2 = {"w": n1["w"] + torch.tensor([1.0, 1.0])}                               # round 2: delta = [1,1]; v = 0.5*[1,2] + [1,1]
        n2, v2 = T.server_update(n1, avg2, v1, 0.5, 1.0)
        self.assertTrue(torch.allclose(v2["w"], torch.tensor([1.5, 2.0])) and torch.allclose(n2["w"], n1["w"] + v2["w"]))
        n3, _ = T.server_update(g, avg1, None, 0.0, 0.5)                               # server_lr scales the step
        self.assertTrue(torch.allclose(n3["w"], torch.tensor([1.5, 2.0])))

    def test_max_local_steps_caps_optimizer_steps(self):
        data, hosp = toy(300, 2); count = {"n": 0}; real = torch.optim.Adam.step

        def counting(self, *a, **k):
            count["n"] += 1; return real(self, *a, **k)
        torch.optim.Adam.step = counting
        try:
            T.set_seed(0); m = T.make_model(5, 0, P["neural"]); rng = np.random.default_rng(0)
            T._train_epochs(m, data, np.arange(300), dict(P["neural"], batch_size=16), T._loss_fn(None), rng, 5, max_steps=7)
            capped = count["n"]; count["n"] = 0
            T._train_epochs(m, data, np.arange(300), dict(P["neural"], batch_size=16), T._loss_fn(None), rng, 1)
        finally:
            torch.optim.Adam.step = real
        self.assertEqual(capped, 7); self.assertEqual(count["n"], 19)                 # uncapped: ceil(300/16) steps for one epoch

    def test_federated_logistic_regression_learns_and_is_linear(self):
        data, hosp = toy(400, 3); tr = np.arange(0, 280); va = np.arange(280, 400)
        fcfg = dict(P["federated"], rounds=25, local_epochs=2, patience=25, lr=0.02, batch_size=32)
        g, h = T.run_fedavg(data, {k: tr[hosp[tr] == k] for k in (1, 2, 3)}, va, 5, 0, P["neural"], fcfg, 0, kind="linear")
        self.assertIsInstance(g, T.LinearNet); self.assertGreater(h["best_val"], 0.8)

    def test_stabilised_fedavg_still_learns(self):
        data, hosp = toy(400, 4); tr = np.arange(0, 280); va = np.arange(280, 400)
        ncfg = dict(P["neural"], hidden=[16])
        fcfg = dict(P["federated"], rounds=25, local_epochs=3, patience=25, lr=0.01, batch_size=32, lr_decay=0.95, max_local_steps=6, server_momentum=0.5, fedprox_mu=0.01)
        g, h = T.run_fedavg(data, {k: tr[hosp[tr] == k] for k in (1, 2, 3)}, va, 5, 0, ncfg, fcfg, 0)
        self.assertGreater(h["best_val"], 0.8)
