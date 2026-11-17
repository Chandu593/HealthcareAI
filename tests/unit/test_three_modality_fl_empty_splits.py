"""Regression tests for hospitals with NO patients in a split (found on the real data: a hospital with zero test patients crashed Config 4's embedding step, and a hospital with zero validation
patients was silently dropped from the federation under a misleading 'graph construction failed' reason, for LR / MLP as well)."""
import copy
import unittest
from unittest import mock

import numpy as np
import torch

from src.three_modality import fl_algorithms as fa
from src.three_modality import fl_gbdt_adapter as fga
from src.three_modality.downstream_config import DOWNSTREAM_DEFAULTS
from src.three_modality.fl_config import FL_DEFAULTS, validate_fl_cfg
from src.three_modality.fl_server import train_federated
from tests.unit._fl_synth import make_client

DS = copy.deepcopy(DOWNSTREAM_DEFAULTS)


def cfg(**rounds):
    c = validate_fl_cfg(copy.deepcopy(FL_DEFAULTS))
    c["rounds"].update(max_rounds=3, patience=3, min_rounds=1, local_steps=2)
    c["rounds"].update(rounds)
    return c


class TestEmptySplitsInTheClient(unittest.TestCase):
    def test_scaled_returns_an_empty_float32_array_for_an_empty_split_and_is_unchanged_otherwise(self):
        for n_va, n_te in ((0, 20), (20, 0), (0, 0)):
            c = make_client(11, n_tr=60, n_va=n_va, n_te=n_te, min_pos=1)
            self.assertEqual(c.scaled("val").shape, (n_va, c.dim))
            self.assertEqual(c.scaled("test").shape, (n_te, c.dim))
            self.assertEqual(c.scaled("test").dtype, np.float32)
        c = make_client(11, n_tr=60, n_va=10, n_te=10)
        np.testing.assert_allclose(c.scaled("val"), c.scaler.transform(c.Z["val"]).astype(np.float32))

    def test_lr_and_mlp_clients_with_an_empty_split_are_kept_never_dropped_and_forward_returns_nothing(self):
        for fam in ("lr", "mlp"):
            clients = [make_client(11, n_tr=60, n_va=0, n_te=20, seed=1, min_pos=1), make_client(12, n_tr=60, n_va=20, n_te=0, seed=2, min_pos=1), make_client(13, n_tr=60, seed=3)]
            problems, failed = fa.make_problems(clients, fam, DS)
            self.assertEqual((sorted(problems), failed), ([11, 12, 13], []), f"{fam}: a hospital with an empty split must still be a client")
            model = fa.build_model(fam, clients[0].dim, DS, 0)
            self.assertEqual(problems[11].logits(model, "val").shape, (0,))
            self.assertEqual(problems[12].logits(model, "test").shape, (0,))

    def test_gcn_clients_with_an_empty_split_build_score_and_embed(self):
        hidden = DS["gcn"]["hidden"]
        for g in ("A", "B"):
            for n_va, n_te in ((0, 20), (20, 0), (0, 0)):
                with self.subTest(graph=g, n_val=n_va, n_test=n_te):
                    c = make_client(11, n_tr=60, n_va=n_va, n_te=n_te, seed=4, min_pos=1)
                    problems, failed = fa.make_problems([c], "gcn", DS, g)
                    self.assertFalse(failed)
                    pr = problems[11]
                    model = fa.build_model("gcn", c.dim, DS, 0)
                    model.eval()
                    with torch.no_grad():
                        self.assertEqual(pr.logits(model, "val").shape, (n_va,))
                        self.assertEqual(pr.logits(model, "test").shape, (n_te,))
                        self.assertEqual(pr.embeddings(model, "val").shape, (n_va, hidden))
                        self.assertEqual(pr.embeddings(model, "test").shape, (n_te, hidden))
                        emb_train = pr.embeddings(model, "train")
                        h, _ = model(pr.x_graph_train, pr.ei_tr, pr.ew_tr)
                    self.assertEqual(emb_train.shape, (60, hidden))
                    np.testing.assert_allclose(emb_train, h[: pr.n_tr].numpy(), atol=1e-5)    # train-node embeddings do not depend on the attached query nodes

    def test_only_the_gcn_family_may_drop_a_client_and_lr_mlp_construction_errors_are_loud(self):
        c = make_client(11, n_tr=60, seed=5)
        boom = mock.patch.object(fa, "ClientProblem", side_effect=RuntimeError("boom"))
        with boom:
            with self.assertRaises(RuntimeError):
                fa.make_problems([c], "lr", DS)
            with self.assertRaises(RuntimeError):
                fa.make_problems([c], "mlp", DS)
            problems, failed = fa.make_problems([c], "gcn", DS, "A")                       # a graph that cannot be built is excluded AND reported
        self.assertEqual(problems, {})
        self.assertIn("hospital-local graph construction failed", failed[0]["reason"])


class TestEmptySplitsInTraining(unittest.TestCase):
    def setUp(self):
        self.clients = [make_client(11, n_tr=90, n_va=0, n_te=20, seed=1, min_pos=1), make_client(12, n_tr=70, n_va=25, n_te=0, seed=2, min_pos=1), make_client(13, n_tr=80, seed=3),
                        make_client(14, n_tr=40, n_va=12, n_te=12, seed=4)]

    def train(self, family, method="global", graph=None):
        problems, failed = fa.make_problems(self.clients, family, DS, graph)
        self.assertFalse(failed)
        c = cfg()
        return train_federated(problems, family, DS, c, algo="fedavg", method=method, alphas=fa.make_alphas(problems, DS, c, False), seed=1, tag="t"), problems

    def test_every_client_trains_every_round_and_only_clients_with_validation_patients_score(self):
        for fam, method, g in (("lr", "global", None), ("mlp", "fedper", None), ("gcn", "global", "A")):
            with self.subTest(family=fam, method=method):
                res, problems = self.train(fam, method, g)
                self.assertEqual(res["n_clients"], 4)
                self.assertTrue(all(h["clients_updated"] == 4 for h in res["history"]), "an empty split must not stop a client from contributing its training data")
                n_val = sum(p.client.n_val for p in problems.values())
                self.assertEqual(int(res["val_pos"].sum() + res["val_neg"].sum()), n_val, "the validation histogram holds exactly the clients' validation patients")

    def test_fedgbdt_trains_and_scores_with_empty_validation_and_test_rows(self):
        c = cfg()
        c["gbdt"].update(max_rounds=4, early_stopping_rounds=3)
        r = fga.train_fedgbdt(self.clients, c["gbdt"], c["validation"], 3)
        self.assertEqual(r["margins"][11]["val"].shape, (0,))
        self.assertEqual(r["margins"][12]["test"].shape, (0,))
        self.assertEqual(r["margins"][13]["test"].shape, (self.clients[2].n_test,))
        self.assertEqual(r["n_clients"], 4)


if __name__ == "__main__":
    unittest.main()
