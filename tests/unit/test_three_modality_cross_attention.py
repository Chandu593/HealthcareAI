"""Three-token cross-attention: shape, patient order, token order, per-modality contribution, missing-observation handling, gradient flow, landmark-only training, validation-only
checkpoint selection, and the stage's upstream/alignment verification (synthetic data only)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.three_modality import cross_attention3 as ca
from src.three_modality.downstream_config import get_cross_attention_cfg
from src.three_modality.multitask_training import train_multitask
from tests.unit._xattn_synth import build_synth_upstream

DIMS = (10, 6, 7)


def small_cfg(**train):
    over = {"model": {"attn_dim": 16, "num_heads": 2, "output_dim": 12, "dropout": 0.0},
            "training": {"max_epochs": 4, "batch_size": 64, "patience": 3, "lr": 5e-3, **train}}
    return get_cross_attention_cfg({"three_modality": {"cross_attention": over}})


def model(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    return ca.make_cross_attention_model(DIMS, {"attn_dim": 16, "num_heads": 2, "output_dim": 12, "dropout": dropout}).eval()


def batch(b=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(b, d, generator=g) for d in DIMS)


ALL = lambda b: torch.ones(b, 3, dtype=torch.bool)    # noqa: E731


class TestShapesAndOrder(unittest.TestCase):
    def test_output_shapes(self):
        m = model()
        emb, ml, sl, aux = m(*batch(8), ALL(8))
        self.assertEqual(tuple(emb.shape), (8, 12))
        self.assertEqual(tuple(ml.shape), (8,))
        self.assertEqual(tuple(sl.shape), (8,))
        self.assertIsNone(aux)
        self.assertTrue(torch.isfinite(emb).all())

    def test_token_order_is_gru_tabular_tcd(self):
        self.assertEqual(ca.TOKEN_ORDER, ("gru", "tabular", "tcd"))
        self.assertEqual(model().token_order, ("gru", "tabular", "tcd"))
        self.assertEqual(model().dims, DIMS)                                  # token 0 reads the 128-d-style GRU slot, token 1 the tabular, token 2 the TCD

    def test_each_token_depends_only_on_its_own_modality(self):
        m = model()
        g, t, c = batch(6)
        base = m.tokenize(g, t, c, ALL(6)).detach()
        n = [torch.randn_like(x) for x in (g, t, c)]                         # noise, not a constant: the input LayerNorm is (by design) invariant to a constant shift
        for i, pert in enumerate(((g + n[0], t, c), (g, t + n[1], c), (g, t, c + n[2]))):
            moved = m.tokenize(*pert, ALL(6)).detach()
            changed = [(moved[:, j] - base[:, j]).abs().max().item() > 1e-6 for j in range(3)]
            self.assertEqual(changed, [j == i for j in range(3)], f"perturbing modality {ca.TOKEN_ORDER[i]} must move only token {i}")

    def test_swapping_modalities_changes_the_output(self):
        # a model that ignored the token order (or modality identity) would give the same fused embedding for swapped same-width inputs
        torch.manual_seed(0)
        m = ca.make_cross_attention_model((6, 6, 6), {"attn_dim": 16, "num_heads": 2, "output_dim": 12, "dropout": 0.0}).eval()
        a, b, c = (torch.randn(5, 6) for _ in range(3))
        e1 = m(a, b, c, ALL(5))[0]
        e2 = m(b, a, c, ALL(5))[0]
        self.assertGreater((e1 - e2).abs().max().item(), 1e-4)

    def test_every_modality_contributes_to_the_fused_embedding(self):
        m = model()
        g, t, c = batch(6)
        base = m(g, t, c, ALL(6))[0]
        n = [torch.randn_like(x) for x in (g, t, c)]
        for i, pert in enumerate(((g + n[0], t, c), (g, t + n[1], c), (g, t, c + n[2]))):
            self.assertGreater((m(*pert, ALL(6))[0] - base).abs().max().item(), 1e-5, f"{ca.TOKEN_ORDER[i]} has no effect on the fused embedding")

    def test_row_i_of_the_output_is_patient_i(self):
        m = model()
        g, t, c = batch(9)
        avail = ALL(9)
        avail[3, 0] = False
        avail[5, 2] = False
        out = m(g, t, c, avail)[0]
        perm = torch.randperm(9, generator=torch.Generator().manual_seed(4))
        out_p = m(g[perm], t[perm], c[perm], avail[perm])[0]
        self.assertTrue(torch.allclose(out[perm], out_p, atol=1e-5), "outputs are not row-wise (patient-wise) independent / order-preserving")


class TestAlignment(unittest.TestCase):
    def test_reorders_by_id_not_position(self):
        ref = np.array([30, 10, 20, 40])
        ids = np.array([40, 20, 10, 30])
        emb = np.array([[4.0], [2.0], [1.0], [3.0]], dtype=np.float32)       # value = id/10
        out, rep = ca.align_embeddings_by_patient_id(ref, ids, emb, "gru")
        self.assertEqual(out[:, 0].tolist(), [3.0, 1.0, 2.0, 4.0])
        self.assertTrue(rep["reordered_by_id"] and not rep["stored_in_reference_order"])

    def test_already_aligned_is_reported(self):
        ref = np.array([5, 6, 7])
        out, rep = ca.align_embeddings_by_patient_id(ref, ref.copy(), np.zeros((3, 2), np.float32), "tcd")
        self.assertTrue(rep["stored_in_reference_order"])

    def test_missing_extra_duplicate_and_nan_are_errors(self):
        ref, e = np.array([1, 2, 3]), np.zeros((3, 2), np.float32)
        with self.assertRaisesRegex(ValueError, "missing 1"):
            ca.align_embeddings_by_patient_id(ref, np.array([1, 2, 9]), e, "x")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ca.align_embeddings_by_patient_id(ref, np.array([1, 1, 3]), e, "x")
        with self.assertRaisesRegex(ValueError, "embedding rows"):
            ca.align_embeddings_by_patient_id(ref, np.array([1, 2]), e, "x")
        bad = e.copy(); bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            ca.align_embeddings_by_patient_id(ref, ref, bad, "x")


class TestMissingObservations(unittest.TestCase):
    def test_unavailable_modality_input_is_ignored_completely(self):
        m = model()
        g, t, c = batch(6)
        avail = ALL(6)
        avail[:, 0] = False
        e1 = m(g, t, c, avail)[0]
        e2 = m(g * 100 + 7, t, c, avail)[0]                                  # arbitrary garbage in the unavailable GRU slot
        self.assertTrue(torch.equal(e1, e2), "an unavailable modality's values leaked into the fused embedding")

    def test_missing_is_not_a_zero_measurement(self):
        m = model()
        g, t, c = batch(6)
        avail_missing = ALL(6); avail_missing[:, 2] = False
        e_missing = m(g, t, c, avail_missing)[0]
        e_zero = m(g, t, torch.zeros_like(c), ALL(6))[0]                    # a genuine all-zero TCD measurement vector
        self.assertGreater((e_missing - e_zero).abs().max().item(), 1e-4)

    def test_unavailable_token_carries_no_input_content(self):
        m = model()
        g, t, c = batch(4)
        avail = ALL(4); avail[1, 1] = False
        tok = m.tokenize(g, t, c, avail).detach()
        self.assertTrue(torch.allclose(tok[1, 1], m.type_emb[1].detach()))      # only the modality identity, no measurement
        self.assertFalse(torch.allclose(tok[0, 1], tok[1, 1]))

    def test_availability_flags_reach_the_output_layer(self):
        m = model()
        g, t, c = batch(3)
        a1, a2 = ALL(3), ALL(3)
        a2[:, 0] = False
        self.assertGreater((m(g, t, c, a1)[0] - m(g, t, c, a2)[0]).abs().max().item(), 1e-5)

    def test_row_with_nothing_available_is_finite(self):
        m = model()
        g, t, c = batch(3)
        out = m(g, t, c, torch.zeros(3, 3, dtype=torch.bool))[0]
        self.assertTrue(torch.isfinite(out).all())

    def test_unavailable_rows_do_not_change_available_rows(self):
        m = model()
        g, t, c = batch(5)
        avail = ALL(5)
        base = m(g, t, c, avail)[0]
        avail2 = avail.clone(); avail2[0, 1] = False                        # only patient 0 loses a modality
        out2 = m(g, t, c, avail2)[0]
        self.assertTrue(torch.allclose(base[1:], out2[1:], atol=1e-6))
        self.assertGreater((base[0] - out2[0]).abs().max().item(), 1e-5)

    def test_availability_from_gru_mask(self):
        mask = np.zeros((4, 3, 2), np.uint8)
        mask[1, 2, 1] = 1
        mask[3, 0, 0] = 1
        self.assertEqual(ca.gru_availability(mask, chunk=2).tolist(), [False, True, False, True])

    def test_availability_from_tcd_schema_uses_only_n_ge_1_lab_features(self):
        schema = {"features": [{"name": "a", "kind": "lab", "requires": "n>=1"}, {"name": "b", "kind": "lab", "requires": "always defined"},
                               {"name": "age", "kind": "patient-level", "requires": "always defined"}]}
        raw = np.array([[np.nan, 5.0, 40.0], [1.0, 5.0, 40.0], [np.nan, 0.0, 50.0]], dtype=np.float32)
        self.assertEqual(ca.tcd_availability(raw, schema).tolist(), [False, True, False])   # always-defined columns must not make a no-lab patient 'available'
        with self.assertRaisesRegex(ValueError, "cannot derive observation availability"):
            ca.tcd_availability(raw, {"features": [{"name": "age", "kind": "patient-level", "requires": "always defined"}]})
        self.assertTrue(ca.tabular_availability(5).all())


class TestGradientFlow(unittest.TestCase):
    def test_gradients_reach_every_parameter_and_all_three_inputs(self):
        torch.manual_seed(0)
        m = ca.make_cross_attention_model(DIMS, {"attn_dim": 16, "num_heads": 2, "output_dim": 12, "dropout": 0.0}).train()
        xs = [x.clone().requires_grad_(True) for x in batch(12, seed=3)]
        avail = ALL(12)
        avail[0, 0] = avail[1, 1] = avail[2, 2] = False                    # every modality is unavailable for at least one patient -> its stand-in tokens are exercised
        emb, ml, sl, _ = m(*xs, avail)
        (emb.pow(2).sum() + ml.sum() + sl.sum()).backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} got no gradient")
            self.assertGreater(p.grad.abs().max().item(), 0.0, f"{name} received an all-zero gradient")
        for i, x in enumerate(xs):
            g = x.grad
            self.assertIsNotNone(g)
            rows_avail = avail[:, i]
            self.assertGreater(g[rows_avail].abs().max().item(), 0.0, f"input token {ca.TOKEN_ORDER[i]} got no gradient")
            self.assertEqual(g[~rows_avail].abs().max().item(), 0.0, f"gradient reached an UNAVAILABLE {ca.TOKEN_ORDER[i]} input")


def _toy_training_data(n=240, seed=0):
    rs = np.random.default_rng(seed)
    G, T, C = (rs.normal(size=(n, d)).astype(np.float32) for d in DIMS)
    mort = (rs.random(n) < 0.3).astype(float)
    sep = (rs.random(n) < 0.3).astype(float)
    G[:, 0] += 2 * mort
    split = np.array(["train"] * 140 + ["val"] * 50 + ["test"] * 50)
    A = np.ones((n, 3), bool)
    return G, T, C, A, mort, sep, split


def _fit(G, T, C, A, mort, sep, split_eff, max_epochs=6, patience=100, lr=5e-3):
    cfg = copy.deepcopy(small_cfg(max_epochs=max_epochs, patience=patience, lr=lr)["training"])
    bf = lambda idx: (torch.from_numpy(G[idx]), torch.from_numpy(T[idx]), torch.from_numpy(C[idx]), torch.from_numpy(A[idx]))   # noqa: E731
    return train_multitask(lambda: ca.make_cross_attention_model(DIMS, {"attn_dim": 16, "num_heads": 2, "output_dim": 12, "dropout": 0.0}), bf, mort, sep, split_eff, cfg)


def _state(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


class TestTrainingPopulationAndSelection(unittest.TestCase):
    def setUp(self):
        self.G, self.T, self.C, self.A, self.mort, self.sep, self.split = _toy_training_data()
        self.landmark = np.ones(len(self.split), bool)
        self.landmark[::4] = False
        self.split_eff = np.where(self.landmark, self.split, "excluded")

    def test_non_landmark_patients_never_train_or_select(self):
        m1, h1 = _fit(self.G, self.T, self.C, self.A, self.mort, self.sep, self.split_eff)
        mort2, sep2 = self.mort.copy(), self.sep.copy()
        mort2[~self.landmark], sep2[~self.landmark] = 1 - mort2[~self.landmark], 1 - sep2[~self.landmark]
        G2 = self.G.copy(); G2[~self.landmark] += 50.0                       # corrupted features of non-landmark patients too
        m2, h2 = _fit(G2, self.T, self.C, self.A, mort2, sep2, self.split_eff)
        self.assertTrue(_same(_state(m1), _state(m2)))
        self.assertEqual(h1["n_train"], int(((self.split == "train") & self.landmark).sum()))
        self.assertEqual(h1["n_val"], int(((self.split == "val") & self.landmark).sum()))

    def test_test_labels_never_influence_training_or_selection(self):
        m1, h1 = _fit(self.G, self.T, self.C, self.A, self.mort, self.sep, self.split_eff)
        mort2, sep2 = self.mort.copy(), self.sep.copy()
        t = self.split == "test"
        mort2[t], sep2[t] = 1 - mort2[t], 1 - sep2[t]
        m2, h2 = _fit(self.G, self.T, self.C, self.A, mort2, sep2, self.split_eff)
        self.assertTrue(_same(_state(m1), _state(m2)))
        self.assertEqual(h1["best_epoch"], h2["best_epoch"])

    def test_validation_labels_do_not_drive_the_weight_updates(self):
        m1, _ = _fit(self.G, self.T, self.C, self.A, self.mort, self.sep, self.split_eff, max_epochs=1)
        mort2, sep2 = self.mort.copy(), self.sep.copy()
        v = self.split == "val"
        mort2[v], sep2[v] = 1 - mort2[v], 1 - sep2[v]
        m2, _ = _fit(self.G, self.T, self.C, self.A, mort2, sep2, self.split_eff, max_epochs=1)
        self.assertTrue(_same(_state(m1), _state(m2)), "validation labels changed the epoch-1 weights (they must only select the checkpoint)")

    def test_checkpoint_is_the_validation_loss_minimum(self):
        # tiny training set + a high learning rate: validation loss turns around, so the selected epoch is strictly before the last one
        rs = np.random.default_rng(3)
        n = 260
        G, T, C = (rs.normal(size=(n, d)).astype(np.float32) for d in DIMS)
        mort, sep = (rs.random(n) < 0.4).astype(float), (rs.random(n) < 0.4).astype(float)     # pure label noise -> overfits
        split_eff = np.array(["train"] * 30 + ["val"] * 130 + ["test"] * 100)
        A = np.ones((n, 3), bool)
        full, h = _fit(G, T, C, A, mort, sep, split_eff, max_epochs=25, patience=1000, lr=2e-2)
        vl = [e["val_loss"] for e in h["epochs"]]
        self.assertEqual(h["best_epoch"], int(np.argmin(vl)) + 1)
        self.assertLess(h["best_epoch"], len(vl), "fixture did not overfit; the test would be vacuous")
        capped, _ = _fit(G, T, C, A, mort, sep, split_eff, max_epochs=h["best_epoch"], patience=1000, lr=2e-2)
        self.assertTrue(_same(_state(full), _state(capped)), "the returned weights are not the best-validation-epoch weights")


class TestStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.up = build_synth_upstream(self.root / "up", seed=1)
        self.tcfg = {"window_minutes": 1440}
        self.cfg = small_cfg()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, up=None, out="out", force=True, run_id=None, patient_table=None):
        up = up or self.up
        return ca.build_cross_attention(up["runs"], up["bases"], up["patient_table"] if patient_table is None else patient_table, self.cfg, self.tcfg,
                                        self.root / out, run_id=run_id, force=force)

    def test_end_to_end_shape_order_manifest(self):
        res = self._run()
        m = res["manifest"]
        z = np.load(Path(res["run_dir"]) / "fused_embeddings.npz")
        self.assertEqual(z["embeddings"].shape, (len(self.up["ids"]), 12))
        self.assertTrue(np.array_equal(z["patient_ids"], self.up["ids"]), "fused rows are not in the TCD patient order")
        self.assertEqual(m["token_order"], ["gru", "tabular", "tcd"])
        self.assertEqual(m["input_dims"], {"gru": 10, "tabular": 6, "tcd": 7})
        for k in ("encoder_runs", "alignment_checks", "split_identity", "training_population", "selected_checkpoint", "availability", "incident_label_run"):
            self.assertIn(k, m)
        self.assertEqual(m["encoder_runs"]["gru"]["run_id"], "grue_synth")
        self.assertTrue(all(m["alignment_checks"]["embeddings_vs_tcd_patient_order"][k]["embedding_file_sha256_verified"] for k in ca.TOKEN_ORDER))
        self.assertEqual(m["selected_checkpoint"]["state_dict_sha256"], __import__("hashlib").sha256((Path(res["run_dir"]) / "cross_attention.pt").read_bytes()).hexdigest())
        self.assertIn("NOT federated", m["regime"])
        self.assertEqual(m["split_identity"]["patients_by_split"], {s: int((self.up["split"] == s).sum()) for s in ("train", "val", "test")})

    def test_availability_matches_the_no_lab_patients(self):
        res = self._run()
        a = np.load(Path(res["run_dir"]) / "availability.npz")
        self.assertTrue(np.array_equal(a["gru"], ~self.up["no_lab"]))
        self.assertTrue(np.array_equal(a["tcd"], ~self.up["no_lab"]))
        self.assertTrue(a["tabular"].all())

    def test_training_population_is_landmark_only(self):
        pop = self._run()["manifest"]["training_population"]
        self.assertTrue(pop["landmark_only"])
        lm = pop["landmark_patients_by_split"]
        for s in ("train", "val", "test"):
            self.assertEqual(lm[s], int(((self.up["split"] == s) & self.up["landmark"]).sum()))
        self.assertEqual(self._run(force=True)["history"]["n_train"], lm["train"])

    def test_storage_order_of_a_modality_does_not_change_the_result(self):
        a = self._run()
        other = build_synth_upstream(self.root / "up2", seed=1, shuffle_gru=True, shuffle_tab=True)
        b = ca.build_cross_attention(other["runs"], other["bases"], other["patient_table"], self.cfg, self.tcfg, self.root / "out2", force=True)
        fa, fb = (np.load(Path(r["run_dir"]) / "fused_embeddings.npz") for r in (a, b))
        self.assertTrue(np.array_equal(fa["patient_ids"], fb["patient_ids"]))
        self.assertTrue(np.allclose(fa["embeddings"], fb["embeddings"], atol=1e-6), "shuffling a stored embedding file changed the fused result: alignment is not by patient ID")
        self.assertTrue(b["manifest"]["alignment_checks"]["embeddings_vs_tcd_patient_order"]["gru"]["reordered_by_id"])

    def test_test_labels_and_non_landmark_labels_never_reach_training(self):
        base = self._run(out="out_a")
        pt = self.up["patient_table"].copy()
        sp = pd.Series(self.up["split"], index=self.up["ids"])
        lm = pd.Series(self.up["landmark"], index=self.up["ids"])
        flip = pt["patientunitstayid"].map(lambda i: sp[i] == "test" or not lm[i]).to_numpy()
        pt.loc[flip, "hospitaldischargestatus"] = pt.loc[flip, "hospitaldischargestatus"].map({"Alive": "Expired", "Expired": "Alive"})
        again = self._run(out="out_b", patient_table=pt)
        fa, fb = (np.load(Path(r["run_dir"]) / "fused_embeddings.npz")["embeddings"] for r in (base, again))
        self.assertTrue(np.array_equal(fa, fb))
        self.assertEqual(base["manifest"]["selected_checkpoint"]["state_dict_sha256"], again["manifest"]["selected_checkpoint"]["state_dict_sha256"])

    def test_same_seed_is_reproducible(self):
        a, b = self._run(out="r1"), self._run(out="r2")
        self.assertEqual(a["manifest"]["selected_checkpoint"]["state_dict_sha256"], b["manifest"]["selected_checkpoint"]["state_dict_sha256"])

    def test_cache_hit_and_never_overwrite(self):
        first = self._run(force=False, run_id="xattn_one")
        again = self._run(force=False)
        self.assertTrue(again["cache_hit"])
        self.assertEqual(again["run_id"], "xattn_one")
        with self.assertRaises(FileExistsError):
            self._run(force=True, run_id="xattn_one")

    def test_tampered_embedding_file_is_rejected(self):
        p = Path(self.up["runs"]["tcd"]) / "tcd_embeddings.npz"
        z = np.load(p)
        np.savez_compressed(p, patient_ids=z["patient_ids"], embeddings=z["embeddings"] + 1.0)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._run()

    def test_encoders_from_different_label_runs_are_rejected(self):
        mp = Path(self.up["runs"]["gru"]) / "manifest.json"
        m = json.loads(mp.read_text()); m["incident_label_run"] = "inc_other"; mp.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "incident label run"):
            self._run()

    def test_inputs_with_a_different_split_are_rejected(self):
        p = Path(self.up["bases"]["gru_inputs"]) / "gruin_synth" / "patients.csv"
        d = pd.read_csv(p)
        d.loc[0, "split"] = "train" if d.loc[0, "split"] != "train" else "test"
        d.to_csv(p, index=False)
        with self.assertRaisesRegex(ValueError, "differs from the TCD feature run"):
            self._run()


if __name__ == "__main__":
    unittest.main()
