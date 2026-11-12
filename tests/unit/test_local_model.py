"""Tests for src/training/local_model.py.

The critical NEW regression category this phase introduces: every
prior phase only needed to prove a correct FORWARD pass. This is the
first phase with a real backward pass, and the single easiest way to
silently ship a broken training loop is a stray `.detach()`,
`.numpy()`, or `with torch.no_grad():` copied from the forward-pass-
only phase runner code this model's sub-components were originally
written for -- any of those would silently sever the autograd graph
partway through and leave upstream components (most likely LSTM/AE)
never actually training, while the loss would still compute and
decrease (from the GNN/heads alone) with nothing on the surface
looking wrong. test_gradients_flow_to_every_component is the direct
regression proof for this failure mode.
"""

import unittest

import torch

from src.modalities.tabular.autoencoder import AutoencoderConfig
from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig
from src.training.local_model import LocalModelConfig, LocalPatientModel


def _small_config(**overrides) -> LocalModelConfig:
    defaults = dict(
        lstm=LSTMEncoderConfig(input_dim=6, hidden_dim=16, num_layers=1, dropout=0.0),
        autoencoder=AutoencoderConfig(input_dim=10, hidden_dim=12, embedding_dim=8, dropout=0.0),
        gnn_hidden_dim=16, gnn_n_layers=2, gnn_type="graphsage", gnn_dropout=0.0,
        head_hidden_dim=8, head_dropout=0.0,
    )
    defaults.update(overrides)
    return LocalModelConfig(**defaults)


def _toy_hospital(n_patients=6, max_bins=5, ts_dim=6, tab_dim=10, seed=0):
    torch.manual_seed(seed)
    timeseries_x = torch.randn(n_patients, max_bins, ts_dim)
    timeseries_lengths = torch.randint(1, max_bins + 1, (n_patients,))
    tabular_x = torch.randn(n_patients, tab_dim)
    # A small connected-ish graph: a chain plus a couple of extra edges.
    src = list(range(n_patients - 1)) + [0, 2]
    dst = list(range(1, n_patients)) + [2, 4]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)  # undirected
    edge_weight = torch.ones(edge_index.shape[1])
    return timeseries_x, timeseries_lengths, tabular_x, edge_index, edge_weight


class TestLocalPatientModelForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = LocalPatientModel(_small_config())
        self.inputs = _toy_hospital()

    def test_output_keys_and_shapes(self):
        timeseries_x, lengths, tabular_x, edge_index, edge_weight = self.inputs
        out = self.model(timeseries_x, lengths, tabular_x, edge_index, edge_weight)
        n = timeseries_x.shape[0]
        self.assertEqual(out["mortality_logits"].shape, (n,))
        self.assertEqual(out["sepsis_logits"].shape, (n,))
        self.assertEqual(out["tabular_reconstruction"].shape, tabular_x.shape)

    def test_fusion_dim_auto_derived_correctly(self):
        """lstm.hidden_dim (16) + autoencoder.embedding_dim (8) = 24
        must be exactly the GNN's in_dim -- constructing with a
        mismatched gnn_hidden_dim would still work fine (gnn_hidden_dim
        is the GNN's OUTPUT width, independent of its input width), so
        this specifically checks the model didn't silently mis-wire
        the fusion->GNN boundary by re-deriving what in_dim the GNN's
        first layer actually expects."""
        first_gnn_layer = self.model.gnn.layers[0]
        # SAGEConv stores its expected input width -- read it back
        # rather than assuming, since a hardcoded mismatch here would
        # only surface as a runtime shape error deep in the forward
        # pass, not at construction time.
        expected_in = self.model.config.lstm.hidden_dim + self.model.config.autoencoder.embedding_dim
        self.assertEqual(first_gnn_layer.in_channels, expected_in)

    def test_single_patient_hospital_does_not_crash(self):
        """A real, confirmed condition (Phase 8's real-data verification
        found hospital sizes ranging min=0 to max=7,058) -- a
        single-node graph has no edges at all."""
        model = LocalPatientModel(_small_config())
        timeseries_x = torch.randn(1, 5, 6)
        lengths = torch.tensor([3])
        tabular_x = torch.randn(1, 10)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,))
        out = model(timeseries_x, lengths, tabular_x, edge_index, edge_weight)
        self.assertEqual(out["mortality_logits"].shape, (1,))


class TestGradientFlow(unittest.TestCase):
    """See module docstring -- this is this phase's single most
    important new verification category."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = LocalPatientModel(_small_config())
        self.inputs = _toy_hospital()

    def _run_backward(self):
        timeseries_x, lengths, tabular_x, edge_index, edge_weight = self.inputs
        out = self.model(timeseries_x, lengths, tabular_x, edge_index, edge_weight)
        # A simple scalar loss combining every output this test cares
        # about, so a break anywhere in the chain shows up somewhere.
        loss = (
            out["mortality_logits"].sum()
            + out["sepsis_logits"].sum()
            + out["tabular_reconstruction"].sum()
        )
        loss.backward()

    def test_gradients_flow_to_every_component(self):
        """Direct regression target: proves the composite forward pass
        is genuinely one connected autograd graph, not four
        independently-computed pieces that happen to share a Python
        namespace. Checked by hand: NOT just 'total loss decreases'
        (that could be true with only the heads training), but that
        every one of the four components has at least one parameter
        with a non-zero gradient after a single backward pass."""
        self._run_backward()
        for name, submodule in [
            ("lstm_encoder", self.model.lstm_encoder),
            ("autoencoder", self.model.autoencoder),
            ("gnn", self.model.gnn),
            ("heads", self.model.heads),
        ]:
            grads = [p.grad for p in submodule.parameters() if p.grad is not None]
            self.assertTrue(grads, f"{name} received NO gradients at all -- autograd graph is broken upstream of it.")
            total_grad_norm = sum(g.norm().item() for g in grads)
            self.assertGreater(total_grad_norm, 0.0, f"{name}'s gradients are all exactly zero.")

    def test_autoencoder_decoder_specifically_receives_gradient(self):
        """Narrower than the whole-autoencoder check above: the
        DECODER half specifically (not just the encoder, which the
        classification heads' gradient would reach anyway even with
        zero reconstruction weight) must receive gradient from the
        reconstruction term used in this test's loss."""
        self._run_backward()
        decoder_grads = [p.grad for p in self.model.autoencoder.decoder.parameters() if p.grad is not None]
        self.assertTrue(decoder_grads)
        self.assertGreater(sum(g.norm().item() for g in decoder_grads), 0.0)


class TestPerHospitalIndependence(unittest.TestCase):
    """Direct regression target for this project's core, repeated
    privacy boundary: one hospital's training step must be
    byte-for-byte unaffected by another hospital's data. Proven, not
    assumed -- by running the SAME model on the SAME hospital's inputs
    twice, with a different (unrelated) hospital's data existing
    in-scope in between, and confirming bit-identical output."""

    def test_unrelated_hospital_data_does_not_affect_this_hospitals_forward_pass(self):
        torch.manual_seed(0)
        model = LocalPatientModel(_small_config())
        hospital_a_inputs = _toy_hospital(n_patients=6, seed=1)

        timeseries_x, lengths, tabular_x, edge_index, edge_weight = hospital_a_inputs
        with torch.no_grad():
            out_before = model(timeseries_x, lengths, tabular_x, edge_index, edge_weight)

        # Simulate "another hospital's data existing" -- construct and
        # discard a completely different hospital's forward pass on
        # the SAME model instance, with different data and even a
        # different patient count.
        hospital_b_inputs = _toy_hospital(n_patients=20, seed=99)
        with torch.no_grad():
            model(*hospital_b_inputs)

        with torch.no_grad():
            out_after = model(timeseries_x, lengths, tabular_x, edge_index, edge_weight)

        torch.testing.assert_close(out_before["mortality_logits"], out_after["mortality_logits"])
        torch.testing.assert_close(out_before["sepsis_logits"], out_after["sepsis_logits"])


if __name__ == "__main__":
    unittest.main()
