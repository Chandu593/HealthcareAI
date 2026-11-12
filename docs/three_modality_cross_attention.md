# Three-token cross-attention (stage `cross_attention`)

`src/three_modality/cross_attention3.py` (sibling of the untouched two-token `src/fusion/cross_attention_fusion.py`).

**Inputs (frozen, never updated):** GRU 128-d, tabular autoencoder 64-d, TCD 64-d embeddings from three completed encoder runs. Each embedding file is hash-verified against its encoder
manifest and re-indexed **by patient ID** to the TCD patient order (duplicate / missing / extra / non-finite = error). `verify_upstream` also checks that all three encoders come from the
same incident-label run and the same TCD feature run, and that `patients.csv` (ids, order, split, hospitalid) is identical across the feature / GRU-input / tabular-input runs.

**Model:** per token LayerNorm -> Linear to `attn_dim`; learned token-type embedding; token order fixed **GRU -> Tabular -> TCD**; ONE multi-head self-attention block across the three tokens,
residual + LayerNorm; flatten + the three availability flags -> Linear -> GELU -> LayerNorm = the fused embedding (`output_dim`, default 128); two linear heads (mortality, incident sepsis)
supply the supervised focal objective.

**Missing observations:** a modality with no observation for a patient (no lab value in the window for GRU / TCD; derived from the GRU mask and the TCD schema's `n>=1` lab features) is never
fed as zeros: its token is only the modality's type embedding, it is masked out as an attention key, its output slot is a learned vector, and the availability flags reach the output layer.
Tabular is available for every patient (demographics always exist); apachePredVar-row absence is not modelled at this level.

**Training:** centrally pooled (a centrally pretrained shared representation, **not** federated); landmark-only; only `train` updates weights; only `val` selects the checkpoint; test labels
are blanked before training. Separate focal alpha/gamma/weight per task.

**Outputs** (`outputs/three_modality/cross_attention/<run_id>/`): `fused_embeddings.npz` (patient_ids in TCD order + embeddings), `availability.npz`, `task_labels.npz` (labels incl. test, for
evaluation only), `cross_attention.pt`, `training_log.json`, `label_counts.json`, `config_snapshot.json`, `manifest.json` (encoder run IDs, dimensions, alignment checks, split identity,
training population, selected checkpoint + SHA-256).

```
python run_three_modality.py --config configs/three_modality.yaml --stage cross_attention --tcd-encoder-run-id tcde_20261101T060716Z_3ab8104a --gru-encoder-run-id grue_20261105T011752Z_11f4c5dd --tabular-ae-run-id tabae_20261105T021259Z_ff592810
```
Tests: `tests/unit/test_three_modality_cross_attention.py` (shape, patient order, token order, per-modality contribution, missing handling, gradient flow, landmark-only training, validation-only
selection, stage / manifest / provenance checks).
