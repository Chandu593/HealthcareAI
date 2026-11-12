"""Three-token cross-attention over the three FROZEN modality embeddings (new sibling module; the existing two-token `src/fusion/cross_attention_fusion.py` is untouched).

Tokens, in this fixed order: GRU (128-d) -> Tabular autoencoder (64-d) -> TCD (64-d).
  1. each frozen embedding -> LayerNorm -> Linear into a common width `attn_dim`; a learned token-type embedding is added;
  2. a modality that has NO observation for a patient (no lab value in the window for the GRU / TCD tokens) is not fed as a zero vector: its token is replaced by a learned
     stand-in (the modality's type embedding only), it is masked out as an attention KEY, and its output slot is a learned `missing_out` vector; the 3 availability flags are also given to the output layer;
  3. ONE multi-head self-attention block across the 3 tokens, residual connection + LayerNorm;
  4. flatten (3 x attn_dim) + the 3 availability flags -> Linear -> GELU -> LayerNorm = the FUSED embedding (output_dim) for every patient;
  5. two linear heads (mortality, incident sepsis) on the fused embedding supply the supervised focal objective.

Training regime (fixed): the three encoders stay frozen (their saved embeddings are inputs, never updated); the cross-attention itself is trained CENTRALLY on the pooled training patients of
all hospitals, so this is a centrally pretrained shared representation and NOT a federated model. Only split 'train' updates weights, only split 'val' selects the checkpoint, test labels never
reach training, and with `landmark_only` (default) only patients still in the ICU at the landmark are used (all patients are still embedded). Output: one fused embedding per patient in the
established TCD patient order, plus a manifest with encoder run ids, dimensions, alignment checks, split identity, training population and the selected checkpoint.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.three_modality import artifacts
from src.three_modality.multitask_training import prepare_task_population, train_multitask
from src.three_modality.tcd_encoder import build_task_labels, label_counts, landmark_mask

logger = logging.getLogger("eicu_fl")

CROSS_ATTENTION_DEFINITION_VERSION = "cross_attention3_v1"
TOKEN_ORDER = ("gru", "tabular", "tcd")
ENCODER_EMBEDDING_FILES = {"gru": "gru_embeddings.npz", "tabular": "tab_ae_embeddings.npz", "tcd": "tcd_embeddings.npz"}


# =====================================================================================================================================
# alignment + availability (torch-free)
# =====================================================================================================================================
def align_embeddings_by_patient_id(reference_ids: np.ndarray, ids: np.ndarray, emb: np.ndarray, name: str) -> tuple:
    """Re-index `emb` (rows labelled by `ids`) to `reference_ids` by explicit patient-ID lookup. Raises on any duplicate, missing or extra patient -- never aligns by position alone.
    Returns (aligned_embeddings, report)."""
    reference_ids, ids = np.asarray(reference_ids, dtype=np.int64), np.asarray(ids, dtype=np.int64)
    if emb.shape[0] != len(ids):
        raise ValueError(f"{name}: {emb.shape[0]} embedding rows but {len(ids)} patient ids")
    for label, arr in (("reference", reference_ids), (name, ids)):
        if len(np.unique(arr)) != len(arr):
            raise ValueError(f"{label}: duplicate patient ids ({len(arr) - len(np.unique(arr))} repeated)")
    missing, extra = np.setdiff1d(reference_ids, ids), np.setdiff1d(ids, reference_ids)
    if len(missing) or len(extra):
        raise ValueError(f"{name}: patient ids differ from the reference (missing {len(missing)}, extra {len(extra)})")
    same_order = bool(np.array_equal(reference_ids, ids))
    if same_order:
        out = np.asarray(emb)
    else:
        order = np.argsort(ids, kind="stable")
        pos = np.searchsorted(ids[order], reference_ids)
        out = np.asarray(emb)[order[pos]]
        assert np.array_equal(ids[order[pos]], reference_ids)
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: embeddings contain NaN/Inf")
    return out, {"modality": name, "n_patients": int(len(ids)), "dim": int(emb.shape[1]), "stored_in_reference_order": same_order, "reordered_by_id": not same_order,
                 "missing": 0, "extra": 0, "all_finite": True}


def gru_availability(mask: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """True where the patient has at least one OBSERVED lab cell in the 24 x n_vars window (the mask channel; `mask` may be a memmap)."""
    out = np.zeros(mask.shape[0], dtype=bool)
    for i in range(0, mask.shape[0], chunk):
        out[i:i + chunk] = np.asarray(mask[i:i + chunk]).reshape(min(chunk, mask.shape[0] - i), -1).any(axis=1)
    return out


def tcd_availability(raw: np.ndarray, schema: dict, chunk: int = 20000) -> np.ndarray:
    """True where at least one lab-derived TCD statistic that needs only n>=1 measurement ('requires' == 'n>=1', kind 'lab') is finite, i.e. the patient has >= 1 lab measurement."""
    names = [f["name"] for f in schema["features"]]
    cols = [i for i, f in enumerate(schema["features"]) if f.get("kind") == "lab" and f.get("requires") == "n>=1"]
    if not cols:
        raise ValueError("TCD feature schema has no kind='lab' feature with requires='n>=1'; cannot derive observation availability")
    out = np.zeros(raw.shape[0], dtype=bool)
    for i in range(0, raw.shape[0], chunk):
        out[i:i + chunk] = np.isfinite(np.asarray(raw[i:i + chunk])[:, cols]).any(axis=1)
    assert len(names) == raw.shape[1], "TCD raw matrix width does not match its schema"
    return out


def tabular_availability(n: int) -> np.ndarray:
    """The static token is always available (demographics come from the patient table for every stay); partial missingness inside it is already encoded by the tabular inputs."""
    return np.ones(n, dtype=bool)


# =====================================================================================================================================
# model (torch, lazy)
# =====================================================================================================================================
def make_cross_attention_model(dims: tuple, mcfg: dict):
    import torch
    import torch.nn as nn

    class ThreeTokenCrossAttention(nn.Module):
        def __init__(self):
            super().__init__()
            d = int(mcfg["attn_dim"])
            self.token_order = TOKEN_ORDER
            self.dims = tuple(int(x) for x in dims)
            self.attn_dim = d
            self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(k), nn.Linear(k, d)) for k in self.dims])
            self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
            self.missing_out = nn.Parameter(torch.randn(3, d) * 0.02)      # output slot of an unavailable modality
            self.attn = nn.MultiheadAttention(d, int(mcfg["num_heads"]), dropout=float(mcfg["dropout"]), batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(float(mcfg["dropout"]))
            self.out = nn.Sequential(nn.Linear(3 * d + 3, int(mcfg["output_dim"])), nn.GELU(), nn.LayerNorm(int(mcfg["output_dim"])))
            self.mortality_head = nn.Linear(int(mcfg["output_dim"]), 1)
            self.sepsis_head = nn.Linear(int(mcfg["output_dim"]), 1)

        def tokenize(self, gru, tab, tcd, avail):
            """(B, 3, attn_dim) tokens in the fixed order GRU, Tabular, TCD. An unavailable modality's token does not depend on its input at all."""
            xs = (gru, tab, tcd)
            toks = torch.stack([self.proj[i](xs[i]) for i in range(3)], dim=1) + self.type_emb[None]
            avail = avail.to(torch.bool)
            return torch.where(avail[..., None], toks, self.type_emb[None].expand_as(toks))      # modality identity only: no measurement, and not a zero vector

        def forward(self, gru, tab, tcd, avail):
            avail = avail.to(torch.bool)
            tok = self.tokenize(gru, tab, tcd, avail)
            kpm = ~avail                                                   # True = this token is not attended to as a KEY
            kpm = kpm & ~kpm.all(dim=1, keepdim=True)                      # degenerate row with nothing available: keep all keys (their content is only the type embeddings) to avoid NaN softmax
            a, _ = self.attn(tok, tok, tok, key_padding_mask=kpm, need_weights=False)
            h = self.norm(tok + self.drop(a))
            h = torch.where(avail[..., None], h, self.missing_out[None])
            emb = self.out(torch.cat([h.flatten(1), avail.to(h.dtype)], dim=1))
            return emb, self.mortality_head(emb).squeeze(-1), self.sepsis_head(emb).squeeze(-1), None

    return ThreeTokenCrossAttention()


def embed_all(model, G, T, C, A, chunk: int = 8192) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, G.shape[0], chunk):
            sl = slice(i, i + chunk)
            e, _, _, _ = model(torch.from_numpy(G[sl]), torch.from_numpy(T[sl]), torch.from_numpy(C[sl]), torch.from_numpy(A[sl]))
            out.append(e.numpy())
    return np.concatenate(out).astype(np.float32)


# =====================================================================================================================================
# upstream provenance + alignment verification
# =====================================================================================================================================
def _read_json(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def verify_upstream(runs: dict, bases: dict) -> dict:
    """`runs` = {'tcd': tcd_encoder_dir, 'gru': gru_encoder_dir, 'tabular': tabular_ae_dir, 'incident': incident_dir}; `bases` = {'tcd_features', 'gru_inputs', 'tabular_inputs'} base dirs.
    Verifies that every encoder was trained from the SAME incident-label run, the SAME TCD feature run (hence the same permanent split manifest) and the SAME patient list / split /
    hospital assignment. Returns the report stored in the manifest; raises ValueError naming the first inconsistency."""
    man = {k: _read_json(Path(v) / "manifest.json") for k, v in runs.items()}
    for k in ("tcd", "gru", "tabular"):
        if man[k].get("incident_label_run") != man["incident"]["run_id"] or man[k].get("incident_identity") != man["incident"]["identity"]:
            raise ValueError(f"{k} encoder run {man[k]['run_id']} was trained from incident label run {man[k].get('incident_label_run')}, not {man['incident']['run_id']}")
    feat_id = man["tcd"]["feature_run"]
    gin_dir, tin_dir = Path(bases["gru_inputs"]) / man["gru"]["inputs_run"], Path(bases["tabular_inputs"]) / man["tabular"]["inputs_run"]
    feat_dir = Path(bases["tcd_features"]) / feat_id
    gin, tin, feat = (_read_json(d / "manifest.json") for d in (gin_dir, tin_dir, feat_dir))
    if gin.get("tcd_feature_run") != feat_id or tin.get("tcd_feature_run") != feat_id:
        raise ValueError(f"GRU inputs ({gin.get('tcd_feature_run')}) / tabular inputs ({tin.get('tcd_feature_run')}) were not built from the TCD feature run {feat_id} used by the TCD encoder")
    if man["gru"].get("inputs_identity") != gin["identity"] or man["tabular"].get("inputs_identity") != tin["identity"] or man["tcd"].get("feature_identity") != feat["identity"]:
        raise ValueError("an encoder manifest's recorded input identity does not match the input run found on disk")
    pats = {"tcd_features": pd.read_csv(feat_dir / "patients.csv"), "gru_inputs": pd.read_csv(gin_dir / "patients.csv"), "tabular_inputs": pd.read_csv(tin_dir / "patients.csv")}
    ref = pats["tcd_features"]
    checks = {}
    for k, p in pats.items():
        ok_ids = bool(np.array_equal(p["patientunitstayid"].to_numpy(), ref["patientunitstayid"].to_numpy()))
        ok_split = bool(np.array_equal(p["split"].to_numpy(), ref["split"].to_numpy()))
        ok_h = bool(np.array_equal(p["hospitalid"].to_numpy(), ref["hospitalid"].to_numpy()))
        checks[k] = {"same_patient_ids_and_order": ok_ids, "same_split": ok_split, "same_hospitalid": ok_h}
        if not (ok_ids and ok_split and ok_h):
            raise ValueError(f"{k} patients.csv differs from the TCD feature run's (ids/order {ok_ids}, split {ok_split}, hospitalid {ok_h})")
    inc = pd.read_csv(Path(runs["incident"]) / "incident_sepsis_labels.csv", usecols=["patientunitstayid", "hospitalid", "split"])
    m = ref[["patientunitstayid", "hospitalid", "split"]].merge(inc, on="patientunitstayid", how="left", suffixes=("", "_inc"))
    if m["split_inc"].isna().any() or not (m["split"] == m["split_inc"]).all() or not (m["hospitalid"] == m["hospitalid_inc"]).all():
        raise ValueError("the incident label run's split / hospitalid assignment differs from the TCD feature run's")
    checks["incident_labels"] = {"covers_every_patient": True, "same_split": True, "same_hospitalid": True}
    if feat.get("split_manifest_hash") != man["incident"].get("split_manifest_hash"):
        raise ValueError("split manifest hash differs between the TCD feature run and the incident label run")
    return {"incident_label_run": man["incident"]["run_id"], "tcd_feature_run": feat_id, "gru_inputs_run": gin["run_id"], "tabular_inputs_run": tin["run_id"],
            "split_manifest_hash": feat["split_manifest_hash"], "patients_checks": checks, "patients": pats, "feature_dir": str(feat_dir), "gru_inputs_dir": str(gin_dir)}


# =====================================================================================================================================
# stage
# =====================================================================================================================================
def cross_attention_identity(enc_manifests: dict, incident_manifest: dict, xcfg: dict, window_minutes: int) -> str:
    return artifacts.hash_json({"definition": CROSS_ATTENTION_DEFINITION_VERSION, "token_order": TOKEN_ORDER,
                                "encoders": {k: enc_manifests[k]["identity"] for k in TOKEN_ORDER}, "incident_label": incident_manifest["identity"],
                                "mortality_source": "hospitaldischargestatus", "model": xcfg["model"], "training": xcfg["training"], "window_minutes": window_minutes})


def build_cross_attention(runs: dict, bases: dict, patient_table: pd.DataFrame, xcfg: dict, tcfg: dict, out_base, run_id: "str | None" = None, force: bool = False) -> dict:
    t0 = time.time()
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    enc_man = {k: _read_json(Path(runs[k]) / "manifest.json") for k in TOKEN_ORDER}
    iman = _read_json(Path(runs["incident"]) / "manifest.json")
    identity = cross_attention_identity(enc_man, iman, xcfg, tcfg["window_minutes"])
    if not force:
        hit = artifacts.find_reusable_run(out_base, identity)
        if hit:
            logger.info("[CROSS_ATTN] cache HIT: reusing verified run %s (identity %s)", hit["run_id"], identity[:12])
            return {"run_dir": hit["run_dir"], "run_id": hit["run_id"], "cache_hit": True, "manifest": hit}
    rid, run_dir = artifacts.new_run_dir(out_base, "xattn", identity, run_id)
    logger.info("[CROSS_ATTN] cache MISS (identity %s) -> computing into %s", identity[:12], run_dir)
    import torch

    up = verify_upstream(runs, bases)
    ref = up["patients"]["tcd_features"]
    ids = ref["patientunitstayid"].to_numpy(np.int64)
    split, hosp = ref["split"].to_numpy(), ref["hospitalid"].to_numpy(np.int64)
    aligned, align_reports = {}, {}
    for k in TOKEN_ORDER:
        emb_path = Path(runs[k]) / ENCODER_EMBEDDING_FILES[k]
        recorded = (enc_man[k].get("files") or {}).get(ENCODER_EMBEDDING_FILES[k])
        if recorded is None:
            raise ValueError(f"{k} encoder manifest does not record a hash for {ENCODER_EMBEDDING_FILES[k]}; cannot verify provenance")
        if artifacts.sha256_file(emb_path) != recorded:
            raise ValueError(f"{emb_path} does not match the SHA-256 recorded in its encoder manifest (file replaced or modified)")
        z = np.load(emb_path)
        aligned[k], align_reports[k] = align_embeddings_by_patient_id(ids, z["patient_ids"], z["embeddings"].astype(np.float32), k)
        if align_reports[k]["dim"] != int(enc_man[k]["embedding_dim"]):
            raise ValueError(f"{k}: embedding width {align_reports[k]['dim']} != manifest embedding_dim {enc_man[k]['embedding_dim']}")
        align_reports[k]["encoder_run"] = enc_man[k]["run_id"]
        align_reports[k]["embedding_file_sha256_verified"] = True
    dims = tuple(aligned[k].shape[1] for k in TOKEN_ORDER)

    gru_mask = np.load(Path(up["gru_inputs_dir"]) / "gru_mask.npy", mmap_mode="r")
    tcd_raw = np.load(Path(up["feature_dir"]) / "tcd_features_raw.npy", mmap_mode="r")
    tcd_schema = _read_json(Path(up["feature_dir"]) / "feature_schema.json")
    if gru_mask.shape[0] != len(ids) or tcd_raw.shape[0] != len(ids):
        raise ValueError(f"GRU mask ({gru_mask.shape[0]}) / TCD raw ({tcd_raw.shape[0]}) rows do not match the {len(ids)} reference patients")
    avail = np.stack([gru_availability(gru_mask), tabular_availability(len(ids)), tcd_availability(tcd_raw, tcd_schema)], axis=1)
    avail_counts = {k: {"available": int(avail[:, i].sum()), "unavailable": int((~avail[:, i]).sum())} for i, k in enumerate(TOKEN_ORDER)}
    logger.info("[CROSS_ATTN] dims %s | availability %s", dims, avail_counts)

    inc = pd.read_csv(Path(runs["incident"]) / "incident_sepsis_labels.csv")
    mort, sep = build_task_labels(ref, patient_table, inc)
    counts = label_counts(mort, sep, split)
    tr = xcfg["training"]
    split_eff, pop = prepare_task_population(ref, patient_table, tcfg["window_minutes"], tr["landmark_only"], mort, sep)
    mort_t, sep_t = mort.copy(), sep.copy()
    mort_t[split == "test"] = np.nan
    sep_t[split == "test"] = np.nan                                         # TEST labels never reach training or checkpoint selection
    G, T, C = (np.ascontiguousarray(aligned[k], dtype=np.float32) for k in TOKEN_ORDER)
    A = np.ascontiguousarray(avail)
    batch_fn = lambda idx: (torch.from_numpy(G[idx]), torch.from_numpy(T[idx]), torch.from_numpy(C[idx]), torch.from_numpy(A[idx]))   # noqa: E731
    model, hist = train_multitask(lambda: make_cross_attention_model(dims, xcfg["model"]), batch_fn, mort_t, sep_t, split_eff, tr)
    hist["training_population"] = pop
    fused = embed_all(model, G, T, C, A)
    if not np.isfinite(fused).all() or fused.shape != (len(ids), int(xcfg["model"]["output_dim"])):
        raise ValueError(f"fused embeddings have the wrong shape/values: {fused.shape}")
    logger.info("[CROSS_ATTN] best epoch %s of %d; fused embeddings %s", hist["best_epoch"], len(hist["epochs"]), fused.shape)

    run_dir.mkdir(parents=True)
    b = io.BytesIO(); np.savez_compressed(b, patient_ids=ids, embeddings=fused)
    artifacts.atomic_write(run_dir / "fused_embeddings.npz", b.getvalue())
    b = io.BytesIO(); np.savez_compressed(b, patient_ids=ids, gru=avail[:, 0], tabular=avail[:, 1], tcd=avail[:, 2])
    artifacts.atomic_write(run_dir / "availability.npz", b.getvalue())
    landmark = landmark_mask(ref, patient_table, tcfg["window_minutes"])
    b = io.BytesIO()
    np.savez_compressed(b, patient_ids=ids, hospitalid=hosp, split=split.astype("U5"), landmark=landmark, mortality=mort.astype(np.float32), sepsis=sep.astype(np.float32))
    artifacts.atomic_write(run_dir / "task_labels.npz", b.getvalue())          # labels incl. test: used ONLY to evaluate, never to train / select (see hospital_data)
    bb = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "model_config": dict(xcfg["model"]), "dims": list(dims), "token_order": list(TOKEN_ORDER), "definition": CROSS_ATTENTION_DEFINITION_VERSION}, bb)
    artifacts.atomic_write(run_dir / "cross_attention.pt", bb.getvalue())
    artifacts.atomic_write(run_dir / "training_log.json", json.dumps(hist, indent=1, default=str))
    artifacts.atomic_write(run_dir / "label_counts.json", json.dumps(counts, indent=1))
    artifacts.atomic_write(run_dir / "config_snapshot.json", json.dumps({"model": xcfg["model"], "training": tr, "encoder_runs": {k: enc_man[k]["run_id"] for k in TOKEN_ORDER},
                                                                       "incident_label_run": iman["run_id"]}, indent=1))
    names = ["fused_embeddings.npz", "availability.npz", "task_labels.npz", "cross_attention.pt", "training_log.json", "label_counts.json", "config_snapshot.json"]
    manifest = {
        "run_id": rid, "definition_version": CROSS_ATTENTION_DEFINITION_VERSION, "identity": identity, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {n: artifacts.sha256_file(run_dir / n) for n in names},
        "regime": "central pooled training of the cross-attention on frozen encoder embeddings -- a centrally pretrained shared representation, NOT federated",
        "token_order": list(TOKEN_ORDER), "input_dims": dict(zip(TOKEN_ORDER, dims)), "embedding_dim": int(fused.shape[1]), "embedding_shape": list(fused.shape),
        "encoder_runs": {k: {"run_id": enc_man[k]["run_id"], "identity": enc_man[k]["identity"], "embedding_dim": int(enc_man[k]["embedding_dim"])} for k in TOKEN_ORDER},
        "incident_label_run": iman["run_id"], "incident_identity": iman["identity"], "tcd_feature_run": up["tcd_feature_run"], "gru_inputs_run": up["gru_inputs_run"],
        "tabular_inputs_run": up["tabular_inputs_run"],
        "alignment_checks": {"embeddings_vs_tcd_patient_order": align_reports, "upstream_patients": up["patients_checks"], "reference_patient_order": "TCD feature run patients.csv",
                             "n_patients": int(len(ids))},
        "split_identity": {"split_manifest_hash": up["split_manifest_hash"], "patients_by_split": {s: int((split == s).sum()) for s in ("train", "val", "test")}},
        "availability": avail_counts, "training_population": pop, "label_counts": counts, "model": xcfg["model"],
        "selected_checkpoint": {"best_epoch": hist["best_epoch"], "best_score": hist["best_score"], "monitor": hist["monitor"], "epochs_run": len(hist["epochs"]),
                                "state_dict_file": "cross_attention.pt", "state_dict_sha256": artifacts.sha256_file(run_dir / "cross_attention.pt")},
        "seed": hist["seed"], "cache": {"hit": False, "forced_recompute": bool(force)}, "seconds": round(time.time() - t0, 1)}
    artifacts.atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=1, default=str))
    logger.info("[CROSS_ATTN] COMPLETE in %.1fs -> %s", time.time() - t0, run_dir)
    return {"run_dir": str(run_dir), "run_id": rid, "cache_hit": False, "manifest": manifest, "history": hist}


def load_cross_attention_run(run_dir) -> dict:
    d = Path(run_dir)
    z, a, t = np.load(d / "fused_embeddings.npz"), np.load(d / "availability.npz"), np.load(d / "task_labels.npz")
    return {"manifest": _read_json(d / "manifest.json"), "patient_ids": z["patient_ids"], "fused": z["embeddings"],
            "availability": {k: a[k] for k in TOKEN_ORDER}, "labels": {k: t[k] for k in t.files}}
