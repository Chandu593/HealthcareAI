"""Generic pooled (central) focal multitask trainer shared by the GRU, autoencoder and fusion stages (NOT federated; hospitals are not used as training units).

Contract: `model_factory()` builds the model (called AFTER seeding); `model(*batch_fn(idx))` returns `(embedding, mortality_logit, sepsis_logit, aux)`; `aux` is anything the optional `aux_loss_fn(aux, idx)` needs (the autoencoder's
reconstruction). Loss = w_m * focal_m + w_s * focal_s [+ aux_weight * aux], with SEPARATE alpha/gamma per task (repo masked_focal_loss on logits) and per-task label masks
(a patient with no label for a task contributes nothing to that task). Only split 'train' updates weights; only split 'val' selects the model (best validation total loss,
weights restored); every other split (including test, whose labels are blanked upstream) is never touched. Deterministic given the seed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("eicu_fl")


def focal_multitask_loss(m_logit, s_logit, m_y, s_y, focal: dict, weights: dict):
    """-> (loss or None, {'mortality': float|None, 'sepsis': float|None}). None when neither task has a label in the batch."""
    from src.training.losses import masked_focal_loss
    parts, vals = [], {}
    for name, logit, y in (("mortality", m_logit, m_y), ("sepsis", s_logit, s_y)):
        loss, n = masked_focal_loss(logit, y, gamma=focal[name]["gamma"], alpha=focal[name]["alpha"])
        vals[name] = None if loss is None else float(loss.item())
        if loss is not None and weights[name] > 0:
            parts.append(weights[name] * loss)
    return (sum(parts) if parts else None), vals


def prepare_task_population(pats, patient_table, tcfg_window: int, landmark_only: bool, mort, sep) -> tuple:
    """(split_eff, population dict). split_eff marks non-landmark patients 'excluded' so neither updates nor early stopping can see them (they are still embedded)."""
    from src.three_modality.tcd_encoder import label_counts, landmark_mask
    split = pats["split"].to_numpy()
    landmark = landmark_mask(pats, patient_table, tcfg_window) if landmark_only else np.ones(len(pats), bool)
    split_eff = np.where(landmark, split, "excluded")
    tc = label_counts(mort, sep, split_eff)
    pop = {"landmark_only": bool(landmark_only), "rule": f"valid ICU exit >= {tcfg_window} min" if landmark_only else "all stays",
           "patients_by_split": {sp: int((split == sp).sum()) for sp in ("train", "val", "test")},
           "landmark_patients_by_split": {sp: int(((split == sp) & landmark).sum()) for sp in ("train", "val", "test")},
           "excluded_from_training_and_early_stopping": {sp: int(((split == sp) & ~landmark).sum()) for sp in ("train", "val")},
           "labels_used_for_training": {sp: tc[sp] for sp in ("train", "val")}}
    return split_eff, pop


def _val_aucs(vm, vs, mva, sva) -> dict:
    from sklearn.metrics import roc_auc_score
    out = {}
    for nm, lg, y in (("val_auroc_mortality", vm, mva), ("val_auroc_sepsis", vs, sva)):
        ok = ~np.isnan(y)
        out[nm] = float(roc_auc_score(y[ok], lg[ok])) if ok.sum() > 0 and len(np.unique(y[ok])) == 2 else None
    return out


def train_multitask(model_factory, batch_fn, mort, sep, split_eff, cfg: dict, aux_loss_fn=None, aux_weight: float = 0.0, eval_chunk: int = 4096) -> tuple:
    """`model_factory` is a zero-argument callable that BUILDS the model. The trainer seeds the RNGs and only then calls it, so the initial weights are a pure function of
    cfg['seed'] (a model built by the caller before seeding would start from whatever RNG state the process happened to be in: not reproducible across runs)."""
    import torch
    seed = int(cfg["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_factory()
    tr, va = np.flatnonzero(split_eff == "train"), np.flatnonzero(split_eff == "val")
    if len(tr) == 0:
        raise ValueError("no training patients")
    mt, st = torch.from_numpy(mort.astype(np.float32)), torch.from_numpy(sep.astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    bs, clip = int(cfg["batch_size"]), float(cfg.get("grad_clip") or 0)
    use_aux = aux_loss_fn is not None and aux_weight > 0
    best, best_state, bad = float("inf"), None, 0
    hist = {"epochs": [], "best_epoch": None, "stopped_early": False, "hyperparameters": {k: v for k, v in cfg.items()}, "seed": seed, "n_train": int(len(tr)), "n_val": int(len(va)),
            "aux_weight": float(aux_weight) if use_aux else 0.0, "monitor": "validation total loss" if len(va) else "training total loss (no validation patients)"}
    for ep in range(int(cfg["max_epochs"])):
        model.train()
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + ep)).numpy()
        tot, nb = 0.0, 0
        for i in range(0, len(tr), bs):
            idx = tr[perm[i:i + bs]]
            _, ml, sl, aux = model(*batch_fn(idx))
            loss, _ = focal_multitask_loss(ml, sl, mt[idx], st[idx], cfg["focal"], cfg["task_weights"])
            if use_aux:
                al = aux_loss_fn(aux, idx)
                loss = aux_weight * al if loss is None else loss + aux_weight * al
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep + 1}")
            opt.zero_grad(); loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += float(loss.item()); nb += 1
        model.eval()
        rec = {"epoch": ep + 1, "train_loss": tot / max(nb, 1)}
        with torch.no_grad():
            if len(va):
                vm, vs, aux_sum = [], [], 0.0
                for i in range(0, len(va), eval_chunk):
                    idx = va[i:i + eval_chunk]
                    _, ml, sl, aux = model(*batch_fn(idx))
                    vm.append(ml); vs.append(sl)
                    if use_aux:
                        aux_sum += float(aux_loss_fn(aux, idx).item()) * len(idx)
                vm, vs = torch.cat(vm), torch.cat(vs)
                floss, _ = focal_multitask_loss(vm, vs, mt[va], st[va], cfg["focal"], cfg["task_weights"])
                focal_v = float(floss.item()) if floss is not None else 0.0
                aux_v = aux_sum / len(va) if use_aux else 0.0
                rec.update({"val_loss": focal_v + aux_weight * aux_v, "val_focal": focal_v, **({"val_aux": aux_v} if use_aux else {})},)
                rec.update(_val_aucs(vm.numpy(), vs.numpy(), mt[va].numpy(), st[va].numpy()))
            else:
                rec["val_loss"] = rec["train_loss"]
        hist["epochs"].append(rec)
        score = rec["val_loss"]
        if score < best - 1e-9:
            best, bad, hist["best_epoch"] = score, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                hist["stopped_early"] = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hist["best_score"] = float(best)
    return model, hist


def encode_all(model, batch_fn, n: int, chunk: int = 4096) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            e, _, _, _ = model(*batch_fn(np.arange(i, min(n, i + chunk))))
            out.append(e.numpy())
    return np.concatenate(out).astype(np.float32)
