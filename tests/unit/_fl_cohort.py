"""Synthetic upstream + cross-attention run + frozen cohort for the FL pipeline / leakage tests (same fixture family as the Non-FL tests)."""
import copy
from pathlib import Path

import numpy as np

from src.three_modality import cross_attention3 as ca
from src.three_modality import hw_data
from src.three_modality.downstream_config import DOWNSTREAM_DEFAULTS, get_cross_attention_cfg
from src.three_modality.fl_config import FL_DEFAULTS, validate_fl_cfg
from tests.unit._xattn_synth import build_synth_upstream

SIZES = [220, 220, 200, 24]                       # the last hospital cannot satisfy the Non-FL minimum-data rule: it must appear as fl_only, never silently dropped


def build_fl_cohort(root, per=SIZES, seed=5, sets=("fused", "gru", "tabular", "tcd")):
    root = Path(root)
    up = build_synth_upstream(root, per=list(per), seed=seed)
    xcfg = get_cross_attention_cfg({"three_modality": {"cross_attention": {"model": {"attn_dim": 8, "num_heads": 2, "output_dim": 8}, "training": {"max_epochs": 2, "batch_size": 64}}}})
    res = ca.build_cross_attention(up["runs"], up["bases"], up["patient_table"], xcfg, {"window_minutes": 1440}, root / "cross_attention")
    return up, hw_data.load_cohort(res["run_dir"], root, sets), res


def fast_ds():
    c = copy.deepcopy(DOWNSTREAM_DEFAULTS)
    c["gcn"].update(max_epochs=8, patience=4)
    c["classifiers"]["lr"].update(max_epochs=25, patience=5)
    c["classifiers"]["mlp"].update(max_epochs=25, patience=5)
    c["classifiers"]["lightgbm"].update(n_estimators=15, early_stopping_rounds=5)
    c["classifiers"]["gbdt"].update(n_estimators=10)
    return c


def fast_fl(**rounds):
    c = validate_fl_cfg(copy.deepcopy(FL_DEFAULTS))
    c["rounds"].update(max_rounds=5, patience=3, min_rounds=2, local_steps=2)
    c["rounds"].update(rounds)
    c["gbdt"].update(max_rounds=5, early_stopping_rounds=3)
    c["personalization"]["finetune"].update(max_epochs=4, patience=2)
    c["checkpoint"]["every_rounds"] = 1
    return c


def clone_cohort(cohort):
    c = copy.copy(cohort)
    c.labels = {k: v.copy() for k, v in cohort.labels.items()}
    c.embeddings = {k: v.copy() for k, v in cohort.embeddings.items()}
    return c


def poison_test_labels(cohort):
    """A copy of the cohort with EVERY test label flipped (both tasks). Anything that legitimately never reads test labels is bit-identical on the original and on this copy."""
    c = clone_cohort(cohort)
    test = c.split == "test"
    for t, y in c.labels.items():
        m = test & ~np.isnan(y)
        y[m] = 1.0 - y[m]
    return c


def expected_test(up, hospital, task):
    truth = up["mort"] if task == "mortality" else up["sep"]
    m = (up["hosp"] == hospital) & (up["split"] == "test") & up["landmark"] & ~np.isnan(truth)
    return up["ids"][m], truth[m]
