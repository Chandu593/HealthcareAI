"""
Ablation framework -- config-driven experiment grid expansion.

Confirmed with the user (ablation-framework decision #12): the grid
must be DERIVED from `configs/base.yaml`'s `ablation:` stanza, never
hardcoded in Python -- changing `ablation.ts_models` / `tab_models` /
`fusion_modes` / `classifiers` immediately changes the experiment
matrix with no code change required. `ablation.subset` (optional) lets
someone run a smaller, explicit slice of the full grid (e.g. for a
quick pilot check of 2 combos) without touching this file at all.

Two independent grids are produced, matching decision #6 and #7 (the
72-config grid is NOT multiplied by GNN type or federated algorithm,
and the graph-embedding ablation is a SEPARATE, smaller experiment
using the existing default architecture's trained GNN output, not a
second cross product):

  - `expand_embedding_experiment_grid()`: Section 1's TS x TAB x
    Fusion x Classifier x Task grid (72 combos x 2 tasks = 144
    experiments with the full default lists).
  - `expand_graph_experiment_grid()`: Section 2's graph-embedding
    ablation -- one fixed representation source x graph_classifiers x
    task (6 experiments with the full default lists).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from src.classifiers.boosting_classifiers import VALID_CLASSIFIERS
from src.training.flat_multimodal_model import VALID_FUSION_MODES, VALID_TAB_MODELS, VALID_TS_MODELS

VALID_TASKS = ("mortality", "sepsis")


@dataclass(frozen=True)
class EmbeddingComboSpec:
    """One point in the TS x TAB x Fusion grid (Section 1). Classifier
    and task are a separate, later axis (see EmbeddingExperimentSpec)
    since they don't require re-training the encoders at all --
    decision #1 fits classifiers on the SAME frozen embedding this
    combo produces, once, regardless of how many classifiers/tasks are
    evaluated against it."""

    ts_model: str
    tab_model: str
    fusion_mode: str

    @property
    def combo_id(self) -> str:
        fusion_label = "beforeca" if self.fusion_mode == "concatenation" else "afterca"
        return f"ts-{self.ts_model}_tab-{self.tab_model}_fusion-{fusion_label}"


@dataclass(frozen=True)
class EmbeddingExperimentSpec:
    """One fully-specified experiment: a trained-embedding combo,
    evaluated by one classifier, for one task -- the unit the results
    writer persists one record per (see src/ablation/results_writer.py)."""

    combo: EmbeddingComboSpec
    classifier: str
    task: str

    @property
    def experiment_id(self) -> str:
        return f"{self.combo.combo_id}_clf-{self.classifier}_task-{self.task}"


@dataclass(frozen=True)
class GraphExperimentSpec:
    """One graph-embedding-ablation experiment (Section 2): the single,
    fixed representation source (the existing, locked default
    architecture's trained GNN output) evaluated by one classifier, for
    one task."""

    classifier: str
    task: str

    @property
    def experiment_id(self) -> str:
        return f"graph-embedding_clf-{self.classifier}_task-{self.task}"


def validate_ablation_config(ablation_cfg: dict) -> None:
    """Raises ValueError immediately on any unrecognized model/
    classifier name in the config -- a typo in `ablation.ts_models`
    should fail loudly at grid-expansion time, not silently produce an
    empty or partial grid three hours into a real run."""
    for key, valid in (
        ("ts_models", VALID_TS_MODELS), ("tab_models", VALID_TAB_MODELS),
        ("fusion_modes", VALID_FUSION_MODES), ("classifiers", VALID_CLASSIFIERS),
        ("graph_classifiers", VALID_CLASSIFIERS),
    ):
        for value in ablation_cfg.get(key, []):
            if value not in valid:
                raise ValueError(f"ablation.{key} contains {value!r}, which is not one of {valid}.")


def expand_embedding_combo_grid(ablation_cfg: dict) -> "list[EmbeddingComboSpec]":
    """Returns the TS x TAB x Fusion grid -- or the explicit, filtered
    subset from `ablation.subset.combos` if given (decision #12: no
    code change needed to run a smaller slice)."""
    explicit = (ablation_cfg.get("subset") or {}).get("combos")
    if explicit:
        return [
            EmbeddingComboSpec(ts_model=c["ts_model"], tab_model=c["tab_model"], fusion_mode=c["fusion_mode"])
            for c in explicit
        ]
    ts_models = ablation_cfg.get("ts_models", [])
    tab_models = ablation_cfg.get("tab_models", [])
    fusion_modes = ablation_cfg.get("fusion_modes", [])
    return [
        EmbeddingComboSpec(ts_model=ts, tab_model=tab, fusion_mode=fusion)
        for ts, tab, fusion in itertools.product(ts_models, tab_models, fusion_modes)
    ]


def expand_embedding_experiment_grid(ablation_cfg: dict) -> "list[EmbeddingExperimentSpec]":
    combos = expand_embedding_combo_grid(ablation_cfg)
    subset = ablation_cfg.get("subset") or {}
    classifiers = subset.get("classifiers") or ablation_cfg.get("classifiers", [])
    tasks = subset.get("tasks") or list(VALID_TASKS)
    return [
        EmbeddingExperimentSpec(combo=combo, classifier=clf, task=task)
        for combo, clf, task in itertools.product(combos, classifiers, tasks)
    ]


def expand_graph_experiment_grid(ablation_cfg: dict) -> "list[GraphExperimentSpec]":
    subset = ablation_cfg.get("subset") or {}
    classifiers = subset.get("graph_classifiers") or ablation_cfg.get("graph_classifiers", [])
    tasks = subset.get("tasks") or list(VALID_TASKS)
    return [GraphExperimentSpec(classifier=clf, task=task) for clf, task in itertools.product(classifiers, tasks)]
