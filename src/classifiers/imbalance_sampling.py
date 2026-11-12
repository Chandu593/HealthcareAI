"""
Class-imbalance sampling for the ablation framework's classifier stage
(confirmed with the user 2026-09-20).

Scope and constraints, all confirmed rather than assumed:

- Applies ONLY to TRAIN-split rows. Validation and test rows are never
  touched by anything in this module -- callers pass already-split
  X_train/y_train separately, and `search_sampling_ratio()`'s
  X_val/y_val/X_test/y_test are evaluated as-is, unresampled.
- HOSPITAL-WISE: resampling is computed independently within each
  hospital's own train rows (never across hospitals -- the same
  privacy boundary every other phase in this project enforces), then
  the per-hospital resampled rows are concatenated back together for
  the (pooled, cross-hospital) classifier fit -- matching how
  `fit_and_evaluate_classifier()` already fits one global classifier
  on pooled embeddings, per ablation-framework decision #1.
- Simple RANDOM over/undersampling only, fixed seed, no SMOTE or other
  synthetic-data technique (explicitly ruled out by the user).

  UPDATE 2026-10-03: the no-synthetic-data rule was explicitly revisited
  and lifted for this phase, after the user's own real-data finding that
  random under/oversampling underperform the flat no_sampling baseline
  on most (combo, classifier, task) groups. Three ADVANCED strategies
  were added -- `"adasyn"`, `"enn"`, `"adasyn_enn"` -- via the
  `imbalanced-learn` library (a new, confirmed dependency; see
  requirements.txt). The HOSPITAL-WISE and TRAIN-ONLY constraints above
  apply identically to these: ADASYN's/ENN's k-nearest-neighbor
  computation is run independently within each hospital's own train
  rows, never pooled across hospitals, for the exact same privacy-
  boundary reason the original random strategies already respect it.
  The original `"undersample"`/`"oversample"` strategies are UNCHANGED
  and kept as the random baseline this comparison is measured against,
  not replaced.
- The sampling RATIO (not the classifier's own hyperparameters, which
  the user confirmed stay at the existing tuned `ablation.
  classifier_params` values) is the one thing tuned here, via a small
  validation-driven search rather than a single hardcoded default --
  "best, efficient, research-grade... not default" per the user's own
  framing.
- Selection priority across ratio-search candidates, and across
  sampling strategies more generally, is F1 (primary) -> recall
  (secondary) -> precision -> AUROC -> AUPRC -> balanced accuracy ->
  specificity -> Brier score (lower is better), all confirmed with the
  user 2026-09-20 to replace AUPRC (already used once, for the
  no-sampling ablation) as the selection criterion for this
  comparison.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

from src.classifiers.boosting_classifiers import ClassifierAblationResult, fit_and_evaluate_classifier

# The two ORIGINAL (random, no-synthetic-data) strategies -- unchanged,
# kept as the baseline the advanced strategies below are measured
# against, not replaced by them.
RANDOM_SAMPLING_STRATEGIES = ("undersample", "oversample")
# Advanced strategies added 2026-10-03 (see module docstring "UPDATE"
# note for why/when the no-synthetic-data rule was revisited).
# "adasyn": oversample only, via ADASYN's own difficulty-adaptive
#   synthetic generation (more synthetic points for minority examples
#   that are HARDER to learn -- i.e. surrounded by more majority
#   neighbors -- rather than uniform interpolation like plain SMOTE).
# "enn": undersample/clean only, via Edited Nearest Neighbours --
#   removes MAJORITY rows whose own k-NN neighborhood disagrees with
#   their label (noise/boundary cleaning), rather than removing
#   majority rows at random.
# "adasyn_enn": both in sequence, within each hospital independently --
#   ADASYN oversamples the minority class first, then ENN cleans the
#   result -- the same two-step shape as imblearn's own SMOTEENN, with
#   ADASYN in place of SMOTE given sepsis_strict's extreme (~0.1%)
#   imbalance, where ADASYN's adaptive generation is the better-suited
#   of the two per its own difficulty-focused design.
ADVANCED_SAMPLING_STRATEGIES = ("adasyn", "enn", "adasyn_enn")
VALID_SAMPLING_STRATEGIES = ("no_sampling",) + RANDOM_SAMPLING_STRATEGIES + ADVANCED_SAMPLING_STRATEGIES

# Selection priority order (confirmed with the user 2026-09-20):
# F1 primary, recall next, everything else after. "balanced_accuracy"
# is derived from sensitivity/specificity when not already present
# (compute_binary_classification_metrics() doesn't compute it
# directly today) rather than treated as always-missing.
_METRIC_PRIORITY = ("f1", "recall", "precision", "auroc", "auprc", "balanced_accuracy", "specificity", "brier_score")
_LOWER_IS_BETTER = {"brier_score"}


def _adasyn_one_hospital(
    X_h: np.ndarray, y_h: np.ndarray, minority_class, ratio: float, seed: int, hid, logger: logging.Logger,
) -> "tuple[np.ndarray, np.ndarray]":
    """ADASYN for ONE hospital's own rows. Falls back to this module's
    existing plain random oversample (never raises, never crashes the
    combo) whenever ADASYN's own requirements aren't met by this
    hospital's data -- real eICU hospitals range from 0 to 7,058
    patients (Phase 8's confirmed real distribution), so a hospital
    with only 1-4 minority-class train rows is an expected, not
    hypothetical, case here.
    """
    classes, counts = np.unique(y_h, return_counts=True)
    n_minority = int(counts[classes == minority_class][0])
    # ADASYN needs n_neighbors+1 minority samples at minimum; cap at 5
    # (imblearn's own default) and skip straight to the fallback below
    # 2 minority samples, where no neighbor relationship can exist at
    # all.
    k = min(5, n_minority - 1)
    if k < 1:
        logger.info(
            f"[ADASYN] hospital={hid}: only {n_minority} minority train row(s) -- too few for any "
            f"neighbor-based synthesis; falling back to plain random oversample for this hospital only."
        )
        return hospital_wise_resample(X_h, y_h, np.full(len(y_h), hid), "oversample", ratio, seed)
    try:
        sampler = ADASYN(sampling_strategy=ratio, random_state=seed, n_neighbors=k)
        return sampler.fit_resample(X_h, y_h)
    except Exception as exc:  # noqa: BLE001 -- ADASYN can raise for data-dependent reasons
        # (e.g. every minority sample already "safe", no synthesis
        # possible) beyond the n_minority check above; never let that
        # crash the whole combo over one hospital.
        logger.warning(
            f"[ADASYN] hospital={hid}: ADASYN raised ({type(exc).__name__}: {exc}) -- falling back to plain "
            f"random oversample for this hospital only."
        )
        return hospital_wise_resample(X_h, y_h, np.full(len(y_h), hid), "oversample", ratio, seed)


def _enn_one_hospital(
    X_h: np.ndarray, y_h: np.ndarray, n_neighbors: int, hid, logger: logging.Logger,
) -> "tuple[np.ndarray, np.ndarray]":
    """Edited Nearest Neighbours for ONE hospital's own rows. Falls
    back to the hospital's ORIGINAL, unfiltered rows (never an empty
    class, never a crash) whenever ENN's own requirements aren't met,
    or whenever ENN would remove an entire class outright -- a cleaning
    step that deletes every majority (or minority) row is a bug
    signature, not a legitimate result, for this project's purposes.
    """
    classes, counts = np.unique(y_h, return_counts=True)
    k = min(n_neighbors, int(counts.min()) - 1)
    if k < 1:
        logger.info(
            f"[ENN] hospital={hid}: smallest class has only {int(counts.min())} train row(s) -- too few for "
            f"any neighbor-based cleaning; passing this hospital's rows through unfiltered."
        )
        return X_h, y_h
    try:
        sampler = EditedNearestNeighbours(n_neighbors=k, kind_sel="all")
        X_clean, y_clean = sampler.fit_resample(X_h, y_h)
        if len(np.unique(y_clean)) < len(classes):
            logger.warning(
                f"[ENN] hospital={hid}: cleaning would remove an entire class outright -- passing this "
                f"hospital's rows through unfiltered instead."
            )
            return X_h, y_h
        return X_clean, y_clean
    except Exception as exc:  # noqa: BLE001 -- see _adasyn_one_hospital's identical rationale
        logger.warning(
            f"[ENN] hospital={hid}: ENN raised ({type(exc).__name__}: {exc}) -- passing this hospital's "
            f"rows through unfiltered."
        )
        return X_h, y_h


def hospital_wise_resample(
    X: np.ndarray, y: np.ndarray, hospital_ids: np.ndarray, strategy: str, ratio: "float | None", seed: int = 42,
    enn_n_neighbors: int = 3, logger: "logging.Logger | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Random, hospital-wise, train-only resampling. `X`/`y`/
    `hospital_ids` must already be the same length and row order (the
    caller is responsible for this -- see `search_sampling_ratio()`,
    which masks `hospital_ids` the same way `_mask_known()` masks
    `X`/`y` before ever constructing this call).

    `strategy="undersample"`: randomly drops MAJORITY-class rows,
    within each hospital independently, until that hospital's
    minority:majority ratio reaches (at least) `ratio` -- never removes
    a minority-class row, never invents one.
    `strategy="oversample"`: randomly duplicates (WITH replacement)
    MINORITY-class rows, within each hospital independently, until that
    hospital's minority count reaches `ratio * majority_count` -- never
    removes a majority-class row.
    `strategy="adasyn"`: ADASYN's own synthetic minority generation,
    within each hospital independently, targeting `ratio` the same way
    as `"oversample"` (imblearn's `sampling_strategy` float target).
    `strategy="enn"`: Edited Nearest Neighbours cleaning, within each
    hospital independently. `ratio` is ignored (ENN has no ratio
    target -- it removes majority rows whose neighborhood disagrees
    with their label, not a fixed proportion).
    `strategy="adasyn_enn"`: `"adasyn"` then `"enn"`, in sequence,
    within each hospital independently (never across the two steps).

    A hospital with fewer than 2 distinct classes present in its train
    rows cannot be resampled at all (there is no minority/majority to
    rebalance) -- its rows are passed through UNCHANGED rather than
    raising, matching this project's established "small/degenerate
    hospital never crashes the run" discipline (Phase 8/9's empty- and
    single-node-graph handling, Phase 11's non-finite-loss skip, etc.).
    The advanced strategies extend this same discipline with their own
    additional, data-dependent fallbacks -- see `_adasyn_one_hospital()`
    /`_enn_one_hospital()`'s docstrings.
    Deterministic for a fixed `seed`: same inputs -> same output every
    time, no hidden global RNG state (ADASYN's own internal randomness
    is seeded identically via `random_state=seed`; ENN is a
    deterministic rule, no randomness to seed).
    """
    if strategy not in VALID_SAMPLING_STRATEGIES or strategy == "no_sampling":
        raise ValueError(f"strategy must be one of {RANDOM_SAMPLING_STRATEGIES + ADVANCED_SAMPLING_STRATEGIES}, got {strategy!r}.")
    if strategy != "enn" and not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0.0, 1.0], got {ratio!r}.")
    if len(X) != len(y) or len(X) != len(hospital_ids):
        raise ValueError(
            f"X ({len(X)}), y ({len(y)}), and hospital_ids ({len(hospital_ids)}) must be the same length -- "
            f"would otherwise silently misalign rows across hospitals."
        )
    logger = logger or logging.getLogger("eicu_fl")

    rng = np.random.RandomState(seed)
    hospital_ids = np.asarray(hospital_ids)
    X_parts: list = []
    y_parts: list = []

    for hid in np.unique(hospital_ids):
        h_mask = hospital_ids == hid
        X_h, y_h = X[h_mask], y[h_mask]
        classes, counts = np.unique(y_h, return_counts=True)

        if len(classes) < 2:
            X_parts.append(X_h)
            y_parts.append(y_h)
            continue

        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        minority_idx = np.where(y_h == minority_class)[0]
        majority_idx = np.where(y_h == majority_class)[0]
        n_minority, n_majority = len(minority_idx), len(majority_idx)

        if strategy == "adasyn":
            X_res, y_res = _adasyn_one_hospital(X_h, y_h, minority_class, ratio, seed, hid, logger=logger)
            X_parts.append(X_res)
            y_parts.append(y_res)
            continue
        if strategy == "enn":
            X_res, y_res = _enn_one_hospital(X_h, y_h, enn_n_neighbors, hid, logger=logger)
            X_parts.append(X_res)
            y_parts.append(y_res)
            continue
        if strategy == "adasyn_enn":
            X_mid, y_mid = _adasyn_one_hospital(X_h, y_h, minority_class, ratio, seed, hid, logger=logger)
            X_res, y_res = _enn_one_hospital(X_mid, y_mid, enn_n_neighbors, hid, logger=logger)
            X_parts.append(X_res)
            y_parts.append(y_res)
            continue

        if strategy == "undersample":
            # Keep every minority row; drop majority rows down to (at
            # most) n_minority / ratio, never below n_minority itself
            # and never MORE than the hospital already has (this is a
            # downsampling operation only).
            target_majority = max(n_minority, int(round(n_minority / ratio)))
            target_majority = min(target_majority, n_majority)
            selected_majority = rng.choice(majority_idx, size=target_majority, replace=False)
            selected_minority = minority_idx
        else:  # oversample
            # Keep every majority row; duplicate minority rows (with
            # replacement) up to ratio * n_majority, never below the
            # hospital's own original n_minority (this is an
            # upsampling operation only).
            target_minority = max(n_minority, int(round(ratio * n_majority)))
            selected_minority = rng.choice(minority_idx, size=target_minority, replace=True)
            selected_majority = majority_idx

        selected = np.concatenate([selected_minority, selected_majority])
        X_parts.append(X_h[selected])
        y_parts.append(y_h[selected])

    if not X_parts:
        return X[:0], y[:0]
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def _balanced_accuracy_or_none(metrics: dict) -> "float | None":
    if metrics.get("balanced_accuracy") is not None:
        return metrics["balanced_accuracy"]
    sens, spec = metrics.get("sensitivity"), metrics.get("specificity")
    if sens is not None and spec is not None:
        return (sens + spec) / 2.0
    return None


def metric_sort_key(metrics: dict) -> tuple:
    """Sort key over `_METRIC_PRIORITY`, ordered so a LARGER key is
    always a BETTER result -- `brier_score` (an error, lower-is-better)
    is negated so "larger = better" holds uniformly. A metric that's
    None (e.g. AUROC/AUPRC undefined with only one class present in a
    split, or specificity undefined with zero true negatives+false
    positives) sorts as WORSE than any real value, including negative
    ones -- never silently treated as 0.0, which could look better
    than a genuinely poor but present score.
    """
    key: list = []
    for name in _METRIC_PRIORITY:
        value = _balanced_accuracy_or_none(metrics) if name == "balanced_accuracy" else metrics.get(name)
        if value is None:
            key.append((-1.0, 0.0))
            continue
        signed = -float(value) if name in _LOWER_IS_BETTER else float(value)
        key.append((0.0, signed))
    return tuple(key)


def select_best_candidate(candidates: list, metrics_of=lambda c: c) -> object:
    """Returns the candidate maximizing `metric_sort_key(metrics_of(c))`
    -- ties broken by the metric priority order itself falling through
    to the next metric; if every metric ties too, `max()`'s stable
    left-to-right behavior keeps the FIRST candidate encountered (e.g.
    the smallest/cheapest ratio tried), never an arbitrary pick.
    Raises on an empty list rather than returning None, so a caller
    accidentally selecting over zero candidates fails loudly instead of
    propagating a silent None downstream.
    """
    if not candidates:
        raise ValueError("select_best_candidate() called with an empty candidate list.")
    return max(candidates, key=lambda c: metric_sort_key(metrics_of(c)))


def search_sampling_ratio(
    classifier_name: str, params: "dict | None",
    X_train: np.ndarray, y_train: np.ndarray, hospital_ids_train: np.ndarray,
    X_val: "np.ndarray | None", y_val: "np.ndarray | None",
    X_test: "np.ndarray | None", y_test: "np.ndarray | None",
    strategy: str, candidate_ratios: "list[float]", task: str,
    seed: int = 42, enn_n_neighbors: int = 3, logger: "logging.Logger | None" = None,
) -> "tuple[ClassifierAblationResult, list[dict]]":
    """Tries every ratio in `candidate_ratios` for `strategy`
    ("undersample" | "oversample" | "adasyn" | "enn" | "adasyn_enn"),
    refitting `classifier_name` on the hospital-wise-resampled TRAIN
    split each time (validation/test are passed straight through to
    `fit_and_evaluate_classifier()`, unresampled). Selection is by
    VALIDATION metrics only, via `select_best_candidate()`/
    `metric_sort_key()` (F1 -> recall -> ... -> Brier) -- test metrics
    are still computed and returned for every ratio (matching this
    project's existing practice of recording train/val/test for every
    experiment row), but never consulted for selection.

    `strategy="enn"` has no ratio to search (see `hospital_wise_
    resample()`'s docstring) -- whatever `candidate_ratios` the caller
    passes is IGNORED and replaced with a single `[None]` placeholder
    internally, so ENN only ever runs once per (combo, classifier,
    task) rather than wastefully repeating an identical, ratio-
    independent result N times. This keeps the calling convention
    uniform across all five strategies for `sampling_comparison_
    runner.py`'s own call sites, which don't need to special-case ENN.

    Returns `(best_result, all_trials)`: `best_result` is the winning
    ratio's `ClassifierAblationResult`; `all_trials` is
    `[{"ratio": r, "result": ClassifierAblationResult}, ...]` for every
    ratio tried, for full traceability in the written record.

    `hospital_ids_train` must be the SAME LENGTH and row order as the
    UNMASKED `X_train`/`y_train` (i.e. before `_mask_known()`'s
    NaN-label filtering) -- this function applies the identical
    `~np.isnan(y_train)` mask to `hospital_ids_train` before it's ever
    used, so hospital identity stays correctly aligned to the exact
    rows `fit_and_evaluate_classifier()` will actually fit on.
    """
    if strategy not in RANDOM_SAMPLING_STRATEGIES + ADVANCED_SAMPLING_STRATEGIES:
        raise ValueError(
            f"search_sampling_ratio() strategy must be one of "
            f"{RANDOM_SAMPLING_STRATEGIES + ADVANCED_SAMPLING_STRATEGIES}, got {strategy!r}."
        )
    if not candidate_ratios:
        raise ValueError("candidate_ratios must be non-empty.")
    logger = logger or logging.getLogger("eicu_fl")

    label_mask = ~np.isnan(y_train)
    hospital_ids_masked = np.asarray(hospital_ids_train)[label_mask]

    effective_ratios = [None] if strategy == "enn" else candidate_ratios
    if strategy == "enn" and len(candidate_ratios) > 1:
        logger.info(
            f"[SAMPLING_RATIO_SEARCH] {classifier_name}/{task}/enn: ratio has no effect on ENN -- running once, "
            f"ignoring the {len(candidate_ratios)} candidate ratio(s) passed in."
        )

    trials: list = []
    for ratio in effective_ratios:
        resampler = functools.partial(
            hospital_wise_resample, hospital_ids=hospital_ids_masked, strategy=strategy, ratio=ratio, seed=seed,
            enn_n_neighbors=enn_n_neighbors, logger=logger,
        )
        result = fit_and_evaluate_classifier(
            classifier_name, params, X_train, y_train, X_val, y_val, X_test, y_test,
            task=task, logger=logger, resampler=resampler,
        )
        trials.append({"ratio": ratio, "result": result})
        logger.info(
            f"[SAMPLING_RATIO_SEARCH] {classifier_name}/{task}/{strategy} ratio={ratio}: "
            f"val_f1={(result.val_metrics or {}).get('f1')} val_recall={(result.val_metrics or {}).get('recall')} "
            f"skipped={result.skipped}"
        )

    usable = [t for t in trials if not t["result"].skipped and t["result"].val_metrics]
    if not usable:
        logger.warning(
            f"[SAMPLING_RATIO_SEARCH] {classifier_name}/{task}/{strategy}: every candidate ratio was skipped or "
            f"produced no validation metrics -- returning the first trial's (skipped) result."
        )
        return trials[0]["result"], trials

    best = select_best_candidate(usable, metrics_of=lambda t: t["result"].val_metrics)
    logger.info(
        f"[SAMPLING_RATIO_SEARCH] {classifier_name}/{task}/{strategy}: selected ratio={best['ratio']} "
        f"(val_f1={best['result'].val_metrics.get('f1')}, val_recall={best['result'].val_metrics.get('recall')}) "
        f"out of {len(effective_ratios)} candidate(s)."
    )
    return best["result"], trials
