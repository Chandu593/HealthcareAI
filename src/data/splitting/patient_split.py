"""
Ablation framework -- permanent, patient-level train/val/test split
manifest.

Design decisions confirmed with the user (2026-09-10, ablation-framework
decisions #9 and #10 in IMPLEMENTATION_TRACKER.md):

  - Patient-level split WITHIN each hospital, not hospital-level --
    every hospital contributes patients to all three partitions, so the
    federated structure stays representative in every partition (a
    hospital-level split, e.g. "hospitals 1-150 train, 151-180 val,
    181-208 test", would be a materially different and much stranger
    experimental design for a per-hospital-client federated setup).
  - ONE permanent manifest, generated once and reused by every
    downstream experiment (the 72-config TS/TAB/fusion/classifier
    ablation, the graph-embedding ablation, and the split-aware retrain
    of Phase 11-13) -- never silently regenerated, so "the same patient
    lands in the same partition" holds across the entire ablation
    study. Re-running the generator with the same config/seed reproduces
    byte-identical splits (pure function of partition_result + labels +
    seed); it will refuse to silently overwrite an existing manifest
    unless explicitly told to (see `build_and_save_split_manifest`).
  - Every patient appears in EXACTLY ONE partition -- enforced as a
    hard invariant, checked and raised on, not just intended by
    construction.
  - Stratified by `stratify_on` (mortality | sepsis | none), falling
    back to an unstratified split for any hospital too small or too
    label-imbalanced to stratify (e.g. zero positives) -- logged
    clearly, never silently degraded or crashed on. This mirrors this
    project's "never silently default or guess" discipline (see
    multitask.sepsis_cohort's own validation in
    src/heads/multitask_heads.py).

Reuses `align_labels_to_patient_ids()` (Phase 10) for label lookup by
patient ID rather than re-deriving the mortality/sepsis join logic --
same "one ID-based alignment implementation, reused everywhere" pattern
already established for Phase 7/10/11's own label and embedding joins.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

VALID_STRATIFY = ("mortality", "sepsis", "none")
MIN_PATIENTS_FOR_THREE_WAY_SPLIT = 3   # below this, everyone goes to train -- see docstring
MIN_PER_CLASS_FOR_STRATIFY = 2         # sklearn's own floor for a class to survive a 2-way stratified split


@dataclass
class SplitManifest:
    seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    stratify_on: str
    sepsis_cohort: "str | None"
    # hospitalid (str, JSON-safe) -> {"train": [...], "val": [...], "test": [...]}
    hospital_splits: dict = field(default_factory=dict)
    # patientunitstayid (str, JSON-safe) -> "train" | "val" | "test" -- built once, O(1) lookup
    patient_to_split: dict = field(default_factory=dict)
    n_patients_total: int = 0
    n_hospitals_total: int = 0
    n_hospitals_stratified: int = 0
    n_hospitals_fallback_unstratified: int = 0
    n_hospitals_all_train_too_small: int = 0

    def split_of(self, patient_id: int) -> "str | None":
        return self.patient_to_split.get(str(int(patient_id)))

    def patient_ids_for_hospital(self, hospital_id, split: str) -> list[int]:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split!r}.")
        return [int(pid) for pid in self.hospital_splits.get(str(int(hospital_id)), {}).get(split, [])]

    def all_patient_ids(self, split: str) -> list[int]:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split!r}.")
        out: list[int] = []
        for hid_splits in self.hospital_splits.values():
            out.extend(int(pid) for pid in hid_splits.get(split, []))
        return out

    def to_dict(self) -> dict:
        return {
            "seed": self.seed, "train_frac": self.train_frac, "val_frac": self.val_frac,
            "test_frac": self.test_frac, "stratify_on": self.stratify_on, "sepsis_cohort": self.sepsis_cohort,
            "hospital_splits": self.hospital_splits, "patient_to_split": self.patient_to_split,
            "n_patients_total": self.n_patients_total, "n_hospitals_total": self.n_hospitals_total,
            "n_hospitals_stratified": self.n_hospitals_stratified,
            "n_hospitals_fallback_unstratified": self.n_hospitals_fallback_unstratified,
            "n_hospitals_all_train_too_small": self.n_hospitals_all_train_too_small,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SplitManifest":
        return cls(**d)


def _stratify_labels_for_hospital(
    patient_ids: list[int], cohort_patients: pd.DataFrame, mortality: pd.Series,
    sepsis_df: "pd.DataFrame | None", stratify_on: str, sepsis_cohort: "str | None",
) -> "np.ndarray | None":
    """Returns a 1D array of stratification keys (one per patient_id, in
    the same order), or None if stratify_on == 'none'. NaN (unknown)
    labels are folded into their own explicit 'unknown' bucket rather
    than dropped or coerced to a class -- a hospital where every
    patient's sepsis status happens to be unknown should fall back to
    unstratified (caught by the caller's class-count check below), not
    crash on an all-NaN stratify array.
    """
    if stratify_on == "none":
        return None

    from src.heads.multitask_heads import align_labels_to_patient_ids

    label_info = align_labels_to_patient_ids(
        patient_ids, cohort_patients, mortality, sepsis_df if sepsis_df is not None else pd.DataFrame(),
        sepsis_cohort or "broad",
    )
    raw = label_info["mortality"] if stratify_on == "mortality" else label_info["sepsis"]
    # Fold to string categories so NaN becomes its own real category
    # ("unknown"), not a float NaN that would break sklearn's stratify.
    keys = np.where(np.isnan(raw), "unknown", np.where(raw > 0.5, "positive", "negative"))
    return keys


def _split_one_hospital(
    patient_ids: list[int], stratify_keys: "np.ndarray | None", train_frac: float, val_frac: float,
    test_frac: float, seed: int, hospital_id, logger: "logging.Logger | None" = None,
) -> tuple[list[int], list[int], list[int], bool]:
    """Returns (train_ids, val_ids, test_ids, was_stratified). Falls
    back to an unstratified split (rather than raising) whenever
    stratification genuinely can't hold -- too few patients overall, or
    any stratify class with fewer than MIN_PER_CLASS_FOR_STRATIFY
    members. Logged, not silent.
    """
    n = len(patient_ids)
    logger = logger or logging.getLogger("eicu_fl")

    if n < MIN_PATIENTS_FOR_THREE_WAY_SPLIT:
        return list(patient_ids), [], [], False

    rng_seed = (int(seed) * 1_000_003 + int(hospital_id)) % (2**31 - 1)

    can_stratify = stratify_keys is not None
    if can_stratify:
        _, counts = np.unique(stratify_keys, return_counts=True)
        if counts.min() < MIN_PER_CLASS_FOR_STRATIFY:
            can_stratify = False
            logger.info(
                f"[SPLIT] hospital {hospital_id}: falling back to an UNSTRATIFIED split "
                f"({n} patients, smallest stratify class has {counts.min()} member(s), need >= "
                f"{MIN_PER_CLASS_FOR_STRATIFY})."
            )

    stratify_arg = stratify_keys if can_stratify else None
    ids = np.array(patient_ids)

    # Two-step split: carve off test first, then split the remainder into train/val.
    train_val_ids, test_ids = train_test_split(
        ids, test_size=test_frac, random_state=rng_seed, stratify=stratify_arg,
    )
    remaining_val_frac = val_frac / (train_frac + val_frac)

    # A second stratify check: the train/val remainder can have an even
    # smaller minority class than the full hospital did. Derive the
    # remainder's own stratify labels by position lookup (ids and
    # stratify_keys share the same order by construction) rather than
    # asking train_test_split to co-split a second array -- co-splitting
    # a second array breaks the moment stratify_arg is None (which is
    # exactly the fallback case this function exists to handle
    # gracefully), since sklearn then tries to index into `None`.
    can_stratify_2 = can_stratify
    train_val_strat = None
    if can_stratify_2:
        id_to_stratify = dict(zip(ids.tolist(), stratify_keys.tolist()))
        train_val_strat = np.array([id_to_stratify[i] for i in train_val_ids])
        _, counts2 = np.unique(train_val_strat, return_counts=True)
        if counts2.min() < MIN_PER_CLASS_FOR_STRATIFY:
            can_stratify_2 = False
            train_val_strat = None
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=remaining_val_frac, random_state=rng_seed + 1,
        stratify=train_val_strat if can_stratify_2 else None,
    )
    return sorted(int(x) for x in train_ids), sorted(int(x) for x in val_ids), sorted(int(x) for x in test_ids), can_stratify


def build_split_manifest(
    partition_result, cohort_patients: pd.DataFrame, mortality: pd.Series, sepsis_df: "pd.DataFrame | None",
    train_frac: float = 0.7, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42,
    stratify_on: str = "mortality", sepsis_cohort: "str | None" = None,
) -> SplitManifest:
    """Builds one permanent split manifest across every hospital in
    `partition_result` (Phase 2's own patient lists -- already respects
    debug/subset/full's cohort scale, no separate config needed here).

    Every patient in `partition_result` ends up in EXACTLY ONE of
    train/val/test for their own hospital -- verified as a hard
    invariant at the end (raises AssertionError, not a warning, if it
    ever doesn't hold; this manifest is the leakage boundary every
    downstream ablation experiment depends on).
    """
    logger = logging.getLogger("eicu_fl")
    if stratify_on not in VALID_STRATIFY:
        raise ValueError(f"stratify_on must be one of {VALID_STRATIFY}, got {stratify_on!r}.")
    total_frac = train_frac + val_frac + test_frac
    if not np.isclose(total_frac, 1.0, atol=1e-6):
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {total_frac}.")

    hospital_splits: dict = {}
    patient_to_split: dict = {}
    n_stratified = 0
    n_fallback = 0
    n_all_train = 0
    n_total = 0
    seen_patient_ids: set = set()

    for hid in sorted(partition_result.partitions.keys()):
        partition = partition_result.partitions[hid]
        patient_ids = [int(pid) for pid in partition.patient_ids]
        if not patient_ids:
            hospital_splits[str(hid)] = {"train": [], "val": [], "test": []}
            continue

        dup = seen_patient_ids.intersection(patient_ids)
        if dup:
            raise AssertionError(
                f"hospital {hid}: {len(dup)} patient(s) already assigned under another hospital "
                f"(e.g. {sorted(dup)[:5]}) -- Phase 2's partitions are expected to be disjoint; "
                f"investigate before trusting this split."
            )
        seen_patient_ids.update(patient_ids)

        stratify_keys = _stratify_labels_for_hospital(
            patient_ids, cohort_patients, mortality, sepsis_df, stratify_on, sepsis_cohort,
        )
        train_ids, val_ids, test_ids, was_stratified = _split_one_hospital(
            patient_ids, stratify_keys, train_frac, val_frac, test_frac, seed, hid, logger,
        )
        if len(patient_ids) < MIN_PATIENTS_FOR_THREE_WAY_SPLIT:
            n_all_train += 1
        elif was_stratified:
            n_stratified += 1
        else:
            n_fallback += 1

        hospital_splits[str(hid)] = {"train": train_ids, "val": val_ids, "test": test_ids}
        for pid in train_ids:
            patient_to_split[str(pid)] = "train"
        for pid in val_ids:
            patient_to_split[str(pid)] = "val"
        for pid in test_ids:
            patient_to_split[str(pid)] = "test"
        n_total += len(patient_ids)

    # Hard invariant: every patient assigned, none duplicated, none dropped.
    all_assigned = set()
    for hid_splits in hospital_splits.values():
        for split_name in ("train", "val", "test"):
            ids = hid_splits[split_name]
            overlap = all_assigned.intersection(ids)
            if overlap:
                raise AssertionError(f"Split-manifest leakage: patient(s) {sorted(overlap)[:5]} appear in more than one partition.")
            all_assigned.update(ids)
    if all_assigned != seen_patient_ids:
        missing = seen_patient_ids - all_assigned
        raise AssertionError(f"Split-manifest is missing {len(missing)} patient(s) from partition_result, e.g. {sorted(missing)[:5]}.")

    logger.info(
        f"[SPLIT] {n_total} patients across {len(hospital_splits)} hospitals: "
        f"{n_stratified} hospitals stratified on {stratify_on!r}, {n_fallback} fell back to "
        f"unstratified, {n_all_train} too small for a 3-way split (all patients -> train)."
    )

    return SplitManifest(
        seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, stratify_on=stratify_on,
        sepsis_cohort=sepsis_cohort, hospital_splits=hospital_splits, patient_to_split=patient_to_split,
        n_patients_total=n_total, n_hospitals_total=len(hospital_splits), n_hospitals_stratified=n_stratified,
        n_hospitals_fallback_unstratified=n_fallback, n_hospitals_all_train_too_small=n_all_train,
    )


def save_split_manifest(manifest: SplitManifest, path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_split_manifest(path: "str | Path") -> SplitManifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {path} -- generate it once with "
            f"build_and_save_split_manifest() before running any ablation experiment."
        )
    with open(path) as f:
        return SplitManifest.from_dict(json.load(f))


def build_and_save_split_manifest(
    partition_result, cohort_patients: pd.DataFrame, mortality: pd.Series, sepsis_df: "pd.DataFrame | None",
    path: "str | Path", train_frac: float = 0.7, val_frac: float = 0.15, test_frac: float = 0.15,
    seed: int = 42, stratify_on: str = "mortality", sepsis_cohort: "str | None" = None, overwrite: bool = False,
) -> SplitManifest:
    """Builds the manifest and persists it -- refuses to silently
    overwrite an existing manifest (per decision #9: ONE permanent
    manifest reused by every experiment) unless `overwrite=True` is
    passed explicitly. Returns the EXISTING manifest, unread and
    untouched, if one is already on disk and overwrite=False -- this is
    the normal, expected path for every experiment after the first.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        logging.getLogger("eicu_fl").info(f"[SPLIT] reusing existing permanent split manifest at {path}.")
        return load_split_manifest(path)

    manifest = build_split_manifest(
        partition_result, cohort_patients, mortality, sepsis_df, train_frac, val_frac, test_frac,
        seed, stratify_on, sepsis_cohort,
    )
    save_split_manifest(manifest, path)
    return manifest
