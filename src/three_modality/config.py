"""Defaults + validation for the three-modality suite. Anything under `three_modality:` in a YAML config
(or --set three_modality.x.y=v) is deep-merged over THREE_MODALITY_DEFAULTS."""

from __future__ import annotations

import copy

from src.paper.config import SEPSIS_DECONFOUND_REGEX, SEPSIS_INGREDIENT_GROUPS

# Bump when the label's SEMANTICS change (not for refactors): it is part of the run identity hash, so an old run is
# never silently reused for a changed definition.
INCIDENT_LABEL_DEFINITION_VERSION = "incident_sepsis_v1"
# Same rule for the TCD feature definitions: bump when what a feature MEANS changes, so an old feature cache is never silently reused.
TCD_FEATURE_DEFINITION_VERSION = "tcd_v1"
GRU_INPUT_DEFINITION_VERSION = "gru_inputs_v1"
GRU_ENCODER_DEFINITION_VERSION = "gru_encoder_v1"
TABULAR_INPUT_DEFINITION_VERSION = "tabular_inputs_v1"
TABULAR_AE_DEFINITION_VERSION = "tabular_ae_v1"

THREE_MODALITY_DEFAULTS: dict = {
    "incident_sepsis": {
        # --- windows (ICU hours since unit admission; whole hours only: evidence is binned to 1-hour bins) ---
        "cutoff_hours": 24,              # observation window = (-inf, cutoff]; incident-event window = (cutoff, horizon_end]
        "horizon_end_hours": 96,
        "prevalent_lookback_hours": 24,  # pre-ICU (negative-offset) evidence considered when testing for a PREVALENT case
        "co_occurrence_window_hours": 24,  # infection evidence and organ-dysfunction evidence must fall within this span of each other
        # --- suspected-infection evidence ---
        "infection": {
            "evidence": "antibiotic",       # "antibiotic" | "antibiotic_plus_culture" (culture within +/- co_occurrence_window of the antibiotic start; BOTH inside the window)
            "medication_tables": ["medication", "medication_imputed", "medication_imputed1"],
        },
        # --- organ dysfunction: time-resolved SOFA-like score from lab / infusionDrug / treatment only ---
        "organ_dysfunction": {
            "sofa_threshold": 2,
            "use_ventilation_support": True,   # respiratory bands 3-4 need a mechanical-ventilation record in the trailing window (else capped at 2)
            "ventilation_substrings": ["mechanical ventilation"],
            # EXACT (case-insensitive, stripped) lab names. Verify against `lab.labname.unique()` on the real data -- the evidence
            # coverage block of the cohort report shows how many patients each name actually matched.
            "lab_names": {
                "platelets": ["platelets x 1000", "platelets"],
                "bilirubin": ["total bilirubin"],
                "creatinine": ["creatinine"],
                "pao2": ["pao2"],
                "fio2": ["fio2"],
            },
        },
        # --- follow-up / censoring (patients leaving the ICU before the horizon ends) ---
        "followup": {
            # alive_discharge_as_negative : PRIMARY analysis. An event-free ALIVE ICU discharge before horizon_end is counted as a negative; ICU deaths
            #                               without an event stay excluded (competing risk). Defined at hour 24 (no knowledge of future stay length needed
            #                               beyond the competing-risk exclusion).
            # require_full_followup_all   : complete-follow-up SENSITIVITY analysis. Every patient (positive or negative) must remain in the ICU through
            #                               horizon_end (removes the short-stay effect, but conditions on a length of stay unknown at hour 24).
            # exclude_censored            : additional SENSITIVITY analysis. Event-free patients who left the ICU before horizon_end are EXCLUDED (alive ->
            #                               censored_alive_discharge, died -> censored_death_no_event); positives are kept whatever their follow-up length.
            #                               Selects differentially on length of stay (all negatives long-stay, a third of positives not) -- not a primary.
            "censoring_policy": "alive_discharge_as_negative",
        },
        # --- hospital-level label validity: the antibiotic half cannot fire where medication records are (almost) absent ---
        "label_validity": {"min_medication_coverage": 0.8},   # 0 disables; coverage = share of the hospital's landmark stays with >= 1 row in any medication table
        "chunksize": 2_000_000,
    },
    "tcd": {
        # ---------------- window: first 24 ICU hours only ----------------
        "window_minutes": 1440,           # MUST equal incident_sepsis.cutoff_hours * 60 (checked at run time): inputs end exactly where the event window begins
        "half_split_minutes": 720,        # early window [0, 12h) vs late window [12h, 24h]
        # ---------------- variables (time-stamped value series; offsets in [0, min(ICU exit, window)] only, no pre-ICU rows) ----------------
        "variables": {
            "top_k_by_coverage": 30,               # most-covered lab names among TRAINING patients (fitted on train only)
            "min_train_coverage": 0.02,            # ...and each needs >= 2% of training patients with a measurement in the window
            "priority_min_train_coverage": 0.005,  # clinically named roles are kept down to 0.5% ("where supported by actual data coverage")
            "chunksize": 2_000_000,
            # role -> accepted EXACT lab names (case-insensitive, stripped). Several names of one role are merged into one series.
            "roles": {
                "lactate": ["lactate"], "creatinine": ["creatinine"], "platelets": ["platelets x 1000"], "bilirubin": ["total bilirubin"],
                "wbc": ["wbc x 1000"], "bun": ["bun"], "sodium": ["sodium"], "potassium": ["potassium"], "chloride": ["chloride"],
                "bicarbonate": ["bicarbonate", "hco3"], "glucose": ["glucose", "bedside glucose"], "ph": ["ph"], "paco2": ["paco2"],
                "pao2": ["pao2"], "fio2": ["fio2"], "hgb": ["hgb"], "inr": ["pt - inr"], "ast": ["ast (sgot)"], "alt": ["alt (sgpt)"],
                "albumin": ["albumin"],
            },
            # [low, high] reference limits (null = open end). Project-defined adult limits, CONFIGURABLE, not diagnostic thresholds.
            # Only variables listed here get abnormality-burden / recovery / joint-abnormality features.
            "ranges": {
                "lactate": [None, 2.0], "creatinine": [None, 1.3], "platelets": [150, 450], "bilirubin": [None, 1.2], "wbc": [4, 12],
                "bun": [None, 25], "sodium": [135, 145], "potassium": [3.5, 5.0], "bicarbonate": [22, 29], "glucose": [70, 180],
                "ph": [7.35, 7.45], "paco2": [35, 45], "pao2": [80, None], "hgb": [10, None], "inr": [None, 1.5], "ast": [None, 80],
                "alt": [None, 80], "albumin": [3.0, None],
                "pf_ratio": [300, None], "bun_creat_ratio": [None, 20], "anion_gap_calc": [None, 12],
                "hr": [60, 100], "rr": [12, 20], "spo2": [92, None], "sbp": [90, None], "map": [65, None], "temp": [36.0, 38.3],
                "shock_index": [None, 0.9], "pulse_pressure": [25, 60],
            },
        },
        # ---------------- optional vitals source: used ONLY if the table exists (it does not in the 13-table eICU export) ----------------
        "vitals": {
            "table": "vitalPeriodic", "enabled": "auto",          # auto | off
            # role -> columns tried in order per row (invasive first, then non-invasive); hourly-median binned, t = bin midpoint
            "columns": {"hr": ["heartrate"], "rr": ["respiration"], "spo2": ["sao2"], "temp": ["temperature"],
                        "sbp": ["systemicsystolic", "noninvasivesystolic"], "dbp": ["systemicdiastolic", "noninvasivediastolic"],
                        "map": ["systemicmean", "noninvasivemean"]},
        },
        # ---------------- derived indicators from measurements taken at the IDENTICAL offset; unavailable inputs => indicator unavailable (never imputed) ----------------
        "derived": [
            {"name": "pf_ratio", "inputs": ["pao2", "fio2"], "op": "ratio_fio2"},                     # paO2 / FiO2 fraction
            {"name": "bun_creat_ratio", "inputs": ["bun", "creatinine"], "op": "ratio"},
            {"name": "anion_gap_calc", "inputs": ["sodium", "chloride", "bicarbonate"], "op": "anion_gap"},
            {"name": "shock_index", "inputs": ["hr", "sbp"], "op": "ratio"},                          # needs vitals
            {"name": "pulse_pressure", "inputs": ["sbp", "dbp"], "op": "difference"},               # needs vitals
            {"name": "map", "inputs": ["sbp", "dbp"], "op": "map_from_sbp_dbp", "only_if_missing": True},   # MAP "measured or appropriately derived"
        ],
        # ---------------- cross-variable interactions: both variables measured at the IDENTICAL offset, both in the stated abnormal direction ----------------
        "interactions": [
            {"name": "renal_joint", "a": "creatinine", "a_state": "high", "b": "bun", "b_state": "high"},
            {"name": "acid_base_joint", "a": "ph", "a_state": "low", "b": "lactate", "b_state": "high"},
            {"name": "resp_lab_joint", "a": "pao2", "a_state": "low", "b": "paco2", "b_state": "high"},
            {"name": "coag_joint", "a": "platelets", "a_state": "low", "b": "inr", "b_state": "high"},
            {"name": "shock_joint", "a": "hr", "a_state": "high", "b": "sbp", "b_state": "low"},         # needs vitals
            {"name": "resp_vitals_joint", "a": "rr", "a_state": "high", "b": "spo2", "b_state": "low"},   # needs vitals
            {"name": "hr_map_joint", "a": "hr", "a_state": "high", "b": "map", "b_state": "low"},        # needs vitals
        ],
        "joint_abnormal_min": 3,           # a lab draw with >= 3 abnormal ranged tests counts as a multi-abnormal draw
        # ---------------- gating: statistics that need enough points are NaN (not computed) below these ----------------
        "min_counts": {"std": 2, "mad": 3, "quantiles": 4, "slope": 3, "median_abs_change": 3},
        "min_span_hours": {"slope": 4.0, "rate": 2.0},
        "short_change_hours": [0.5, 6.0],  # max_abs_change_6h considers consecutive pairs 0.5 h <= dt <= 6 h
        "locf_max_gap_hours": 12.0,        # abnormal-TIME features carry a value forward at most this long; longer gaps are unobserved time
        # ---------------- preprocessing: FITTED ON TRAINING PATIENTS ONLY ----------------
        "preprocessing": {"max_missing_fraction": 0.95, "winsor_pct": [0.5, 99.5], "indicator_min_missing": 0.01},
        # ---------------- real-data evidence that nothing after the window enters the features ----------------
        "audit": {"n_patients": 2000, "horizon_minutes": 5760, "seed": 42},   # 0 disables
        # ---------------- ENCODER (existing TabularMLPEncoder trunk + two linear heads, focal multitask) ----------------
        "encoder": {"hidden_dim": 128, "embedding_dim": 64, "dropout": 0.1},
        # landmark_only: gradient updates AND early stopping use only patients still in the ICU at the landmark (valid ICU exit >= window_minutes -- the same strict
        # landmark as the paper track's primary mortality protocol). Short stays have clipped windows whose features reveal the short stay; embeddings are still
        # produced for EVERY patient. Set false to reproduce the all-stays encoder (run tcde_20261031T163020Z_e5fdd41e).
        "training": {"loss": "focal", "gamma": 2.0, "alpha": 0.25, "mortality_weight": 1.0, "sepsis_weight": 1.0,
                     "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 1024, "max_epochs": 30, "patience": 5, "seed": 42, "landmark_only": True},
    },
    "gru": {
        # ---------------- inputs: fresh 24 x 1-hour sequences from the SAME 50 lab series TCD selected (value + observation-mask channels) ----------------
        "bin_minutes": 60, "n_bins": 24,        # n_bins * bin_minutes MUST equal tcd.window_minutes (= incident cutoff): nothing after hour 24, ever
        "scaling": {"winsor_pct": [0.5, 99.5]},  # per-variable, fitted on TRAINING patients' observed values only (pooled; no per-hospital statistics)
        "audit": {"n_patients": 2000, "horizon_minutes": 5760, "seed": 42},
        # ---------------- encoder: existing GRUTimeSeriesEncoder (+ two linear heads for the supervised focal objective) ----------------
        "encoder": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.2},
        "training": {
            # separate focal parameters and task weights per task. NOTE alpha weights the POSITIVE class (the repo's masked_focal_loss): 0.25 down-weights it.
            "focal": {"mortality": {"alpha": 0.25, "gamma": 2.0}, "sepsis": {"alpha": 0.25, "gamma": 2.0}},
            "task_weights": {"mortality": 1.0, "sepsis": 1.0},
            "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 512, "max_epochs": 25, "patience": 4, "grad_clip": 1.0, "seed": 42,
            "landmark_only": True,              # same rule as the TCD encoder: update/early-stop only on patients still in the ICU at the landmark
        },
    },
    "tabular": {
        # ---------------- inputs: the paper track's CLEAN, leakage-guarded static feature set (build_feature_bundle), NOT the legacy Phase-5 features ----------------
        # clean set = the paper's `sepsis_clean` exclusions (label-ingredient groups: APACHE physiology and scores, SOFA labs, antibiotic/vasopressor/treatment/medication/infusion
        # tokens, infection diagnoses, cultures; hospital covariates; deconfounding name patterns; activetx) PLUS every `lab` group (already represented in TCD).
        "feature_set": {"exclude_groups": list(SEPSIS_INGREDIENT_GROUPS) + ["hospital", "lab"], "exclude_name_regex": list(SEPSIS_DECONFOUND_REGEX) + ["^apv_activetx$"]},
        "bundle_overrides": {},               # extra keys for paper.features (e.g. vocabulary sizes); window = tcd.window_minutes, all stays, APACHE scores off, no labs, no sequence block
        "preprocessing": {"winsor_pct": [0.5, 99.5], "max_missing_fraction": 0.95, "indicator_min_missing": 0.01},   # continuous columns only; binary columns are kept as 0/1
        # ---------------- encoder: existing TabularAutoencoder + two linear heads on the bottleneck ----------------
        "encoder": {"hidden_dim": 128, "embedding_dim": 64, "dropout": 0.1},
        "training": {
            "focal": {"mortality": {"alpha": 0.25, "gamma": 2.0}, "sepsis": {"alpha": 0.25, "gamma": 2.0}},     # SUPERVISED focal objective on the bottleneck embedding
            "task_weights": {"mortality": 1.0, "sepsis": 1.0},
            "reconstruction_weight": 1.0,                                                                       # UNSUPERVISED objective: MSE on continuous columns + BCE on binary columns
            "reconstruction": {"continuous_weight": 1.0, "binary_weight": 1.0},
            "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 1024, "max_epochs": 40, "patience": 5, "grad_clip": 1.0, "seed": 42, "landmark_only": True,
        },
    },
}

_CENSORING = ("exclude_censored", "alive_discharge_as_negative", "require_full_followup_all")
_INFECTION = ("antibiotic", "antibiotic_plus_culture")


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def _whole_hours(name: str, v) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"incident_sepsis.{name} must be a number of whole hours, got {v!r}")
    if f != int(f):
        raise ValueError(f"incident_sepsis.{name} must be a WHOLE number of hours (evidence is binned to 1-hour bins), got {v!r}")
    return int(f)


def validate_incident_sepsis_cfg(c: dict) -> dict:
    """Returns a normalised copy (hours as ints) or raises ValueError naming the offending key."""
    c = copy.deepcopy(c)
    c["cutoff_hours"] = _whole_hours("cutoff_hours", c["cutoff_hours"])
    c["horizon_end_hours"] = _whole_hours("horizon_end_hours", c["horizon_end_hours"])
    c["prevalent_lookback_hours"] = _whole_hours("prevalent_lookback_hours", c["prevalent_lookback_hours"])
    c["co_occurrence_window_hours"] = _whole_hours("co_occurrence_window_hours", c["co_occurrence_window_hours"])
    if c["cutoff_hours"] < 1:
        raise ValueError("incident_sepsis.cutoff_hours must be >= 1")
    if c["horizon_end_hours"] <= c["cutoff_hours"]:
        raise ValueError(f"incident_sepsis.horizon_end_hours ({c['horizon_end_hours']}) must be greater than cutoff_hours ({c['cutoff_hours']})")
    if c["prevalent_lookback_hours"] < 0:
        raise ValueError("incident_sepsis.prevalent_lookback_hours must be >= 0")
    if c["co_occurrence_window_hours"] < 1:
        raise ValueError("incident_sepsis.co_occurrence_window_hours must be >= 1")
    if c["infection"]["evidence"] not in _INFECTION:
        raise ValueError(f"incident_sepsis.infection.evidence must be one of {_INFECTION}, got {c['infection']['evidence']!r}")
    if not c["infection"]["medication_tables"]:
        raise ValueError("incident_sepsis.infection.medication_tables must list at least one table")
    if c["followup"]["censoring_policy"] not in _CENSORING:
        raise ValueError(f"incident_sepsis.followup.censoring_policy must be one of {_CENSORING}, got {c['followup']['censoring_policy']!r}")
    cov = float(c["label_validity"]["min_medication_coverage"])
    if not 0.0 <= cov <= 1.0:
        raise ValueError("incident_sepsis.label_validity.min_medication_coverage must be in [0, 1]")
    c["label_validity"]["min_medication_coverage"] = cov
    for dom in ("platelets", "bilirubin", "creatinine", "pao2", "fio2"):
        names = c["organ_dysfunction"]["lab_names"].get(dom)
        if not names:
            raise ValueError(f"incident_sepsis.organ_dysfunction.lab_names.{dom} must list at least one lab name")
    if int(c["organ_dysfunction"]["sofa_threshold"]) < 1:
        raise ValueError("incident_sepsis.organ_dysfunction.sofa_threshold must be >= 1")
    return c


def get_incident_sepsis_cfg(cfg) -> dict:
    """`cfg` is a RunConfig (uses .raw) or a plain dict. Returns the merged, validated incident_sepsis block."""
    raw = getattr(cfg, "raw", cfg) or {}
    over = ((raw.get("three_modality") or {}).get("incident_sepsis")) or {}
    return validate_incident_sepsis_cfg(_deep_merge(THREE_MODALITY_DEFAULTS["incident_sepsis"], over))


def validate_tcd_cfg(c: dict, incident_cutoff_hours: "int | None" = None) -> dict:
    c = copy.deepcopy(c)
    w = c["window_minutes"]
    if float(w) != int(w) or int(w) % 60 != 0 or int(w) < 60:
        raise ValueError(f"tcd.window_minutes must be a whole number of hours in minutes (multiple of 60), got {w!r}")
    c["window_minutes"] = int(w)
    if incident_cutoff_hours is not None and c["window_minutes"] != int(incident_cutoff_hours) * 60:
        raise ValueError(f"tcd.window_minutes ({c['window_minutes']}) must equal incident_sepsis.cutoff_hours*60 ({int(incident_cutoff_hours) * 60}): "
                         "the TCD inputs must end exactly where the incident-event window begins")
    if not 0 < c["half_split_minutes"] < c["window_minutes"]:
        raise ValueError("tcd.half_split_minutes must lie strictly inside the window")
    v = c["variables"]
    if int(v["top_k_by_coverage"]) < 0 or not 0 <= float(v["min_train_coverage"]) <= 1 or not 0 <= float(v["priority_min_train_coverage"]) <= 1:
        raise ValueError("tcd.variables: top_k_by_coverage must be >= 0 and the coverage thresholds in [0, 1]")
    for role, rng in v["ranges"].items():
        if len(rng) != 2:
            raise ValueError(f"tcd.variables.ranges.{role} must be [low, high]")
        if rng[0] is not None and rng[1] is not None and rng[0] >= rng[1]:
            raise ValueError(f"tcd.variables.ranges.{role}: low must be < high")
    if c["vitals"]["enabled"] not in ("auto", "off"):
        raise ValueError("tcd.vitals.enabled must be 'auto' or 'off'")
    pp = c["preprocessing"]
    if not 0 < float(pp["max_missing_fraction"]) <= 1 or not (0 <= pp["winsor_pct"][0] < pp["winsor_pct"][1] <= 100):
        raise ValueError("tcd.preprocessing: max_missing_fraction in (0, 1] and winsor_pct = [lo, hi] within [0, 100]")
    t = c["training"]
    if not isinstance(t.get("landmark_only"), bool):
        raise ValueError("tcd.training.landmark_only must be true or false")
    if t["loss"] != "focal":
        raise ValueError("tcd.training.loss must be 'focal' (the requested objective)")
    if t["gamma"] < 0 or not 0 < t["alpha"] < 1 or t["mortality_weight"] < 0 or t["sepsis_weight"] < 0 or (t["mortality_weight"] + t["sepsis_weight"]) <= 0:
        raise ValueError("tcd.training: gamma >= 0, 0 < alpha < 1, task weights >= 0 and not both zero")
    e = c["encoder"]
    if int(e["hidden_dim"]) < 1 or int(e["embedding_dim"]) < 1 or not 0 <= float(e["dropout"]) < 1:
        raise ValueError("tcd.encoder: hidden_dim/embedding_dim >= 1, 0 <= dropout < 1")
    return c


def get_tcd_cfg(cfg) -> dict:
    """Merged + validated tcd block; also cross-checks the window against the incident cutoff (explicit config values included)."""
    raw = getattr(cfg, "raw", cfg) or {}
    over = ((raw.get("three_modality") or {}).get("tcd")) or {}
    inc = get_incident_sepsis_cfg(cfg)
    return validate_tcd_cfg(_deep_merge(THREE_MODALITY_DEFAULTS["tcd"], over), inc["cutoff_hours"])


def validate_focal_training(t: dict, where: str) -> None:
    for task in ("mortality", "sepsis"):
        f = t["focal"][task]
        if not 0 < f["alpha"] < 1 or f["gamma"] < 0:
            raise ValueError(f"{where}.focal.{task}: need 0 < alpha < 1 and gamma >= 0")
        if t["task_weights"][task] < 0:
            raise ValueError(f"{where}.task_weights.{task} must be >= 0")
    if t["task_weights"]["mortality"] + t["task_weights"]["sepsis"] <= 0:
        raise ValueError(f"{where}.task_weights: both weights are zero")
    if not isinstance(t.get("landmark_only"), bool):
        raise ValueError(f"{where}.landmark_only must be true or false")
    for k in ("lr", "batch_size", "max_epochs", "patience"):
        if t[k] <= 0:
            raise ValueError(f"{where}.{k} must be > 0")


def validate_gru_cfg(c: dict, window_minutes: int) -> dict:
    c = copy.deepcopy(c)
    if int(c["bin_minutes"]) < 1 or int(c["n_bins"]) < 1:
        raise ValueError("gru.bin_minutes and gru.n_bins must be >= 1")
    if int(c["bin_minutes"]) * int(c["n_bins"]) != int(window_minutes):
        raise ValueError(f"gru.n_bins * gru.bin_minutes ({int(c['n_bins']) * int(c['bin_minutes'])}) must equal tcd.window_minutes ({int(window_minutes)}): "
                         "the sequences must cover exactly the first 24 ICU hours, no more and no less")
    lo, hi = c["scaling"]["winsor_pct"]
    if not 0 <= lo < hi <= 100:
        raise ValueError("gru.scaling.winsor_pct must be [lo, hi] within [0, 100]")
    e = c["encoder"]
    if int(e["hidden_dim"]) < 1 or int(e["num_layers"]) < 1 or not 0 <= float(e["dropout"]) < 1:
        raise ValueError("gru.encoder: hidden_dim/num_layers >= 1 and 0 <= dropout < 1")
    validate_focal_training(c["training"], "gru.training")
    return c


def get_gru_cfg(cfg) -> dict:
    raw = getattr(cfg, "raw", cfg) or {}
    over = ((raw.get("three_modality") or {}).get("gru")) or {}
    t = get_tcd_cfg(cfg)
    return validate_gru_cfg(_deep_merge(THREE_MODALITY_DEFAULTS["gru"], over), t["window_minutes"])


def validate_tabular_cfg(c: dict) -> dict:
    c = copy.deepcopy(c)
    pp = c["preprocessing"]
    if not 0 <= pp["winsor_pct"][0] < pp["winsor_pct"][1] <= 100 or not 0 < float(pp["max_missing_fraction"]) <= 1:
        raise ValueError("tabular.preprocessing: winsor_pct = [lo, hi] within [0, 100] and max_missing_fraction in (0, 1]")
    e = c["encoder"]
    if int(e["hidden_dim"]) < 1 or int(e["embedding_dim"]) < 1 or not 0 <= float(e["dropout"]) < 1:
        raise ValueError("tabular.encoder: hidden_dim/embedding_dim >= 1, 0 <= dropout < 1")
    t = c["training"]
    for task in ("mortality", "sepsis"):
        f = t["focal"][task]
        if not 0 < f["alpha"] < 1 or f["gamma"] < 0 or t["task_weights"][task] < 0:
            raise ValueError(f"tabular.training: focal.{task} needs 0 < alpha < 1, gamma >= 0 and task_weights.{task} >= 0")
    r = t["reconstruction"]
    if t["reconstruction_weight"] < 0 or r["continuous_weight"] < 0 or r["binary_weight"] < 0:
        raise ValueError("tabular.training: reconstruction weights must be >= 0")
    if t["reconstruction_weight"] * (r["continuous_weight"] + r["binary_weight"]) + t["task_weights"]["mortality"] + t["task_weights"]["sepsis"] <= 0:
        raise ValueError("tabular.training: every objective weight is zero")
    if not isinstance(t.get("landmark_only"), bool):
        raise ValueError("tabular.training.landmark_only must be true or false")
    for k in ("lr", "batch_size", "max_epochs", "patience"):
        if t[k] <= 0:
            raise ValueError(f"tabular.training.{k} must be > 0")
    return c


def get_tabular_cfg(cfg) -> dict:
    raw = getattr(cfg, "raw", cfg) or {}
    return validate_tabular_cfg(_deep_merge(THREE_MODALITY_DEFAULTS["tabular"], ((raw.get("three_modality") or {}).get("tabular")) or {}))
