"""
Pipeline entry point.

    python run_pipeline.py --config configs/debug.yaml --phase 5

Executes Phases 1-5 (data discovery/validation, hospital partitioning,
cohort/label construction, time-series preprocessing, tabular feature
engineering) -- see IMPLEMENTATION_TRACKER.md for status. This script
deliberately fails loudly (non-zero exit, explicit error) rather than
silently skipping a phase that couldn't run, per Section 43.

Phases 3-5 process hospital-by-hospital via hospital_sharding.py rather
than pooling all sampled patients' raw data into memory at once -- see
docs/memory_efficiency.md for why (a real OOM crash on a 199,108-patient
full-mode run, and the fix).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np

from src.data.cohort.cohort_report import build_cohort_report
from src.data.cohort.hospital_partition import HospitalPartition, PartitionResult, partition_by_hospital
from src.data.cohort.infection_proxy import InfectionProxyConfig, compute_infection_proxy
from src.data.cohort.mortality_label import mortality_label
from src.data.cohort.sepsis_label import compute_sepsis_labels
from src.data.cohort.sofa_like_scoring import SofaLikeConfig, compute_sofa_like_scores
from src.data.ingestion.data_report import generate_table_report, write_report
from src.data.ingestion.discovery import discover_tables
from src.data.validation.schema_validator import validate_all
from src.utils.config import ConfigError, load_config
from src.utils.logging_setup import PhaseTimer, setup_logging


def run_phase1_ingestion(cfg) -> dict:
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE1_DATA_DISCOVERY_AND_VALIDATION"):
        data_root = cfg.data_root
        discovery = discover_tables(data_root)
        for line in discovery.summary_lines():
            logger.info(f"[DATA] {line}")

        if not discovery.is_usable():
            raise RuntimeError(
                "Cannot proceed: 'patient' and/or 'hospital' table not found "
                f"under {data_root}. Found tables: {sorted(discovery.found.keys())}"
            )

        if discovery.missing_confirmed:
            raise RuntimeError(
                f"Required tables missing from {data_root}: {discovery.missing_confirmed}. "
                "See src/data/ingestion/schema_registry.py for the full expected table list."
            )

        validation_results = validate_all(discovery.found)
        failed = {name: r for name, r in validation_results.items() if not r.passed}
        for name, r in validation_results.items():
            if r.passed:
                logger.info(f"[VALIDATION] {name}: OK ({len(r.missing_optional_columns)} optional cols missing)")
            else:
                logger.error(f"[VALIDATION] {name}: FAILED - missing required columns {r.missing_required_columns}")
        if failed:
            raise RuntimeError(f"Schema validation failed for: {list(failed.keys())}")

        report_dir = str(Path(cfg.get("run.output_dir", "outputs")) / "data_reports")
        chunksize = cfg.get("data.chunksize", 200_000)
        table_reports = {}
        for name, path in discovery.found.items():
            report = generate_table_report(name, path, chunksize=chunksize)
            out_path = write_report(report, report_dir)
            table_reports[name] = out_path
            logger.info(
                f"[DATA_REPORT] {name}: rows={report.row_count:,} cols={report.column_count} "
                f"patients={report.unique_patient_count} hospitals={report.unique_hospital_count} "
                f"mem~{report.estimated_memory_mb}MB strategy={report.read_strategy}"
            )
            if report.leakage_risk_columns_present:
                logger.warning(
                    f"[LEAKAGE_RISK] {name} contains outcome-derived columns not safe as model "
                    f"input features: {report.leakage_risk_columns_present}"
                )

        summary = {
            "data_root": discovery.data_root,
            "found_tables": sorted(discovery.found.keys()),
            "table_paths": discovery.found,
            "missing_unconfirmed_tables": sorted(discovery.missing_unconfirmed),
            "unrecognized_files": discovery.unrecognized_files,
            "table_reports": table_reports,
        }
        summary_path = Path(cfg.get("run.output_dir", "outputs")) / "data_reports" / "_phase1_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[PHASE1] Summary written to {summary_path}")

    return summary


def load_phase1_summary(cfg) -> dict:
    """Reloads Phase 1's already-persisted summary for `--start-phase`
    resumption, instead of re-running the (real-data: ~113s+) full
    discovery/validation/chunked-report pass. Phase 1 already writes
    this exact file on every normal run (see run_phase1_ingestion
    above) -- no new persistence needed here, just a loader.

    Raises FileNotFoundError with a clear message if it doesn't exist
    (Phase 1 has never been run for this run.output_dir).
    """
    summary_path = Path(cfg.get("run.output_dir", "outputs")) / "data_reports" / "_phase1_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"--start-phase > 1 requires Phase 1 to have already run and saved {summary_path} -- "
            f"it hasn't. Run with --start-phase 1 (or omit it) at least once first."
        )
    with open(summary_path) as f:
        return json.load(f)


def _run_identity_fields(cfg) -> dict:
    """The exact config fields that determine WHICH patients/hospitals
    end up in the cohort -- used to guard --start-phase resumption
    against silently mixing outputs from two different runs that
    happen to share the same run.output_dir (e.g. debug.yaml's tiny
    cohort_patient_subset=20 subset vs full.yaml's full cohort -- both
    configs point at the same default outputs/ directory, since neither
    overrides run.output_dir). Deliberately does NOT include
    fusion/graph/gnn hyperparameters -- changing those between a
    --start-phase 8 run and the run that produced its inputs is the
    whole point of the feature, not a mismatch to guard against.
    """
    return {
        "data_root": cfg.data_root,
        "patient_subset": cfg.get("cohort.patient_subset"),
        "federated_mode": cfg.get("federated.mode", "subset"),
        "federated_hospitals": cfg.get("federated.hospitals", []) or [],
        "seed": cfg.get("run.seed", 42),
    }


def save_run_identity(cfg) -> None:
    """Records this run's cohort-identity fields (see
    _run_identity_fields()) to outputs/run_identity.json. Called
    whenever Phase 2 actually runs -- see run_phase2_hospital_partition().
    """
    out_dir = Path(cfg.get("run.output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_identity.json", "w") as f:
        json.dump(_run_identity_fields(cfg), f, indent=2)


def validate_run_identity(cfg) -> None:
    """Guards --start-phase resumption: raises a clear RuntimeError if
    the CURRENT config's cohort-identity fields don't match whatever
    produced the outputs/ artifacts being resumed from. Without this,
    e.g. running Phase 1-7 with debug.yaml's 20-patient-per-hospital
    subset, then `--start-phase 8 --phase 9` with full.yaml (both
    configs share the same default run.output_dir), would silently run
    the GNN over debug-scale graphs while the user believes they're
    working with the real full cohort -- same shape of artifact either
    way, no crash, just quietly wrong. Called whenever --start-phase > 2
    (i.e. whenever some phase 3+ output is being loaded rather than
    freshly computed from the CURRENT config's own Phase 2 run).
    """
    identity_path = Path(cfg.get("run.output_dir", "outputs")) / "run_identity.json"
    if not identity_path.exists():
        raise FileNotFoundError(
            f"--start-phase > 2 requires {identity_path} from a prior Phase 2 run -- it doesn't "
            f"exist. Run with --start-phase <= 2 at least once first."
        )
    with open(identity_path) as f:
        recorded = json.load(f)
    current = _run_identity_fields(cfg)
    mismatches = {k: (recorded[k], current[k]) for k in current if recorded.get(k) != current[k]}
    if mismatches:
        details = "\n".join(f"  {k}: recorded={old!r} vs current config={new!r}" for k, (old, new) in mismatches.items())
        raise RuntimeError(
            f"--start-phase resumption blocked: the current config doesn't match the run that "
            f"produced {cfg.get('run.output_dir', 'outputs')}'s existing outputs -- resuming would "
            f"silently mix cohorts from two different configs (e.g. debug.yaml's small subset vs "
            f"full.yaml's full cohort, which both default to the same output directory). "
            f"Mismatched field(s):\n{details}\n"
            f"Either use --start-phase <= 2 to rebuild the cohort fresh with this config, or point "
            f"run.output_dir at a separate directory per config/experiment."
        )


def run_phase2_hospital_partition(cfg, phase1_summary: dict) -> tuple:
    """Returns (partition_result, patient_df, hospital_df)."""
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))
    paths = phase1_summary["table_paths"]

    with PhaseTimer(logger, "PHASE2_HOSPITAL_PARTITIONING"):
        patient_df = pd.read_csv(paths["patient"])
        hospital_df = pd.read_csv(paths["hospital"])

        mode = cfg.get("federated.mode", "subset")
        requested = cfg.get("federated.hospitals", []) or None
        patient_subset = cfg.get("cohort.patient_subset")
        seed = cfg.get("run.seed", 42)

        result = partition_by_hospital(
            patient_df, hospital_df,
            mode=mode, requested_hospitals=requested,
            patient_subset=patient_subset, seed=seed,
        )

        logger.info(
            f"[PARTITION] mode={result.mode} eligible_hospitals={result.eligible_hospitals} "
            f"excluded_requested={result.excluded_hospitals} total_sampled_patients={result.total_patients()}"
        )
        for hid, p in result.partitions.items():
            flag = " [UNDERSAMPLED]" if p.was_undersampled else ""
            logger.info(
                f"[PARTITION] hospital={hid} patients={p.patient_count} "
                f"eligible={p.eligible_patient_count} sampled={p.sampled_patient_count} "
                f"mortality_pos={p.mortality_positive_count} mortality_neg={p.mortality_negative_count}{flag}"
            )
            if p.was_undersampled:
                logger.warning(
                    f"[PARTITION] hospital={hid} had fewer eligible patients "
                    f"({p.eligible_patient_count}) than requested patient_subset={patient_subset}"
                )

        out_dir = Path(cfg.get("run.output_dir", "outputs")) / "cohort_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        partition_json = {
            "mode": result.mode,
            "eligible_hospitals": result.eligible_hospitals,
            "excluded_hospitals": result.excluded_hospitals,
            "hospitals": {
                str(hid): {
                    "patient_count": p.patient_count,
                    "eligible_patient_count": p.eligible_patient_count,
                    "sampled_patient_count": p.sampled_patient_count,
                    "mortality_positive_count": p.mortality_positive_count,
                    "mortality_negative_count": p.mortality_negative_count,
                    "mortality_missing_count": p.mortality_missing_count,
                    "was_undersampled": p.was_undersampled,
                }
                for hid, p in result.partitions.items()
            },
        }
        with open(out_dir / "hospital_partition.json", "w") as f:
            json.dump(partition_json, f, indent=2, default=lambda o: int(o) if hasattr(o, "item") else str(o))
        logger.info(f"[PHASE2] Partition summary written to {out_dir / 'hospital_partition.json'}")

        # Full-fidelity version (includes patient_ids, unlike the
        # human-readable summary above) for --start-phase resumption --
        # a separate file so the existing summary's schema (and the
        # test that checks it) is untouched.
        full_partition_path = out_dir / "partition_result_full.json"
        with open(full_partition_path, "w") as f:
            json.dump(
                {
                    "mode": result.mode,
                    "requested_hospitals": result.requested_hospitals,
                    "eligible_hospitals": result.eligible_hospitals,
                    "excluded_hospitals": result.excluded_hospitals,
                    "partitions": {str(hid): asdict(p) for hid, p in result.partitions.items()},
                },
                f,
                default=lambda o: int(o) if hasattr(o, "item") else str(o),
            )
        logger.info(f"[PHASE2] Full partition result written to {full_partition_path}")

        save_run_identity(cfg)

    return result, patient_df, hospital_df


def load_partition_result(cfg):
    """Reloads Phase 2's full partition_result for `--start-phase`
    resumption. Raises FileNotFoundError with a clear message if it
    doesn't exist (Phase 2 has never been run for this run.output_dir).
    """
    full_partition_path = Path(cfg.get("run.output_dir", "outputs")) / "cohort_reports" / "partition_result_full.json"
    if not full_partition_path.exists():
        raise FileNotFoundError(
            f"--start-phase > 2 requires Phase 2 to have already run and saved {full_partition_path} -- "
            f"it hasn't. Run with --start-phase <= 2 at least once first."
        )
    with open(full_partition_path) as f:
        data = json.load(f)
    partitions = {
        int(hid): HospitalPartition(**p) for hid, p in data["partitions"].items()
    }
    return PartitionResult(
        mode=data["mode"],
        requested_hospitals=data["requested_hospitals"],
        eligible_hospitals=data["eligible_hospitals"],
        excluded_hospitals=data["excluded_hospitals"],
        partitions=partitions,
    )


def run_phase3_cohort_construction(cfg, phase1_summary: dict, partition_result, patient_df: pd.DataFrame) -> dict:
    """Phase 3 -- mortality + dual-cohort sepsis label construction.

    Processes hospital-by-hospital (see hospital_sharding.py) rather
    than pooling all sampled patients' raw table data into memory at
    once. This is what fixes the real OOM crash a 199,108-patient
    full-mode run hit inside combine_medication_sources()'s
    drop_duplicates() on a pooled ~16.3M-row concatenation of all three
    medication tables. Peak memory is now bounded by the largest single
    hospital's data, not the whole dataset -- see
    docs/memory_efficiency.md.
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))
    paths = phase1_summary["table_paths"]

    with PhaseTimer(logger, "PHASE3_COHORT_AND_LABEL_CONSTRUCTION"):
        sampled_patient_ids: set[int] = set()
        for p in partition_result.partitions.values():
            sampled_patient_ids.update(p.patient_ids)

        cohort_patients = patient_df[patient_df["patientunitstayid"].isin(sampled_patient_ids)].copy()
        mortality = mortality_label(cohort_patients)
        logger.info(
            f"[LABEL] mortality: positive={int((mortality == 1).sum())} "
            f"negative={int((mortality == 0).sum())} unknown={int(mortality.isna().sum())}"
        )

        from src.data.ingestion.loading import load_filtered
        from src.data.ingestion.hospital_sharding import load_hospital_shard, shard_table_by_hospital
        from src.data.cohort.infection_proxy import combine_medication_sources

        chunksize = cfg.get("data.chunksize", 200_000)
        cache_dir = cfg.get("run.cache_dir", "cache")

        # apachePredVar is small (171,177 rows / ~75MB dataset-wide) --
        # cheap enough to load once for the whole cohort, no sharding
        # needed. Still column-projected to trim memory further.
        APV_COLUMNS = [
            "patientunitstayid", "pao2", "fio2", "day1pao2", "day1fio2",
            "verbal", "motor", "eyes", "day1verbal", "day1motor", "day1eyes",
            "creatinine", "ventday1",
        ]
        apache_pred_var = load_filtered(paths["apachePredVar"], sampled_patient_ids, chunksize, usecols=APV_COLUMNS)

        patient_hospital_map = dict(zip(cohort_patients["patientunitstayid"], cohort_patients["hospitalid"]))

        # Single chunked pass per large table, routed to per-hospital
        # shards on disk (cached across phases/re-runs).
        shard_manifests = {}
        for table_name in ("lab", "medication", "medication_imputed", "medication_imputed1", "infusionDrug", "microLab"):
            if table_name in paths:
                shard_manifests[table_name] = shard_table_by_hospital(
                    paths[table_name], table_name, patient_hospital_map, cache_dir, chunksize,
                )

        sofa_cfg = SofaLikeConfig(
            threshold=cfg.get("cohort.sepsis.organ_dysfunction.sofa_like_threshold", 2),
            observation_window_minutes=cfg.get("cohort.sepsis.organ_dysfunction.observation_window_minutes", 1440),
        )
        infection_cfg = InfectionProxyConfig(
            corroboration_window_minutes=cfg.get("cohort.sepsis.infection.corroboration_window_minutes", 1440),
            lookback_minutes=cfg.get("cohort.sepsis.infection.lookback_minutes", 1440),
            lookahead_minutes=cfg.get("cohort.sepsis.infection.lookahead_minutes", 2880),
        )

        sofa_frames = []
        infection_frames = []
        total_medication_rows_after_dedup = 0
        n_hospitals_total = sum(1 for p in partition_result.partitions.values() if p.patient_ids)
        n_hospitals_done = 0
        phase3_loop_start = time.time()

        for hospital_id, partition in partition_result.partitions.items():
            if not partition.patient_ids:
                continue  # e.g. a hospital with 0 eligible patients after mortality-label filtering

            hosp_apv = apache_pred_var[apache_pred_var["patientunitstayid"].isin(partition.patient_ids)]
            hosp_lab = (
                load_hospital_shard(shard_manifests["lab"], hospital_id)
                if "lab" in shard_manifests else pd.DataFrame(columns=["patientunitstayid", "labresultoffset", "labname", "labresult"])
            )
            hosp_infusion = (
                load_hospital_shard(shard_manifests["infusionDrug"], hospital_id)
                if "infusionDrug" in shard_manifests else pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname", "infusionrate"])
            )
            hosp_microlab = (
                load_hospital_shard(shard_manifests["microLab"], hospital_id)
                if "microLab" in shard_manifests else pd.DataFrame(columns=["patientunitstayid", "culturetakenoffset"])
            )

            hosp_med_sources = {}
            for table_name in ("medication", "medication_imputed", "medication_imputed1"):
                if table_name in shard_manifests:
                    hosp_med_sources[table_name] = load_hospital_shard(shard_manifests[table_name], hospital_id)
            hosp_medication = combine_medication_sources(hosp_med_sources)
            total_medication_rows_after_dedup += len(hosp_medication)

            hosp_sofa_df = compute_sofa_like_scores(hosp_apv, hosp_lab, hosp_infusion, sofa_cfg)
            hosp_infection_df = compute_infection_proxy(hosp_medication, hosp_microlab, infection_cfg)

            sofa_frames.append(hosp_sofa_df)
            infection_frames.append(hosp_infection_df)

            # Progress logging: without this, a large full-mode run goes
            # completely silent for many minutes (a real run showed a
            # ~9-minute gap between the last [SHARD] line and the next
            # log output), which is indistinguishable from a hang. Log
            # every 20 hospitals or every 60s, whichever comes first.
            n_hospitals_done += 1
            elapsed = time.time() - phase3_loop_start
            if n_hospitals_done % 20 == 0 or n_hospitals_done == n_hospitals_total:
                rate = n_hospitals_done / elapsed if elapsed > 0 else 0
                eta = (n_hospitals_total - n_hospitals_done) / rate if rate > 0 else float("nan")
                logger.info(
                    f"[PHASE3] progress: {n_hospitals_done}/{n_hospitals_total} hospitals "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

        sofa_df = pd.concat(sofa_frames, ignore_index=True) if sofa_frames else pd.DataFrame(
            columns=["patientunitstayid", "sofa_like_total", "n_domains_observed", "max_possible_score", "organ_dysfunction"]
        )
        infection_df = pd.concat(infection_frames, ignore_index=True) if infection_frames else pd.DataFrame(
            columns=["patientunitstayid", "broad_infection", "broad_infection_offset",
                     "strict_infection", "strict_infection_offset", "n_antibiotic_orders", "n_cultures_ordered"]
        )
        logger.info(
            f"[MEDICATION] combined across {len(partition_result.partitions)} hospitals "
            f"-> {total_medication_rows_after_dedup:,} rows after per-hospital dedup"
        )
        logger.info(
            f"[SOFA] scored {len(sofa_df)} patients, "
            f"organ_dysfunction_positive={int(sofa_df['organ_dysfunction'].sum())}"
        )
        logger.info(
            f"[INFECTION] broad_positive={int(infection_df['broad_infection'].sum())} "
            f"strict_positive={int(infection_df['strict_infection'].sum())}"
        )

        sepsis_df = compute_sepsis_labels(infection_df, sofa_df)
        logger.info(
            f"[SEPSIS] broad_positive={int(sepsis_df['sepsis_broad'].sum())} "
            f"strict_positive={int(sepsis_df['sepsis_strict'].sum())}"
        )

        report = build_cohort_report(cohort_patients, mortality, sepsis_df)

        out_dir = Path(cfg.get("run.output_dir", "outputs")) / "cohort_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "cohort_report.json", "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info(f"[PHASE3] Cohort report written to {out_dir / 'cohort_report.json'}")
        logger.info(
            f"[PHASE3] total={report.total_patients} eligible={report.eligible_patients} "
            f"mortality_pos={report.mortality_positive} "
            f"sepsis_strict_pos={report.sepsis_strict_positive} ({report.sepsis_strict_prevalence:.1%}) "
            f"sepsis_broad_pos={report.sepsis_broad_positive} ({report.sepsis_broad_prevalence:.1%}) "
            f"overlap={report.cohort_overlap}"
        )

    persist_dir = Path(cfg.get("run.output_dir", "outputs")) / "cohort_reports"
    persist_dir.mkdir(parents=True, exist_ok=True)
    cohort_patients.to_csv(persist_dir / "cohort_patients.csv", index=False)
    # mortality is a pd.Series sharing cohort_patients' DataFrame index
    # (not patientunitstayid) -- pairing it explicitly with
    # patientunitstayid before saving avoids relying on a fragile
    # positional-index round-trip through CSV, same ID-based-alignment
    # discipline as every embeddings artifact since Phase 7.
    pd.DataFrame({
        "patientunitstayid": cohort_patients["patientunitstayid"].values, "mortality": mortality.values,
    }).to_csv(persist_dir / "mortality_labels.csv", index=False)
    sepsis_df.to_csv(persist_dir / "sepsis_labels.csv", index=False)
    logger.info(f"[PHASE3] cohort_patients/mortality/sepsis_df persisted under {persist_dir} for --start-phase resumption")

    return {
        "cohort_report_path": str(out_dir / "cohort_report.json"),
        "mortality": mortality,
        "sepsis_df": sepsis_df,
        "sofa_df": sofa_df,
        "infection_df": infection_df,
        "cohort_patients": cohort_patients,
        "patient_hospital_map": patient_hospital_map,
        "shard_manifests": shard_manifests,
    }


# Exact table list Phase 3 shards -- kept as one named constant so
# load_phase3_result() (below) can re-derive shard_manifests via the
# SAME list, rather than a separately-maintained copy that could drift.
PHASE3_SHARDED_TABLES = ("lab", "medication", "medication_imputed", "medication_imputed1", "infusionDrug", "microLab")


def load_phase3_result(cfg, phase1_summary: dict, need_shard_manifests: bool = True) -> dict:
    """Reloads Phase 3's result for `--start-phase` resumption.

    `cohort_patients`/`mortality`/`sepsis_df` come from Phase 3's
    persisted CSVs. `shard_manifests` is re-derived by calling
    shard_table_by_hospital() again for PHASE3_SHARDED_TABLES -- this
    is NOT a full re-shard: hospital_sharding.py's existing cache
    mechanism (mtime+size+columns+cohort hash) makes this a fast
    manifest-file read as long as cache/ wasn't cleared and the source
    tables haven't changed, the same caching every normal Phase 3/4 run
    already relies on. `sofa_df`/`infection_df` are intentionally
    `None` here -- confirmed nothing downstream of Phase 3 reads them
    (they're intermediate labeling artifacts, not consumed by Phase
    4/5/7/8/9), so persisting them would be unused cost.

    `need_shard_manifests=False` skips shard_manifests reconstruction
    entirely -- confirmed (by inspection) that ONLY Phase 4 ever reads
    phase3_result['shard_manifests']; Phase 5 and Phase 7 never do.
    This matters beyond just saving time: shard_table_by_hospital()'s
    cache-freshness check needs to stat the ORIGINAL source file (to
    detect if it changed since caching), so it requires that file to
    still exist on disk even on a cache hit -- skipping this
    reconstruction when it's not actually needed is what lets
    `--start-phase 7` (or 8, 9) work with the raw eICU data no longer
    present at all, not just quickly.

    Dtype safety: sepsis_df's bool columns get explicitly re-cast after
    the CSV round-trip -- CSV does not reliably preserve bool dtype
    (this project has a documented real bug from exactly this class of
    issue; see sepsis_label.py's own comment on `.astype(bool)` not
    being cosmetic). mortality is reconstructed as a float Series
    (matching its original {1.0, 0.0, NaN} dtype from mortality_label()),
    explicitly re-joined to cohort_patients' row order by
    patientunitstayid, not by raw CSV row position.
    """
    persist_dir = Path(cfg.get("run.output_dir", "outputs")) / "cohort_reports"
    cohort_patients_path = persist_dir / "cohort_patients.csv"
    mortality_path = persist_dir / "mortality_labels.csv"
    sepsis_path = persist_dir / "sepsis_labels.csv"
    missing = [p for p in (cohort_patients_path, mortality_path, sepsis_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"--start-phase > 3 requires Phase 3 to have already run and saved {missing} -- "
            f"they don't all exist. Run with --start-phase <= 3 at least once first."
        )

    cohort_patients = pd.read_csv(cohort_patients_path)

    mortality_df = pd.read_csv(mortality_path)
    mortality_by_id = dict(zip(mortality_df["patientunitstayid"], mortality_df["mortality"]))
    mortality = cohort_patients["patientunitstayid"].map(mortality_by_id).astype(float)

    sepsis_df = pd.read_csv(sepsis_path)
    for bool_col in ("organ_dysfunction", "broad_infection", "strict_infection", "sepsis_broad", "sepsis_strict"):
        if bool_col in sepsis_df.columns:
            sepsis_df[bool_col] = sepsis_df[bool_col].astype(bool)

    patient_hospital_map = dict(zip(cohort_patients["patientunitstayid"], cohort_patients["hospitalid"]))

    shard_manifests = {}
    if need_shard_manifests:
        from src.data.ingestion.hospital_sharding import shard_table_by_hospital

        paths = phase1_summary["table_paths"]
        chunksize = cfg.get("data.chunksize", 200_000)
        cache_dir = cfg.get("run.cache_dir", "cache")
        for table_name in PHASE3_SHARDED_TABLES:
            if table_name in paths:
                shard_manifests[table_name] = shard_table_by_hospital(
                    paths[table_name], table_name, patient_hospital_map, cache_dir, chunksize,
                )

    return {
        "cohort_report_path": str(persist_dir / "cohort_report.json"),
        "mortality": mortality,
        "sepsis_df": sepsis_df,
        "sofa_df": None,
        "infection_df": None,
        "cohort_patients": cohort_patients,
        "patient_hospital_map": patient_hospital_map,
        "shard_manifests": shard_manifests,
    }


def run_phase4_timeseries_modality(cfg, phase3_result: dict, partition_result, train_patient_ids: "set | None" = None) -> dict:
    """Phase 4 -- time-series modality (Section 10).

    Source tables are lab + medication + infusionDrug, NOT vital signs
    (no vitalPeriodic available -- see schema_registry.py). Produces a
    fixed (patients, max_bins, features) tensor via time binning +
    normalization, always. Runs the LSTM forward pass ONLY if torch is
    importable in this environment -- otherwise logs that step as
    explicitly skipped rather than silently omitted, per Section 43.

    `train_patient_ids` (added 2026-09-11, part of making Phase 11/12/13
    split-aware -- see IMPLEMENTATION_TRACKER.md): OPTIONAL, defaults to
    None, which reproduces this function's ORIGINAL behavior EXACTLY
    (full-cohort-fit normalization, unchanged) -- verified by the full
    pre-existing test suite passing unmodified. When given a set of
    patient IDs, normalization statistics are fit using ONLY those
    patients (per hospital), then applied to every patient -- see
    src/data/temporal/normalization.py's `fit_transform_local_split_aware()`.
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE4_TIMESERIES_MODALITY"):
        from src.data.temporal.time_binning import (
            bin_infusion_features, bin_lab_features, bin_medication_features, select_lab_features,
        )
        from src.data.temporal.sequence_builder import (
            attach_sequence_length, build_sequence_tensor, compute_sequence_length,
        )
        from src.data.temporal.normalization import fit_transform_local
        from src.data.ingestion.hospital_sharding import load_hospital_shard
        from src.data.cohort.infection_proxy import combine_medication_sources

        bin_minutes = cfg.get("temporal.time_bin_minutes", 240)
        max_bins = cfg.get("temporal.max_sequence_length", 48)
        tracked_labnames = cfg.get(
            "modalities.timeseries.lab_features",
            ["platelet", "bilirubin", "creatinine", "fibrinogen", "PT - INR", "pH"],
        )

        cohort_patients = phase3_result["cohort_patients"]
        patient_ids = cohort_patients["patientunitstayid"].tolist()
        hospital_id_by_patient = dict(zip(cohort_patients["patientunitstayid"], cohort_patients["hospitalid"]))

        # Reuse Phase 3's shards (cache hit -- see hospital_sharding.py).
        # Phase 4 is only ever invoked after Phase 3 in this pipeline
        # (run_pipeline.py always runs phases 1..N sequentially), so
        # shard_manifests is always present here -- fail loudly rather
        # than silently deriving from an unset fallback if that
        # assumption is ever broken by a future caller.
        shard_manifests = phase3_result.get("shard_manifests")
        if not shard_manifests:
            raise RuntimeError(
                "Phase 4 requires phase3_result['shard_manifests'] from a preceding "
                "Phase 3 run in the same process -- run_phase3_cohort_construction() "
                "must be called before run_phase4_timeseries_modality()."
            )

        # select_lab_features needs a representative SAMPLE of real lab
        # data to check which configured labname substrings actually
        # match (or fall back to top-frequency labnames) -- sampling a
        # bounded number of hospitals' already-small shards keeps this
        # cheap even in full mode, rather than needing the full pooled
        # `lab` table (lab is tens of millions of rows -- loading it in
        # full just to check for substring matches would reintroduce
        # the OOM risk hospital-sharding was built to avoid).
        #
        # FIXED 2026-08-2x (vocabulary-sampling-bias bug -- see
        # docs/tabular_modality.md): this used to be
        # `list(partition_result.partitions.keys())[:10]`, which always
        # picks the 10 SMALLEST hospital IDs (partitions is built by
        # iterating `sorted(available_hospitals)` in
        # hospital_partition.py), not a representative sample. Fixed to
        # a seeded random sample of the same size -- still bounded and
        # cheap, but no longer systematically biased toward whichever
        # hospitals happen to have the smallest IDs. Seeded with
        # run.seed for reproducibility, same convention as
        # partition_by_hospital's own patient sampling.
        all_hospital_ids = list(partition_result.partitions.keys())
        sample_size = min(10, len(all_hospital_ids))
        sample_hospital_ids = random.Random(cfg.get("run.seed", 42)).sample(all_hospital_ids, sample_size)
        lab_sample_frames = [
            load_hospital_shard(shard_manifests["lab"], hid) for hid in sample_hospital_ids
            if "lab" in shard_manifests
        ]
        lab_sample = pd.concat(lab_sample_frames, ignore_index=True) if lab_sample_frames else pd.DataFrame(columns=["labname"])
        lab_feature_names = select_lab_features(lab_sample, tracked_labnames)
        if set(lab_feature_names) != set(tracked_labnames):
            logger.warning(
                f"[TIMESERIES] configured lab_features {tracked_labnames} had no match in a "
                f"{len(sample_hospital_ids)}-hospital sample of this cohort's lab data -- "
                f"fell back to top-frequency labnames: {lab_feature_names}"
            )

        feature_names = (
            [f"lab_{f.replace(' ', '_')}" for f in lab_feature_names]
            + ["medication_n_active", "medication_antibiotic_active"]
            + ["infusion_n_active", "infusion_vasopressor_active", "infusion_vasopressor_rate_mean"]
        )

        # Bin per hospital and accumulate the (small, already-aggregated)
        # long-format binned frames -- this is what keeps Phase 4 memory
        # bounded the same way Phase 3 now is, without needing to change
        # time_binning.py itself: binning output is at most
        # patients x max_bins x features rows, not patients x raw_events.
        lab_binned_frames, med_binned_frames, inf_binned_frames = [], [], []
        n_hospitals_total = sum(1 for p in partition_result.partitions.values() if p.patient_ids)
        n_hospitals_done = 0
        binning_loop_start = time.time()

        for hospital_id, partition in partition_result.partitions.items():
            if not partition.patient_ids:
                continue
            hosp_lab = load_hospital_shard(shard_manifests["lab"], hospital_id) if "lab" in shard_manifests else pd.DataFrame(columns=["patientunitstayid", "labresultoffset", "labname", "labresult"])
            hosp_infusion = load_hospital_shard(shard_manifests["infusionDrug"], hospital_id) if "infusionDrug" in shard_manifests else pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname", "infusionrate"])

            hosp_med_sources = {}
            for table_name in ("medication", "medication_imputed", "medication_imputed1"):
                if table_name in shard_manifests:
                    hosp_med_sources[table_name] = load_hospital_shard(shard_manifests[table_name], hospital_id)
            hosp_medication = combine_medication_sources(hosp_med_sources)

            lab_binned_frames.append(bin_lab_features(hosp_lab, lab_feature_names, bin_minutes, max_bins))
            med_binned_frames.append(bin_medication_features(hosp_medication, bin_minutes, max_bins))
            inf_binned_frames.append(bin_infusion_features(hosp_infusion, bin_minutes, max_bins))

            # See the matching note in Phase 3 -- without this, a large
            # full-mode run goes silent for many minutes here too (a
            # real run showed ~16 minutes of no log output across this loop).
            n_hospitals_done += 1
            elapsed = time.time() - binning_loop_start
            if n_hospitals_done % 20 == 0 or n_hospitals_done == n_hospitals_total:
                rate = n_hospitals_done / elapsed if elapsed > 0 else 0
                eta = (n_hospitals_total - n_hospitals_done) / rate if rate > 0 else float("nan")
                logger.info(
                    f"[PHASE4] binning progress: {n_hospitals_done}/{n_hospitals_total} hospitals "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

        lab_binned = pd.concat(lab_binned_frames, ignore_index=True) if lab_binned_frames else pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])
        med_binned = pd.concat(med_binned_frames, ignore_index=True) if med_binned_frames else pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])
        inf_binned = pd.concat(inf_binned_frames, ignore_index=True) if inf_binned_frames else pd.DataFrame(columns=["patientunitstayid", "bin_idx", "feature", "value"])

        tensor = build_sequence_tensor([lab_binned, med_binned, inf_binned], patient_ids, feature_names, max_bins)
        seq_lengths = compute_sequence_length(cohort_patients, bin_minutes, max_bins)
        tensor = attach_sequence_length(tensor, seq_lengths)

        coverage = tensor.observed_mask.mean()
        logger.info(
            f"[TIMESERIES] tensor shape={tensor.array.shape} features={feature_names} "
            f"observed_coverage={coverage:.1%} mean_sequence_length={tensor.sequence_length.mean():.1f} bins"
        )

        if train_patient_ids is not None:
            from src.data.temporal.normalization import fit_transform_local_split_aware
            norm_result = fit_transform_local_split_aware(tensor, hospital_id_by_patient, train_patient_ids)
            logger.info(
                f"[TIMESERIES] SPLIT-AWARE normalization: {len(train_patient_ids)} train patients used for "
                f"fitting; scaler transform applied to all {len(patient_ids)} patients."
            )
        else:
            norm_result = fit_transform_local(tensor, hospital_id_by_patient)
        if norm_result.hospitals_with_insufficient_data:
            logger.warning(
                f"[TIMESERIES] hospitals with >=1 feature having ZERO real observations cohort-wide "
                f"(imputed to 0 post-standardization): {norm_result.hospitals_with_insufficient_data}"
            )

        out_dir = Path(cfg.get("run.output_dir", "outputs")) / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        sequences_path = out_dir / "timeseries_sequences.npz"
        np.savez_compressed(
            sequences_path,
            patient_ids=np.array(tensor.patient_ids),
            feature_names=np.array(feature_names),
            array=norm_result.array,
            sequence_length=tensor.sequence_length,
            observed_mask=tensor.observed_mask,
        )
        logger.info(f"[PHASE4] Sequence tensor written to {sequences_path}")

        embeddings_path = None
        encoder_type = cfg.get("modalities.timeseries.encoder", "lstm")
        valid_encoders = ("lstm", "rnn", "gru", "transformer")
        if encoder_type not in valid_encoders:
            raise ValueError(
                f"modalities.timeseries.encoder={encoder_type!r} is not valid -- must be one of "
                f"{valid_encoders}. Never silently defaulted or guessed."
            )

        try:
            from src.modalities.timeseries.lstm_encoder import TORCH_AVAILABLE
        except ImportError:
            TORCH_AVAILABLE = False

        if not TORCH_AVAILABLE:
            logger.warning(
                f"[PHASE4] torch is not installed in this environment -- {encoder_type.upper()} "
                "forward pass SKIPPED (not silently omitted: this is an explicit, logged skip). The "
                "sequence tensor above is a complete, usable Phase 4 deliverable on its own; install "
                "torch (`pip install -r requirements.txt`) to also produce embeddings."
            )
        else:
            import torch
            from src.utils.batching import run_batched_with_backoff, is_memory_error

            hidden_dim = cfg.get("modalities.timeseries.hidden_dim", 128)
            num_layers = cfg.get("modalities.timeseries.num_layers", 2)
            dropout = cfg.get("modalities.timeseries.dropout", 0.2)
            initial_batch_size = cfg.get("modalities.timeseries.lstm_inference_batch_size", 512)

            # ADDED 2026-09-06 (user request): modalities.timeseries.encoder
            # selects the architecture. "lstm" is the ORIGINAL, untouched
            # code path (LSTMTimeSeriesEncoder is never modified by this
            # change) -- every other branch is a new sibling encoder
            # (src/modalities/timeseries/{rnn,gru,transformer}_encoder.py)
            # sharing the EXACT SAME forward(x, lengths) -> (batch,
            # hidden_dim) contract, so the batching/inference code below
            # doesn't need to know or care which one is active.
            if encoder_type == "lstm":
                from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig, LSTMTimeSeriesEncoder
                encoder = LSTMTimeSeriesEncoder(
                    LSTMEncoderConfig(input_dim=len(feature_names), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
                )
            elif encoder_type == "rnn":
                from src.modalities.timeseries.rnn_encoder import RNNEncoderConfig, RNNTimeSeriesEncoder
                encoder = RNNTimeSeriesEncoder(
                    RNNEncoderConfig(input_dim=len(feature_names), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
                )
            elif encoder_type == "gru":
                from src.modalities.timeseries.gru_encoder import GRUEncoderConfig, GRUTimeSeriesEncoder
                encoder = GRUTimeSeriesEncoder(
                    GRUEncoderConfig(input_dim=len(feature_names), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
                )
            else:  # transformer
                from src.modalities.timeseries.transformer_encoder import TransformerEncoderConfig, TransformerTimeSeriesEncoder
                encoder = TransformerTimeSeriesEncoder(
                    TransformerEncoderConfig(
                        input_dim=len(feature_names), hidden_dim=hidden_dim, num_layers=num_layers,
                        num_heads=cfg.get("modalities.timeseries.transformer_num_heads", 4),
                        dim_feedforward=cfg.get("modalities.timeseries.transformer_dim_feedforward", 256),
                        dropout=dropout, max_bins=max_bins,
                    )
                )
            encoder.eval()

            # IMPORTANT: a real full-mode run (199,108 patients) crashed
            # here with a PyTorch CPU allocator OOM
            # ("DefaultCPUAllocator: not enough memory: you tried to
            # allocate 5780424704 bytes") because the entire cohort was
            # run through the LSTM in ONE forward pass. An LSTM's
            # internal working memory scales with total timesteps across
            # the whole batch, not just the input tensor's own size (the
            # input itself was a modest ~420MB; the crash was in the
            # LSTM's internal gate/hidden-state buffers). Fixed by
            # batching inference with automatic backoff -- see
            # src/utils/batching.py and docs/memory_efficiency.md. The
            # same batching applies regardless of which encoder is
            # selected above -- RNN/GRU have the same per-timestep
            # working-memory scaling as LSTM, and Transformer's
            # attention scales quadratically with sequence length, so if
            # anything the new architectures need this protection just
            # as much, not less.
            #
            # torch.from_numpy (not torch.tensor) avoids an extra copy of
            # the ~420MB array; batches are VIEWS into it, not copies.
            x_full = torch.from_numpy(norm_result.array)
            lengths_full = torch.from_numpy(tensor.sequence_length.astype("int64"))
            embeddings = np.empty((len(tensor.patient_ids), hidden_dim), dtype=np.float32)

            def _process_batch(start: int, end: int) -> None:
                with torch.no_grad():
                    embeddings[start:end] = encoder(x_full[start:end], lengths_full[start:end]).numpy()

            final_batch_size = run_batched_with_backoff(
                n_items=len(tensor.patient_ids),
                process_batch=_process_batch,
                initial_batch_size=initial_batch_size,
                logger=logger,
                is_retryable_error=is_memory_error,
                label=f"{encoder_type.upper()}_INFERENCE",
            )
            if final_batch_size < initial_batch_size:
                logger.warning(
                    f"[PHASE4] {encoder_type.upper()} inference needed to shrink batch_size from "
                    f"{initial_batch_size} to {final_batch_size} due to memory pressure -- consider "
                    f"lowering modalities.timeseries.lstm_inference_batch_size in your config to "
                    f"start there next time and skip the retries."
                )

            # "lstm" keeps the ORIGINAL canonical filename, untouched --
            # every other encoder gets its own distinctly-named file, so
            # timeseries_embeddings.npz is NEVER overwritten by anything
            # but a real LSTM run, regardless of what encoder was used
            # most recently.
            embeddings_filename = "timeseries_embeddings.npz" if encoder_type == "lstm" else f"timeseries_embeddings_{encoder_type}.npz"
            embeddings_path = out_dir / embeddings_filename
            np.savez_compressed(embeddings_path, patient_ids=np.array(tensor.patient_ids), embeddings=embeddings)
            logger.info(f"[PHASE4] {encoder_type.upper()} embeddings ({embeddings.shape}) written to {embeddings_path}")

    return {
        "sequences_path": str(sequences_path),
        "embeddings_path": str(embeddings_path) if embeddings_path else None,
        "encoder": encoder_type,
        "feature_names": feature_names,
        "patient_ids": tensor.patient_ids,
        "sequence_length": tensor.sequence_length,
        "tensor": tensor,
        "normalized_array": norm_result.array,
        "torch_available": TORCH_AVAILABLE,
    }


def load_phase4_result(cfg) -> dict:
    """Reconstruction of Phase 4's result for `--start-phase` resumption.

    EXTENDED 2026-08-30 (Phase 11): previously only `embeddings_path`
    was ever read downstream (by Phase 7), so this was intentionally a
    partial round-trip. Phase 11 needs the raw normalized sequence
    tensor itself (patient_ids, array, sequence_length) -- it trains
    the LSTM jointly with the rest of the local model, so it cannot
    work from Phase 4's already-embedded, frozen output the way Phase
    7 does. All four of `timeseries_sequences.npz`'s saved arrays are
    now read back, matching what a freshly-run
    run_phase4_timeseries_modality() call returns at the top level.

    BUG FIXED 2026-09-10 (found via the pre-existing test suite while
    starting the ablation-framework work, not hypothetical):
    `embeddings_path` was hardcoded to the canonical LSTM filename
    (`timeseries_embeddings.npz`) regardless of `modalities.timeseries.
    encoder`. Once that config key was filled in (2026-09-06) to
    actually select an architecture, any non-default encoder value
    writes to `timeseries_embeddings_{encoder}.npz` instead (see
    run_phase4_timeseries_modality()) -- so a `--start-phase 7+` run
    with a non-default encoder in the config would silently reload
    `embeddings_path=None` (file not found under the wrong name) and
    Phase 7 would raise "missing Phase 4 LSTM embeddings" even though
    the real file was sitting right there under its encoder-specific
    name. `test_full_run_then_resume_from_phase7_after_deleting_raw_
    data` reproduced this exactly once `configs/base.yaml`'s checked-in
    `encoder: transformer` value was in effect. Now mirrors
    run_phase4_timeseries_modality()'s own filename logic exactly.
    """
    out_dir = Path(cfg.get("run.output_dir", "outputs")) / "embeddings"
    encoder_type = cfg.get("modalities.timeseries.encoder", "lstm")
    embeddings_filename = "timeseries_embeddings.npz" if encoder_type == "lstm" else f"timeseries_embeddings_{encoder_type}.npz"
    embeddings_path = out_dir / embeddings_filename
    sequences_path = out_dir / "timeseries_sequences.npz"
    if not sequences_path.exists():
        raise FileNotFoundError(
            f"--start-phase > 4 requires Phase 4 to have already run and saved {sequences_path} -- "
            f"it hasn't. Run with --start-phase <= 4 at least once first."
        )
    seq_data = np.load(sequences_path)
    return {
        "embeddings_path": str(embeddings_path) if embeddings_path.exists() else None,
        "sequences_path": str(sequences_path),
        "patient_ids": seq_data["patient_ids"],
        "sequence_length": seq_data["sequence_length"],
        "normalized_array": seq_data["array"],
        "feature_names": seq_data["feature_names"].tolist(),
    }


def run_phase5_tabular_modality(cfg, phase1_summary: dict, phase3_result: dict, partition_result, train_patient_ids: "set | None" = None) -> dict:
    """Phase 5 -- tabular modality (Section 11).

    Source tables: patient, apachePredVar, diagnosis. Feature selection
    uses an explicit, documented ALLOWLIST (not a blocklist) -- see
    src/data/tabular/feature_engineering.py's module docstring for the
    full leakage-prevention rationale (diedinhospital excluded as the
    literal outcome; SOFA-consumed physiology columns excluded since
    they were used to construct the sepsis label).

    Same architecture as Phase 4: hospital-wise processing (no pooling),
    per-hospital local normalization, batched inference with OOM backoff,
    and an explicit skip (not a crash) if torch is unavailable.

    `train_patient_ids` (added 2026-09-11, part of making Phase 11/12/13
    split-aware -- see IMPLEMENTATION_TRACKER.md): OPTIONAL, defaults to
    None, which reproduces this function's ORIGINAL behavior EXACTLY
    (categorical vocabularies AND numeric normalization both fit from
    the full cohort, unchanged) -- verified by the full pre-existing
    test suite passing unmodified. When given a set of patient IDs, BOTH
    the categorical vocabularies (determine_full_cohort_vocabularies()/
    determine_diagnosis_vocabulary_from_shards(), fed a train-only-
    filtered cohort/shards) AND the numeric-feature normalization
    (fit_transform_tabular_local_split_aware()) are fit using ONLY those
    patients, then applied (one-hot-encoded / transformed) for every
    patient. This is a leakage-audit fix, not a new feature: this
    project's held-out split only means something if every fitted
    statistic -- not just the classifier -- never sees val/test data.
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))
    paths = phase1_summary["table_paths"]

    with PhaseTimer(logger, "PHASE5_TABULAR_MODALITY"):
        from src.data.ingestion.hospital_sharding import load_hospital_shard, shard_table_by_hospital
        from src.data.ingestion.loading import load_filtered
        from src.data.tabular.feature_engineering import (
            APACHEPREDVAR_CATEGORICAL_FEATURES, APACHEPREDVAR_NUMERIC_FEATURES,
            PATIENT_NUMERIC_FEATURES,
            assemble_tabular_features, attach_hospital_features,
            determine_diagnosis_vocabulary_from_shards, determine_full_cohort_vocabularies,
        )
        from src.data.tabular.normalization import fit_transform_tabular_local

        cohort_patients = phase3_result["cohort_patients"]
        patient_hospital_map = phase3_result["patient_hospital_map"]

        # FIXED 2026-08-24: hospital_region/hospital_teachingstatus
        # (PATIENT_CATEGORICAL_FEATURES) are sourced from hospital.csv,
        # not apachePredVar -- see attach_hospital_features()'s
        # docstring. hospital.csv is tiny (~208 rows), safe to load in
        # full alongside the small patient/apachePredVar tables.
        hospital_df = pd.read_csv(paths["hospital"])
        cohort_patients = attach_hospital_features(cohort_patients, hospital_df)
        patient_ids = cohort_patients["patientunitstayid"].tolist()

        chunksize = cfg.get("data.chunksize", 200_000)
        cache_dir = cfg.get("run.cache_dir", "cache")
        top_k = cfg.get("modalities.tabular.top_k_categorical", 10)
        top_k_overrides = cfg.get("modalities.tabular.top_k_categorical_overrides", {}) or {}
        diag_lookback = cfg.get("modalities.tabular.diagnosis_lookback_minutes", 1440)
        diag_cutoff = cfg.get("modalities.tabular.diagnosis_cutoff_minutes", 1440)

        # apachePredVar is small -- load once for the whole cohort,
        # column-projected to exactly the allowlisted fields.
        apv_columns = ["patientunitstayid"] + APACHEPREDVAR_NUMERIC_FEATURES + APACHEPREDVAR_CATEGORICAL_FEATURES
        apache_pred_var = load_filtered(paths["apachePredVar"], set(patient_ids), chunksize, usecols=apv_columns)

        # diagnosis is large -- shard by hospital like the Phase 3/4 tables.
        diagnosis_manifest = None
        if "diagnosis" in paths:
            diagnosis_manifest = shard_table_by_hospital(paths["diagnosis"], "diagnosis", patient_hospital_map, cache_dir, chunksize)

        # --- Categorical vocabularies: full cohort / all shards ---
        #
        # FIXED 2026-08-23 (vocabulary-sampling-bias bug): this used to
        # sample only the first `n_vocab_hospitals` hospitals by SORTED
        # hospital ID (`list(partition_result.partitions.keys())[:10]`)
        # -- always the smallest-ID hospitals, not a representative
        # sample, since hospital_partition.py builds `partitions` via
        # `sorted(available_hospitals)`. Confirmed genuinely fixed by
        # the real full-scale run: `hospitaladmitsource`, `diagnosis`,
        # and `apacheadmissiondx` all now show real diversity beyond
        # the old top_k=10 ceiling (see IMPLEMENTATION_TRACKER.md).
        #
        # `teachtype`/`region` ALSO showed vocab=1 in that same original
        # run, which looked like the same bug -- but stayed at vocab=1
        # even after this fix, because it wasn't (solely) this bug: see
        # APACHEPREDVAR_CATEGORICAL_FEATURES' comment and
        # attach_hospital_features() above -- apachePredVar.teachtype/
        # region are genuinely constant in this real eICU export, no
        # sampling strategy would have surfaced diversity that isn't
        # there. Fixed 2026-08-24 by sourcing hospital_region/
        # hospital_teachingstatus from hospital.csv instead, which DOES
        # have real diversity.
        #
        # Fix: patient/apachePredVar vocab now comes from the FULL
        # cohort (safe -- both tables, plus hospital.csv merged in
        # above, are already fully in memory here, not hospital-sharded,
        # so this adds no new cross-hospital pooling; see
        # determine_full_cohort_vocabularies()'s docstring). diagnosis
        # vocab now aggregates across ALL diagnosis-bearing hospital
        # shards via running value counts, one hospital at a time, never
        # pooling raw diagnosis rows (see
        # determine_diagnosis_vocabulary_from_shards()) -- preserving
        # the project's hospital-sharding architecture for the one
        # source table here that's actually large enough to need it.
        vocabularies = determine_full_cohort_vocabularies(
            cohort_patients[cohort_patients["patientunitstayid"].isin(train_patient_ids)] if train_patient_ids is not None else cohort_patients,
            apache_pred_var[apache_pred_var["patientunitstayid"].isin(train_patient_ids)] if train_patient_ids is not None else apache_pred_var,
            top_k, top_k_overrides,
        )
        if train_patient_ids is not None:
            logger.info(f"[TABULAR] SPLIT-AWARE vocabularies: fit from {len(train_patient_ids)} train patients only.")

        if diagnosis_manifest is not None:
            all_diagnosis_hospital_ids = [int(h) for h in diagnosis_manifest.hospital_shard_paths.keys()]

            def _load_shard_for_vocab(hid):
                shard = load_hospital_shard(diagnosis_manifest, hid)
                if train_patient_ids is not None and shard is not None and not shard.empty:
                    shard = shard[shard["patientunitstayid"].isin(train_patient_ids)]
                return shard

            diagnosis_vocabulary = determine_diagnosis_vocabulary_from_shards(
                _load_shard_for_vocab,
                all_diagnosis_hospital_ids,
                top_k_overrides.get("diagnosis", top_k),
            )
        else:
            all_diagnosis_hospital_ids = []
            diagnosis_vocabulary = []

        logger.info(
            f"[TABULAR] vocabularies determined from the full {len(patient_ids):,}-patient cohort "
            f"(patient/hospital/apachePredVar) and all {len(all_diagnosis_hospital_ids)} diagnosis-bearing "
            f"hospitals: "
            + ", ".join(f"{k}={len(v)}" for k, v in vocabularies.items())
            + f", diagnosis={len(diagnosis_vocabulary)}"
        )

        # --- Per-hospital feature assembly (no pooling of raw tables) ---
        feature_frames = []
        n_hospitals_total = sum(1 for p in partition_result.partitions.values() if p.patient_ids)
        n_hospitals_done = 0
        loop_start = time.time()

        for hospital_id, partition in partition_result.partitions.items():
            if not partition.patient_ids:
                continue
            hosp_patients = cohort_patients[cohort_patients["patientunitstayid"].isin(partition.patient_ids)]
            hosp_apv = apache_pred_var[apache_pred_var["patientunitstayid"].isin(partition.patient_ids)]
            hosp_diagnosis = (
                load_hospital_shard(diagnosis_manifest, hospital_id) if diagnosis_manifest is not None
                else pd.DataFrame(columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
            )

            hosp_features = assemble_tabular_features(
                hosp_patients, hosp_apv, hosp_diagnosis, partition.patient_ids,
                vocabularies, diagnosis_vocabulary, diag_lookback, diag_cutoff,
            )
            feature_frames.append(hosp_features)

            n_hospitals_done += 1
            elapsed = time.time() - loop_start
            if n_hospitals_done % 20 == 0 or n_hospitals_done == n_hospitals_total:
                rate = n_hospitals_done / elapsed if elapsed > 0 else 0
                eta = (n_hospitals_total - n_hospitals_done) / rate if rate > 0 else float("nan")
                logger.info(
                    f"[PHASE5] progress: {n_hospitals_done}/{n_hospitals_total} hospitals "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

        all_features_df = pd.concat(feature_frames, axis=0).reindex(patient_ids)
        feature_names = list(all_features_df.columns)

        numeric_column_names = set(PATIENT_NUMERIC_FEATURES) | set(APACHEPREDVAR_NUMERIC_FEATURES)
        numeric_column_mask = np.array([c in numeric_column_names for c in feature_names])
        logger.info(
            f"[TABULAR] assembled {len(feature_names)} features "
            f"({int(numeric_column_mask.sum())} numeric, {int((~numeric_column_mask).sum())} one-hot/categorical) "
            f"for {len(patient_ids):,} patients"
        )

        feature_array = all_features_df.to_numpy(dtype=np.float32)
        if train_patient_ids is not None:
            from src.data.tabular.normalization import fit_transform_tabular_local_split_aware
            norm_result = fit_transform_tabular_local_split_aware(
                feature_array, patient_ids, patient_hospital_map, numeric_column_mask, train_patient_ids,
            )
            logger.info(f"[TABULAR] SPLIT-AWARE normalization: fit from {len(train_patient_ids)} train patients only.")
        else:
            norm_result = fit_transform_tabular_local(feature_array, patient_ids, patient_hospital_map, numeric_column_mask)
        if norm_result.hospitals_with_insufficient_data:
            logger.warning(
                f"[TABULAR] hospitals with >=1 numeric feature having ZERO real observations "
                f"cohort-wide (imputed to 0 post-standardization): {norm_result.hospitals_with_insufficient_data}"
            )

        out_dir = Path(cfg.get("run.output_dir", "outputs")) / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        features_path = out_dir / "tabular_features.npz"
        np.savez_compressed(
            features_path,
            patient_ids=np.array(patient_ids),
            feature_names=np.array(feature_names),
            array=norm_result.array,
        )
        logger.info(f"[PHASE5] Tabular feature matrix written to {features_path}")

        embeddings_path = None
        encoder_type = cfg.get("modalities.tabular.encoder", "autoencoder")
        valid_encoders = ("autoencoder", "mlp")
        if encoder_type not in valid_encoders:
            raise ValueError(
                f"modalities.tabular.encoder={encoder_type!r} is not valid -- must be one of "
                f"{valid_encoders}. Never silently defaulted or guessed."
            )

        try:
            from src.modalities.tabular.autoencoder import TORCH_AVAILABLE
        except ImportError:
            TORCH_AVAILABLE = False

        if not TORCH_AVAILABLE:
            logger.warning(
                f"[PHASE5] torch is not installed in this environment -- {encoder_type} forward "
                "pass SKIPPED (not silently omitted: this is an explicit, logged skip). The tabular "
                "feature matrix above is a complete, usable Phase 5 deliverable on its own; "
                "install torch (`pip install -r requirements.txt`) to also produce embeddings."
            )
        else:
            import torch
            from src.utils.batching import is_memory_error, run_batched_with_backoff

            hidden_dim = cfg.get("modalities.tabular.hidden_dim", 128)
            embedding_dim = cfg.get("modalities.tabular.embedding_dim", 64)
            dropout = cfg.get("modalities.tabular.dropout", 0.1)
            initial_batch_size = cfg.get("modalities.tabular.inference_batch_size", 4096)

            # ADDED 2026-09-06 (user request): modalities.tabular.encoder
            # selects the architecture. "autoencoder" is the ORIGINAL,
            # untouched code path (TabularAutoencoder is never modified
            # by this change). "mlp" is a new sibling
            # (src/modalities/tabular/mlp_encoder.py) with NO decoder/
            # reconstruction -- its forward() returns a single tensor,
            # not the (embedding, reconstruction) tuple TabularAutoencoder
            # returns, hence the `returns_tuple` branch below.
            if encoder_type == "autoencoder":
                from src.modalities.tabular.autoencoder import AutoencoderConfig, TabularAutoencoder
                model = TabularAutoencoder(AutoencoderConfig(
                    input_dim=norm_result.array.shape[1], hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim, dropout=dropout,
                ))
                returns_tuple = True
            else:  # mlp
                from src.modalities.tabular.mlp_encoder import MLPEncoderConfig, TabularMLPEncoder
                model = TabularMLPEncoder(MLPEncoderConfig(
                    input_dim=norm_result.array.shape[1], hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim, dropout=dropout,
                ))
                returns_tuple = False
            model.eval()

            x_full = torch.from_numpy(norm_result.array)
            embeddings = np.empty((len(patient_ids), embedding_dim), dtype=np.float32)

            def _process_batch(start: int, end: int) -> None:
                with torch.no_grad():
                    out = model(x_full[start:end])
                    emb = out[0] if returns_tuple else out
                    embeddings[start:end] = emb.numpy()

            final_batch_size = run_batched_with_backoff(
                n_items=len(patient_ids),
                process_batch=_process_batch,
                initial_batch_size=initial_batch_size,
                logger=logger,
                is_retryable_error=is_memory_error,
                label=f"{encoder_type.upper()}_INFERENCE",
            )
            if final_batch_size < initial_batch_size:
                logger.warning(
                    f"[PHASE5] {encoder_type} inference needed to shrink batch_size from "
                    f"{initial_batch_size} to {final_batch_size} due to memory pressure."
                )

            # "autoencoder" keeps the ORIGINAL canonical filename,
            # untouched -- "mlp" gets its own distinctly-named file, so
            # tabular_embeddings.npz is NEVER overwritten by anything
            # but a real autoencoder run.
            embeddings_filename = "tabular_embeddings.npz" if encoder_type == "autoencoder" else f"tabular_embeddings_{encoder_type}.npz"
            embeddings_path = out_dir / embeddings_filename
            np.savez_compressed(embeddings_path, patient_ids=np.array(patient_ids), embeddings=embeddings)
            label_text = "Autoencoder" if encoder_type == "autoencoder" else encoder_type.upper()
            logger.info(f"[PHASE5] {label_text} embeddings ({embeddings.shape}) written to {embeddings_path}")

    return {
        "features_path": str(features_path),
        "embeddings_path": str(embeddings_path) if embeddings_path else None,
        "encoder": encoder_type,
        "feature_names": feature_names,
        "patient_ids": patient_ids,
        "normalized_array": norm_result.array,
        "torch_available": TORCH_AVAILABLE,
    }


def load_phase5_result(cfg) -> dict:
    """Reconstruction of Phase 5's result for `--start-phase` resumption.

    EXTENDED 2026-08-30 (Phase 11) -- see load_phase4_result()'s
    docstring for the identical rationale: Phase 11 trains the
    autoencoder jointly with the rest of the local model, so it needs
    the raw normalized tabular feature matrix itself, not just Phase
    5's already-embedded, frozen output.

    BUG FIXED 2026-09-10: same class of bug as load_phase4_result()'s
    fix, same day -- `embeddings_path` was hardcoded to the canonical
    autoencoder filename regardless of `modalities.tabular.encoder`.
    Now mirrors run_phase5_tabular_modality()'s own filename logic.
    """
    out_dir = Path(cfg.get("run.output_dir", "outputs")) / "embeddings"
    encoder_type = cfg.get("modalities.tabular.encoder", "autoencoder")
    embeddings_filename = "tabular_embeddings.npz" if encoder_type == "autoencoder" else f"tabular_embeddings_{encoder_type}.npz"
    embeddings_path = out_dir / embeddings_filename
    features_path = out_dir / "tabular_features.npz"
    if not features_path.exists():
        raise FileNotFoundError(
            f"--start-phase > 5 requires Phase 5 to have already run and saved {features_path} -- "
            f"it hasn't. Run with --start-phase <= 5 at least once first."
        )
    feat_data = np.load(features_path)
    return {
        "embeddings_path": str(embeddings_path) if embeddings_path.exists() else None,
        "features_path": str(features_path),
        "patient_ids": feat_data["patient_ids"],
        "normalized_array": feat_data["array"],
        "feature_names": feat_data["feature_names"].tolist(),
    }


def run_phase7_multimodal_fusion(cfg, phase3_result: dict, phase4_result: dict, phase5_result: dict) -> dict:
    """Phase 7 -- multimodal fusion.

    Concatenates Phase 4's LSTM embeddings + Phase 5's autoencoder
    embeddings into one unified per-patient representation. Text
    (Phase 6) is intentionally excluded -- locked decision, no clinical
    notes in this eICU export; see docs/text_modality_decision.md.

    Requires both Phase 4 and Phase 5 to have produced real embeddings
    (torch installed) -- raises rather than silently fusing a subset or
    fabricating a placeholder for whichever modality is missing, per
    this project's "no silent skips or fabrication" rule (Section 43).
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE7_MULTIMODAL_FUSION"):
        from src.fusion.multimodal_fusion import fuse_embeddings

        fusion_type = cfg.get("fusion.type", "concatenation")
        if fusion_type != "concatenation":
            raise NotImplementedError(
                f"fusion.type={fusion_type!r} is not implemented -- only 'concatenation' exists so far."
            )

        lstm_path = phase4_result.get("embeddings_path")
        ae_path = phase5_result.get("embeddings_path")
        missing = []
        if not lstm_path:
            missing.append("Phase 4 LSTM embeddings")
        if not ae_path:
            missing.append("Phase 5 autoencoder embeddings")
        if missing:
            raise RuntimeError(
                f"Phase 7 requires both Phase 4 and Phase 5 embeddings; missing: {', '.join(missing)}. "
                "Both require torch to be installed -- see docs/timeseries_modality.md and "
                "docs/tabular_modality.md."
            )

        lstm_data = np.load(lstm_path)
        ae_data = np.load(ae_path)

        # Authoritative output order: the full cohort as Phase 3 built
        # it, NOT either modality's own storage order (which currently
        # happens to match, by construction -- see fuse_embeddings()'s
        # docstring for why this isn't assumed rather than verified).
        cohort_patient_ids = phase3_result["cohort_patients"]["patientunitstayid"].tolist()

        fused_patient_ids, fused_embeddings = fuse_embeddings(
            lstm_data["patient_ids"].tolist(), lstm_data["embeddings"],
            ae_data["patient_ids"].tolist(), ae_data["embeddings"],
            target_order=cohort_patient_ids,
        )

        lstm_dim = lstm_data["embeddings"].shape[1]
        ae_dim = ae_data["embeddings"].shape[1]
        actual_dim = lstm_dim + ae_dim
        configured_dim = cfg.get("fusion.output_dim")
        if configured_dim is not None and configured_dim != actual_dim:
            logger.warning(
                f"[FUSION] configs' fusion.output_dim={configured_dim} does not match the actual "
                f"concatenated dimension {actual_dim} (lstm={lstm_dim} + autoencoder={ae_dim}). "
                f"fusion.type=concatenation (Option A, locked 2026-08-25) produces whatever dimension "
                f"the source embeddings sum to -- output_dim is descriptive here, not enforced. Update "
                f"config to silence this warning."
            )

        out_dir = Path(cfg.get("run.output_dir")) / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        fused_path = out_dir / "fused_embeddings.npz"
        np.savez_compressed(
            fused_path,
            patient_ids=np.array(fused_patient_ids),
            embeddings=fused_embeddings,
            lstm_dim=lstm_dim,
            autoencoder_dim=ae_dim,
        )
        logger.info(f"[PHASE7] Fused embeddings ({fused_embeddings.shape}) written to {fused_path}")

    return {
        "fused_embeddings_path": str(fused_path),
        "n_patients": len(fused_patient_ids),
        "fused_dim": fused_embeddings.shape[1],
    }


def load_phase7_result(cfg) -> dict:
    """Minimal reconstruction of Phase 7's result for `--start-phase`
    resumption -- only `fused_embeddings_path` is read downstream (by
    Phase 8 and Phase 9)."""
    path = Path(cfg.get("run.output_dir", "outputs")) / "embeddings" / "fused_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"--start-phase > 7 requires Phase 7 to have already run and saved {path} -- it hasn't. "
            f"Run with --start-phase <= 7 at least once first."
        )
    return {"fused_embeddings_path": str(path)}


def run_phase8_patient_graph_construction(cfg, phase7_result: dict, partition_result) -> dict:
    """Phase 8 -- intra-hospital patient graph construction.

    Builds ONE similarity graph PER HOSPITAL from Phase 7's fused
    embeddings, using only that hospital's own patients -- never
    pooling across hospitals, same privacy boundary as every other
    phase (per-hospital normalization, per-hospital vocabulary
    aggregation, hospital-sharded large tables). See
    docs/patient_graph_construction.md for the exact-vs-approximate
    similarity design decision (Option B, locked 2026-08-26).
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE8_PATIENT_GRAPH_CONSTRUCTION"):
        from src.graph.patient_graph import build_hospital_graph

        metric = cfg.get("graph.metric", "cosine")
        if metric != "cosine":
            raise NotImplementedError(f"graph.metric={metric!r} is not implemented -- only 'cosine' exists so far.")

        method = cfg.get("graph.method", "knn")
        k = cfg.get("graph.k", 5)
        cosine_threshold = cfg.get("graph.cosine_threshold", 0.7)
        max_nodes_full_similarity = cfg.get("graph.max_nodes_full_similarity", 2000)

        fused_path = phase7_result.get("fused_embeddings_path")
        if not fused_path:
            raise RuntimeError("Phase 8 requires Phase 7's fused embeddings; none were produced.")

        fused_data = np.load(fused_path)
        fused_patient_ids = fused_data["patient_ids"]
        fused_embeddings = fused_data["embeddings"]
        # Same intersection-based filtering Phase 5 already established
        # (see hosp_patients = cohort_patients[...isin(partition.patient_ids)]
        # above) -- a partition patient absent from the fused embeddings
        # is NORMAL here (cohort construction legitimately excludes some
        # partitioned patients), not an error to raise on.
        row_of_patient = {int(pid): i for i, pid in enumerate(fused_patient_ids)}

        out_dir = Path(cfg.get("run.output_dir")) / "graphs"
        out_dir.mkdir(parents=True, exist_ok=True)

        hospital_ids = sorted(partition_result.partitions.keys())
        hospital_graph_paths: dict[str, str] = {}
        total_nodes = 0
        total_edges = 0

        for i, hid in enumerate(hospital_ids):
            hospital_patient_ids = [
                pid for pid in partition_result.partitions[hid].patient_ids if pid in row_of_patient
            ]
            hospital_embeddings = fused_embeddings[[row_of_patient[pid] for pid in hospital_patient_ids]]

            graph = build_hospital_graph(
                hospital_patient_ids, hospital_embeddings,
                method=method, k=k, cosine_threshold=cosine_threshold,
                max_nodes_full_similarity=max_nodes_full_similarity,
            )

            graph_path = out_dir / f"hospital_{hid}_graph.npz"
            np.savez_compressed(
                graph_path,
                patient_ids=graph["patient_ids"],
                edge_index=graph["edge_index"],
                edge_weight=graph["edge_weight"],
            )
            hospital_graph_paths[str(hid)] = str(graph_path)
            total_nodes += graph["n_nodes"]
            total_edges += graph["n_edges"]

            # Progress every 20 hospitals -- same convention as every
            # other long per-hospital loop in this project, after the
            # real incident where a silent multi-minute gap looked
            # indistinguishable from a hang (see IMPLEMENTATION_TRACKER.md).
            if (i + 1) % 20 == 0 or (i + 1) == len(hospital_ids):
                logger.info(f"[GRAPH] progress: {i + 1}/{len(hospital_ids)} hospitals")

        manifest_path = out_dir / "graph_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {"method": method, "metric": metric, "k": k, "cosine_threshold": cosine_threshold,
                 "hospital_graph_paths": hospital_graph_paths},
                f, indent=2,
            )

        logger.info(
            f"[PHASE8] {len(hospital_ids)} hospital graphs written to {out_dir} "
            f"(total nodes={total_nodes:,}, total edges={total_edges:,}, method={method})"
        )

    return {
        "graph_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "hospital_graph_paths": hospital_graph_paths,
        "n_hospitals": len(hospital_ids),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }


def load_phase8_result(cfg) -> dict:
    """Minimal reconstruction of Phase 8's result for `--start-phase`
    resumption -- only `manifest_path` is read downstream (by Phase
    9), which itself reads `hospital_graph_paths` back out of the
    manifest JSON directly. `hospital_graph_paths` is included here
    too, straight from that same manifest, for consistency with the
    normal return shape."""
    graph_dir = Path(cfg.get("run.output_dir", "outputs")) / "graphs"
    manifest_path = graph_dir / "graph_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"--start-phase > 8 requires Phase 8 to have already run and saved {manifest_path} -- "
            f"it hasn't. Run with --start-phase <= 8 at least once first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {
        "graph_dir": str(graph_dir),
        "manifest_path": str(manifest_path),
        "hospital_graph_paths": manifest["hospital_graph_paths"],
    }


def run_phase9_gnn_forward_pass(cfg, phase7_result: dict, phase8_result: dict) -> dict:
    """Phase 9 -- GNN forward pass over Phase 8's per-hospital graphs.

    Architecture + forward-pass verification only (randomly
    initialized weights, no training loop yet) -- same pattern as
    Phase 4's LSTM and Phase 5's autoencoder. ONE model (shared
    weights, what a genuine federated deployment would eventually
    FedAvg), applied independently to each hospital's own graph in its
    own forward pass -- never pooling patients across hospitals, same
    boundary every phase since Phase 3 has held.

    Phase 8's saved graph files intentionally do NOT duplicate node
    features (only patient_ids/edge_index/edge_weight) -- this reloads
    Phase 7's fused embeddings and re-slices them per hospital by
    patient ID (never by row position), the same ID-based-alignment
    discipline established in Phase 7 and Phase 8.
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE9_GNN_FORWARD_PASS"):
        import torch

        from src.gnn.patient_gnn import PatientGNN

        gnn_type = cfg.get("gnn.type", "graphsage")
        hidden_dim = cfg.get("gnn.hidden_dim", 128)
        n_layers = cfg.get("gnn.layers", 2)
        dropout = cfg.get("gnn.dropout", 0.2)
        seed = cfg.get("run.seed", 42)

        fused_path = phase7_result.get("fused_embeddings_path")
        manifest_path = phase8_result.get("manifest_path")
        if not fused_path or not manifest_path:
            raise RuntimeError(
                "Phase 9 requires both Phase 7's fused embeddings and Phase 8's graph manifest."
            )

        fused_data = np.load(fused_path)
        fused_patient_ids = fused_data["patient_ids"]
        fused_embeddings = fused_data["embeddings"]
        row_of_patient = {int(pid): i for i, pid in enumerate(fused_patient_ids)}
        in_dim = fused_embeddings.shape[1]

        with open(manifest_path) as f:
            manifest = json.load(f)
        hospital_graph_paths = manifest["hospital_graph_paths"]

        torch.manual_seed(seed)
        model = PatientGNN(in_dim, hidden_dim, n_layers=n_layers, gnn_type=gnn_type, dropout=dropout)
        model.eval()

        all_patient_ids: list[int] = []
        all_embeddings: list[np.ndarray] = []

        for i, (hid, graph_path) in enumerate(hospital_graph_paths.items()):
            graph_data = np.load(graph_path)
            hospital_patient_ids = graph_data["patient_ids"]
            if len(hospital_patient_ids) == 0:
                continue  # nothing to do for an empty hospital -- see Phase 8's real-data min=0 finding

            x = fused_embeddings[[row_of_patient[int(pid)] for pid in hospital_patient_ids]]
            x_tensor = torch.tensor(x, dtype=torch.float32)
            edge_index_tensor = torch.tensor(graph_data["edge_index"], dtype=torch.long)
            edge_weight_tensor = torch.tensor(graph_data["edge_weight"], dtype=torch.float32)

            with torch.no_grad():
                out = model(x_tensor, edge_index_tensor, edge_weight=edge_weight_tensor)

            all_patient_ids.extend(int(pid) for pid in hospital_patient_ids)
            all_embeddings.append(out.numpy())

            if (i + 1) % 20 == 0 or (i + 1) == len(hospital_graph_paths):
                logger.info(f"[GNN] progress: {i + 1}/{len(hospital_graph_paths)} hospitals")

        combined = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, hidden_dim))

        out_dir = Path(cfg.get("run.output_dir")) / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        gnn_path = out_dir / "gnn_embeddings.npz"
        # NOTE: patient order here is grouped by hospital (manifest
        # iteration order), NOT necessarily Phase 7's cohort order --
        # patient_ids is saved explicitly alongside embeddings
        # specifically so downstream consumers align by ID, never by
        # row position, same discipline as every embeddings artifact
        # since Phase 7.
        np.savez_compressed(gnn_path, patient_ids=np.array(all_patient_ids), embeddings=combined)

        # ADDED alongside Phase 10 (2026-08-30): nothing needed to
        # reload Phase 9's result until Phase 10 existed, so no
        # manifest was ever written -- gnn_type in particular was only
        # ever available in-memory, in the return dict below, which
        # main() didn't even capture. Mirrors Phase 8's
        # graph_manifest.json pattern exactly. See
        # IMPLEMENTATION_TRACKER.md's dated entry for this fix and
        # load_phase9_result() below.
        gnn_dim = combined.shape[1] if combined.size or hidden_dim else hidden_dim
        manifest_path = out_dir / "gnn_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {"gnn_type": gnn_type, "hidden_dim": hidden_dim, "n_patients": len(all_patient_ids),
                 "gnn_dim": gnn_dim, "gnn_embeddings_path": str(gnn_path)},
                f, indent=2,
            )

        logger.info(
            f"[PHASE9] GNN ({gnn_type}) embeddings ({combined.shape}) written to {gnn_path}"
        )

    return {
        "gnn_embeddings_path": str(gnn_path),
        "manifest_path": str(manifest_path),
        "n_patients": len(all_patient_ids),
        "gnn_dim": gnn_dim,
        "gnn_type": gnn_type,
    }


def load_phase9_result(cfg) -> dict:
    """Reloads Phase 9's result for `--start-phase` resumption.

    ADDED alongside Phase 10 (2026-08-30) -- Phase 9 was previously
    the last phase built, so nothing needed to reload its output yet;
    this (and the gnn_manifest.json write in
    run_phase9_gnn_forward_pass() above) is new infrastructure, not a
    pre-existing gap silently left unfixed. `gnn_type` is read back
    from the manifest rather than left unknown, mirroring
    load_phase8_result()'s use of graph_manifest.json.
    """
    out_dir = Path(cfg.get("run.output_dir", "outputs")) / "embeddings"
    gnn_path = out_dir / "gnn_embeddings.npz"
    manifest_path = out_dir / "gnn_manifest.json"
    missing = [p for p in (gnn_path, manifest_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"--start-phase > 9 requires Phase 9 to have already run and saved {missing} -- "
            f"they don't all exist. Run with --start-phase <= 9 at least once first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {
        "gnn_embeddings_path": str(gnn_path),
        "manifest_path": str(manifest_path),
        "n_patients": manifest["n_patients"],
        "gnn_dim": manifest["gnn_dim"],
        "gnn_type": manifest["gnn_type"],
    }


def run_phase10_multitask_heads(cfg, phase9_result: dict, phase3_result: dict) -> dict:
    """Phase 10 -- multi-task prediction heads (mortality + sepsis).

    Architecture + forward-pass verification only (randomly
    initialized weights, no training loop yet) -- same pattern as
    every prior modeling phase. Runs ONE global forward pass over
    Phase 9's full per-patient GNN embeddings, no hospital looping --
    see src/heads/multitask_heads.py's module docstring for why that's
    correct here, unlike Phase 9's genuinely per-hospital passes.

    Also ID-aligns Phase 3's mortality/sepsis labels to every patient
    the GNN embedded and logs coverage/class-balance stats -- proving
    what Phase 11's eventual training loop will need is correctly
    wired, before that loop exists. `multitask.sepsis_cohort` is
    validated to be exactly "broad" or "strict" inside
    align_labels_to_patient_ids() -- raises ValueError otherwise,
    never silently defaults or guesses (locked requirement, see
    configs/base.yaml).
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE10_MULTITASK_HEADS"):
        import torch

        from src.heads.multitask_heads import MultiTaskHeads, align_labels_to_patient_ids

        sepsis_cohort = cfg.get("multitask.sepsis_cohort")
        head_hidden_dim = cfg.get("multitask.head_hidden_dim", 64)
        head_dropout = cfg.get("multitask.head_dropout", 0.1)
        seed = cfg.get("run.seed", 42)

        gnn_path = phase9_result.get("gnn_embeddings_path")
        if not gnn_path:
            raise RuntimeError("Phase 10 requires Phase 9's GNN embeddings; none were produced.")

        gnn_data = np.load(gnn_path)
        patient_ids = gnn_data["patient_ids"].tolist()
        embeddings = gnn_data["embeddings"]
        in_dim = embeddings.shape[1]

        label_info = align_labels_to_patient_ids(
            patient_ids,
            phase3_result["cohort_patients"],
            phase3_result["mortality"],
            phase3_result["sepsis_df"],
            sepsis_cohort,
        )
        logger.info(
            f"[LABELS] mortality: known={label_info['n_mortality_known']} "
            f"positive={label_info['n_mortality_positive']} negative={label_info['n_mortality_negative']} "
            f"unknown={label_info['n_mortality_unknown']} | "
            f"sepsis_{sepsis_cohort}: known={label_info['n_sepsis_known']} "
            f"positive={label_info['n_sepsis_positive']} negative={label_info['n_sepsis_negative']} "
            f"unknown={label_info['n_sepsis_unknown']}"
        )

        torch.manual_seed(seed)
        model = MultiTaskHeads(in_dim, head_hidden_dim=head_hidden_dim, dropout=head_dropout)
        model.eval()

        x_tensor = torch.tensor(embeddings, dtype=torch.float32)
        with torch.no_grad():
            out = model(x_tensor)
        mortality_logits, sepsis_logits = out["mortality_logits"], out["sepsis_logits"]

        logger.info(
            f"[PHASE10] heads forward pass over {len(patient_ids)} patients -- "
            f"mortality_logits {tuple(mortality_logits.shape)}, sepsis_logits {tuple(sepsis_logits.shape)}"
        )

    return {
        "n_patients": len(patient_ids),
        "in_dim": in_dim,
        "head_hidden_dim": head_hidden_dim,
        "sepsis_cohort": sepsis_cohort,
        "mortality_logits_shape": tuple(mortality_logits.shape),
        "sepsis_logits_shape": tuple(sepsis_logits.shape),
        "n_mortality_known": label_info["n_mortality_known"],
        "n_mortality_positive": label_info["n_mortality_positive"],
        "n_mortality_negative": label_info["n_mortality_negative"],
        "n_mortality_unknown": label_info["n_mortality_unknown"],
        "n_sepsis_known": label_info["n_sepsis_known"],
        "n_sepsis_positive": label_info["n_sepsis_positive"],
        "n_sepsis_negative": label_info["n_sepsis_negative"],
        "n_sepsis_unknown": label_info["n_sepsis_unknown"],
    }


def _load_hospital_batch(
    hid: str, hospital_graph_paths: dict, ts_row_of_patient: dict, ts_array, ts_seq_length,
    tab_row_of_patient: dict, tab_array, phase3_result: dict, sepsis_cohort: str,
) -> "dict | None":
    """Loads and ID-aligns exactly one hospital's full batch: raw
    timeseries/tabular tensors (sliced by patient ID from Phase 4/5's
    already-normalized, full-cohort arrays), its own intra-hospital
    graph (Phase 8), and its patients' mortality/sepsis labels (Phase
    10's `align_labels_to_patient_ids()`, reused as-is). Extracted
    2026-09-03 (alongside Phase 13) from what was previously inline in
    `_run_local_training_round()` -- now shared by that function AND
    Phase 13's centralized-sequential baseline, which needs the
    identical per-hospital batch but in a training loop that never
    resets the model between hospitals (see
    run_phase13_baselines()'s docstring).

    Returns `None` for an empty hospital -- see Phase 8's real-data
    min=0 finding; there is nothing to train on. Raises `RuntimeError`
    if any patient in the graph is missing from Phase 4/5's raw
    tensors -- Phase 3/4/5/8 are expected to cover the exact same
    cohort, so this should never legitimately happen.

    EXTENDED 2026-09-11: the returned dict now also carries
    `patient_ids` (the exact row order every other array in this dict
    is aligned to) -- needed by src/split_aware/*'s train-only masking,
    which has to know WHICH row belongs to which patient to build a
    train/val/test row mask. Purely additive (existing consumers that
    only read the keys they already used are unaffected).
    """
    import torch

    from src.heads.multitask_heads import align_labels_to_patient_ids

    graph_data = np.load(hospital_graph_paths[hid])
    hospital_patient_ids = [int(pid) for pid in graph_data["patient_ids"]]
    if len(hospital_patient_ids) == 0:
        return None

    missing_ts = [pid for pid in hospital_patient_ids if pid not in ts_row_of_patient]
    missing_tab = [pid for pid in hospital_patient_ids if pid not in tab_row_of_patient]
    if missing_ts or missing_tab:
        raise RuntimeError(
            f"hospital {hid}: patient(s) present in Phase 8's graph but absent from Phase "
            f"4/5's raw tensors (timeseries missing={missing_ts[:5]}, tabular missing="
            f"{missing_tab[:5]}) -- Phase 3/4/5/8 are expected to cover the exact same cohort; "
            f"investigate before proceeding rather than silently dropping them."
        )

    timeseries_x = torch.tensor(
        ts_array[[ts_row_of_patient[pid] for pid in hospital_patient_ids]], dtype=torch.float32,
    )
    timeseries_lengths = torch.tensor(
        ts_seq_length[[ts_row_of_patient[pid] for pid in hospital_patient_ids]], dtype=torch.long,
    )
    tabular_x = torch.tensor(
        tab_array[[tab_row_of_patient[pid] for pid in hospital_patient_ids]], dtype=torch.float32,
    )
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    edge_weight = torch.tensor(graph_data["edge_weight"], dtype=torch.float32)

    label_info = align_labels_to_patient_ids(
        hospital_patient_ids, phase3_result["cohort_patients"], phase3_result["mortality"],
        phase3_result["sepsis_df"], sepsis_cohort,
    )
    mortality_labels = torch.tensor(label_info["mortality"], dtype=torch.float32)
    sepsis_labels = torch.tensor(label_info["sepsis"], dtype=torch.float32)

    return {
        "n_patients": len(hospital_patient_ids),
        "patient_ids": hospital_patient_ids,
        "timeseries_x": timeseries_x,
        "timeseries_lengths": timeseries_lengths,
        "tabular_x": tabular_x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "mortality_labels": mortality_labels,
        "sepsis_labels": sepsis_labels,
    }


def _build_train_only_subgraph(
    edge_index: "torch.Tensor", edge_weight: "torch.Tensor", train_row_mask: "torch.Tensor",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Given a hospital's FULL graph (`edge_index`/`edge_weight`, node
    indices 0..N-1 matching the batch's row order) and a boolean
    `train_row_mask` over those same N nodes, returns
    `(sub_edge_index, sub_edge_weight, train_indices)`:

      - `sub_edge_index`/`sub_edge_weight`: only the edges with BOTH
        endpoints in the train set, remapped to the new 0..(n_train-1)
        index space (so they can be fed straight into a model whose
        node features have ALSO been sliced down to `train_indices`).
      - `train_indices`: the ORIGINAL row indices (into the full N-node
        batch arrays) of the train nodes, in the exact order the
        remapped subgraph expects -- the caller slices
        timeseries_x/tabular_x/labels by this same index array.

    This is what makes split-aware training satisfy "validation/test
    patients must only pass through frozen components" LITERALLY, not
    just approximately: a non-train patient's row is never even
    present in the tensors the model's forward/backward pass touches
    during training, so there is no message-passing pathway (or any
    other pathway) for its raw feature values to influence a gradient
    -- not just its label. This also happens to be the textbook
    GraphSAGE INDUCTIVE-learning pattern (train on a subgraph, run
    inference later on an expanded graph that includes new nodes),
    not a bespoke restriction invented for this project.
    """
    import torch

    train_indices = torch.where(train_row_mask)[0]
    old_to_new = torch.full((train_row_mask.shape[0],), -1, dtype=torch.long)
    old_to_new[train_indices] = torch.arange(len(train_indices))

    src, dst = edge_index[0], edge_index[1]
    keep = train_row_mask[src] & train_row_mask[dst]
    sub_edge_index = torch.stack([old_to_new[src[keep]], old_to_new[dst[keep]]], dim=0)
    sub_edge_weight = edge_weight[keep]
    return sub_edge_index, sub_edge_weight, train_indices


def _run_local_training_round(
    cfg, phase3_result: dict, phase4_result: dict, phase5_result: dict, phase8_result: dict,
    model_config, initial_state_dict: dict, checkpoint_dir: Path, logger, round_tag: str = "PHASE11",
    local_epochs_override: "int | None" = None, train_patient_ids: "set | None" = None,
) -> dict:
    """Shared per-hospital local-training mechanics for exactly ONE
    round -- extracted 2026-09-01 (alongside Phase 12) so Phase 12's
    multi-round FedAvg orchestration reuses the EXACT SAME, already
    real-full-scale-verified per-hospital training code
    (run_phase11_local_training() itself becomes a thin wrapper that
    calls this once, with behavior completely unchanged from before
    the extraction -- see that function's own docstring and
    IMPLEMENTATION_TRACKER.md's dated entry for the extraction).

    Parameterized by `model_config` (so every round in a multi-round
    FedAvg run shares the identical architecture) and
    `initial_state_dict` (so a round can start from either a fresh
    seeded init -- Phase 11 alone, or Phase 12's round 1 when not
    reusing prior work -- or a PREVIOUS round's FedAvg-aggregated
    global model -- Phase 12's rounds 2+). `checkpoint_dir` and
    `round_tag` let each round's checkpoints and log lines land in
    their own clearly-labeled location rather than overwriting each
    other.

    `local_epochs_override`, added 2026-09-03 (alongside Phase 13):
    when given, takes precedence over `training.local_epochs` from
    config for THIS call only. Phase 13's local-only baseline uses
    this to train each hospital for the SAME TOTAL epoch budget
    federated hospitals see cumulatively across all rounds
    (`training.local_epochs * federated.n_rounds`), confirmed with the
    user (2026-09-03) as the fair basis for a local-vs-federated
    comparison -- a local baseline trained for only ONE round's worth
    of epochs would look artificially worse than federated purely from
    getting less total training, not from federation itself helping.

    Loss masking is per-TASK, not per-patient (src/training/losses.py):
    a patient with a known mortality outcome but unknown sepsis status
    (or vice versa) still contributes to whichever task's loss it has
    a real label for. A hospital with zero usable labels for BOTH
    tasks (small hospitals, or Phase 10's confirmed sepsis_df-absence
    condition) AND reconstruction disabled/zero-weighted is skipped
    entirely and logged -- never trained against a fabricated zero
    loss. A NaN/Inf total loss also skips that hospital (a SEPARATE,
    clearly distinguished reason) rather than saving a corrupted
    checkpoint -- found while checking Phase 11's real-run quality
    (2026-08-31): `.backward()`/`.step()` run cleanly on a NaN loss
    with no exception, so without this check a corrupted checkpoint
    could be saved silently while the run still reports success.

    Every hospital's loss_history/label-coverage counts/skip reasons
    are persisted to `{checkpoint_dir}/training_manifest.json` -- the
    only place this info survives past the process exiting (see
    IMPLEMENTATION_TRACKER.md's 2026-08-31 entry for why this matters).

    Also saves its OWN copy of `initial_state_dict` to
    `{checkpoint_dir}/global_initial_weights.pt` -- found missing
    (2026-09-02, while double-checking a docs claim before the user
    could hit it, not after a bug report): only
    run_phase11_local_training() saved this file before, at its own
    top-level checkpoint_dir, so Phase 12's per-round subdirectories
    (`checkpoints/phase12_fedavg/round_2/`, `round_3/`, ...) never got
    one, silently breaking `scripts/check_local_training_quality.py`
    (which looks for exactly this filename) for every round except
    Phase 11's own. Saving it here, once, makes every checkpoint_dir
    this helper ever writes to self-contained and independently
    inspectable the same way, regardless of which phase or round
    called it.

    `train_patient_ids`, added 2026-09-11 as part of making Phase
    11/12/13 genuinely leakage-safe against the permanent split
    manifest (src/data/splitting/patient_split.py): OPTIONAL, defaults
    to None, which reproduces this function's ORIGINAL behavior
    EXACTLY (every patient in the hospital trains, unchanged) --
    verified by the full pre-existing test suite passing unmodified.

    When given a set of patient IDs, the GRADIENT-COMPUTING forward/
    backward pass runs on a TRAIN-ONLY SUBGRAPH -- see
    `_build_train_only_subgraph()` above -- not the full hospital
    graph. A non-train (val/test) patient's row is never even present
    in the tensors that pass or fusion or the encoders or the GNN see
    during training, so there is no pathway (message-passing, the
    reconstruction term, or anything else) for its raw data to
    influence a gradient, satisfying "validation/test patients must
    only pass through frozen components" literally, not approximately.

    A REAL DESIGN CORRECTION MADE WHILE VERIFYING THIS (2026-09-11),
    not hypothetical: the first working version of this feature instead
    kept the FULL graph in the forward pass every epoch and masked only
    the LABELS to NaN for non-train patients (the standard semi-
    supervised/transductive GNN training regime, matching what
    src/ablation/graph_embedding_ablation_runner.py still does). A
    dedicated revert->fail->restore->pass-style test
    (test_phase11_split_aware.py's
    `test_split_aware_training_is_NOT_sensitive_to_a_non_train_patients_
    raw_values`) caught that this was NOT sufficient: even with labels
    masked, a non-train node's corrupted FEATURES still changed a
    train neighbor's trained weights via message-passing, because the
    two nodes shared a graph edge. Labels never leaked under that
    design, but raw FEATURE values could, through one specific pathway
    -- which is still a real violation of "only frozen components." The
    train-only-subgraph approach closes this completely: there is no
    edge for a non-train node's features to travel across during
    training, because the node itself isn't in the training graph at
    all. (This also means the earlier "slice the reconstruction term to
    train rows" fix, while correct as far as it went, is now
    subsumed -- reconstruction is computed over the train-only batch
    the whole model already operates on, with nothing extra needed.)

    AFTER training, one additional FROZEN forward pass (`model.eval()`,
    `torch.no_grad()`) is run over the FULL hospital graph (every
    patient, train+val+test) to produce predictions for val/test
    patients -- this is read-only inference against already-fixed
    weights, exactly "passing through frozen components," and is also
    the standard GraphSAGE inductive-learning use case (train on a
    subgraph, infer on an expanded graph including previously-unseen
    nodes) rather than a bespoke restriction invented for this project.
    """
    import torch

    from src.training.local_model import LocalPatientModel
    from src.training.losses import compute_multitask_loss

    torch.save(initial_state_dict, checkpoint_dir / "global_initial_weights.pt")

    sepsis_cohort = cfg.get("multitask.sepsis_cohort")
    mortality_weight = cfg.get("multitask.mortality_weight", 1.0)
    sepsis_weight = cfg.get("multitask.sepsis_weight", 1.0)
    loss_type = cfg.get("multitask.loss", "weighted_bce")

    local_epochs = local_epochs_override if local_epochs_override is not None else cfg.get("training.local_epochs", 5)
    learning_rate = cfg.get("training.learning_rate", 0.001)
    weight_decay = cfg.get("training.weight_decay", 0.0001)
    optimizer_name = cfg.get("training.optimizer", "adam")
    reconstruction_weight = cfg.get("training.reconstruction_weight", 0.0)
    max_hospitals = cfg.get("training.max_hospitals", None)

    if optimizer_name not in ("adam", "sgd"):
        raise ValueError(f"training.optimizer={optimizer_name!r} not recognized -- must be 'adam' or 'sgd'.")

    ts_patient_ids = [int(pid) for pid in phase4_result["patient_ids"]]
    ts_row_of_patient = {pid: i for i, pid in enumerate(ts_patient_ids)}
    ts_array = phase4_result["normalized_array"]
    ts_seq_length = phase4_result["sequence_length"]

    tab_patient_ids = [int(pid) for pid in phase5_result["patient_ids"]]
    tab_row_of_patient = {pid: i for i, pid in enumerate(tab_patient_ids)}
    tab_array = phase5_result["normalized_array"]

    hospital_graph_paths = phase8_result["hospital_graph_paths"]
    hospital_ids = sorted(hospital_graph_paths.keys())
    if max_hospitals is not None:
        hospital_ids = hospital_ids[:max_hospitals]

    hospital_results: dict = {}
    n_trained = 0
    n_skipped_empty = 0
    n_skipped_no_labels = 0
    n_skipped_non_finite = 0
    n_skipped_no_train_patients = 0

    for i, hid in enumerate(hospital_ids):
        batch = _load_hospital_batch(
            hid, hospital_graph_paths, ts_row_of_patient, ts_array, ts_seq_length,
            tab_row_of_patient, tab_array, phase3_result, sepsis_cohort,
        )
        if batch is None:
            n_skipped_empty += 1
            continue  # nothing to do for an empty hospital -- see Phase 8's real-data min=0 finding

        train_row_mask = None
        train_indices = None
        model_ts_x, model_ts_len, model_tab_x = batch["timeseries_x"], batch["timeseries_lengths"], batch["tabular_x"]
        model_edge_index, model_edge_weight = batch["edge_index"], batch["edge_weight"]
        model_mortality, model_sepsis = batch["mortality_labels"], batch["sepsis_labels"]
        if train_patient_ids is not None:
            train_row_mask = torch.tensor([pid in train_patient_ids for pid in batch["patient_ids"]])
            if int(train_row_mask.sum().item()) == 0:
                n_skipped_no_train_patients += 1
                logger.warning(
                    f"[{round_tag}] hospital {hid} ({batch['n_patients']} patients) skipped: zero "
                    f"train-split patients -- nothing to train on for this hospital."
                )
                continue
            model_edge_index, model_edge_weight, train_indices = _build_train_only_subgraph(
                batch["edge_index"], batch["edge_weight"], train_row_mask,
            )
            model_ts_x = batch["timeseries_x"][train_indices]
            model_ts_len = batch["timeseries_lengths"][train_indices]
            model_tab_x = batch["tabular_x"][train_indices]
            model_mortality = batch["mortality_labels"][train_indices]
            model_sepsis = batch["sepsis_labels"][train_indices]

        model = LocalPatientModel(model_config)
        model.load_state_dict(initial_state_dict)  # every hospital starts from the SAME point -- see docstring
        model.train()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        loss_history: list[float] = []
        counts: dict = {"n_mortality": 0, "n_sepsis": 0}
        skip_reason = None
        for _epoch in range(local_epochs):
            optimizer.zero_grad()
            out = model(model_ts_x, model_ts_len, model_tab_x, model_edge_index, model_edge_weight)
            total_loss, _components, counts = compute_multitask_loss(
                out["mortality_logits"], out["sepsis_logits"], model_mortality, model_sepsis,
                mortality_weight, sepsis_weight, loss_type,
                reconstruction=out["tabular_reconstruction"], tabular_target=model_tab_x,
                reconstruction_weight=reconstruction_weight,
            )
            if total_loss is None:
                skip_reason = "no_labels"
                break
            if not torch.isfinite(total_loss):
                skip_reason = "non_finite_loss"
                break
            total_loss.backward()
            optimizer.step()
            loss_history.append(total_loss.item())

        if skip_reason == "no_labels":
            n_skipped_no_labels += 1
            logger.warning(
                f"[{round_tag}] hospital {hid} ({batch['n_patients']} patients) skipped: no usable "
                f"labels for either task and reconstruction disabled/zero-weighted."
            )
            continue
        if skip_reason == "non_finite_loss":
            n_skipped_non_finite += 1
            logger.warning(
                f"[{round_tag}] hospital {hid} ({batch['n_patients']} patients) skipped at epoch "
                f"{len(loss_history)}: loss was NaN/Inf. Checkpoint NOT saved for this hospital -- "
                f"investigate this hospital's data (extreme values, degenerate graph) rather than "
                f"assuming it's rare."
            )
            continue

        checkpoint_path = checkpoint_dir / f"hospital_{hid}_local_weights.pt"
        torch.save(model.state_dict(), checkpoint_path)
        hospital_results[hid] = {
            "n_patients": batch["n_patients"],
            "n_train_patients": int(len(train_indices)) if train_indices is not None else batch["n_patients"],
            "loss_history": loss_history,
            "final_loss": loss_history[-1],
            "n_mortality_known": counts["n_mortality"],
            "n_sepsis_known": counts["n_sepsis"],
            "checkpoint_path": str(checkpoint_path),
        }

        if train_patient_ids is not None:
            # One frozen forward pass over EVERY patient (train+val+
            # test), using the FULL hospital graph -- val/test patients
            # never influenced any weight above (they weren't even
            # present in the training subgraph), this is read-only
            # inference against the now-fixed trained model, exactly
            # "passing through frozen components."
            model.eval()
            with torch.no_grad():
                eval_out = model(
                    batch["timeseries_x"], batch["timeseries_lengths"], batch["tabular_x"],
                    batch["edge_index"], batch["edge_weight"],
                )
            split_labels = ["train" if bool(m) else "not_train" for m in train_row_mask.tolist()]
            hospital_results[hid]["predictions"] = {
                "patient_ids": batch["patient_ids"],
                "mortality_logits": eval_out["mortality_logits"].detach().numpy().tolist(),
                "sepsis_logits": eval_out["sepsis_logits"].detach().numpy().tolist(),
                "mortality_labels": batch["mortality_labels"].numpy().tolist(),
                "sepsis_labels": batch["sepsis_labels"].numpy().tolist(),
                "is_train_row": split_labels,
            }
        n_trained += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(hospital_ids):
            logger.info(f"[{round_tag}] progress: {i + 1}/{len(hospital_ids)} hospitals")

    logger.info(
        f"[{round_tag}] trained {n_trained} hospitals, skipped {n_skipped_empty} empty, "
        f"{n_skipped_no_labels} with no usable labels for either task, "
        f"{n_skipped_non_finite} with a non-finite (NaN/Inf) loss"
        + (f", {n_skipped_no_train_patients} with zero train-split patients" if train_patient_ids is not None else "")
    )

    summary = {
        "n_hospitals_total": len(hospital_ids),
        "n_hospitals_trained": n_trained,
        "n_hospitals_skipped_empty": n_skipped_empty,
        "n_hospitals_skipped_no_labels": n_skipped_no_labels,
        "n_hospitals_skipped_non_finite": n_skipped_non_finite,
        "n_hospitals_skipped_no_train_patients": n_skipped_no_train_patients,
        "local_epochs": local_epochs,
        "loss_type": loss_type,
        "sepsis_cohort": sepsis_cohort,
        "reconstruction_weight": reconstruction_weight,
        "split_aware": train_patient_ids is not None,
        "hospitals": hospital_results,
    }
    manifest_path = checkpoint_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_path": str(manifest_path),
        "n_hospitals_trained": n_trained,
        "n_hospitals_skipped_empty": n_skipped_empty,
        "n_hospitals_skipped_no_labels": n_skipped_no_labels,
        "n_hospitals_skipped_non_finite": n_skipped_non_finite,
        "n_hospitals_skipped_no_train_patients": n_skipped_no_train_patients,
        "hospital_results": hospital_results,
    }


def run_phase11_local_training(
    cfg, phase3_result: dict, phase4_result: dict, phase5_result: dict, phase8_result: dict,
    train_patient_ids: "set | None" = None, checkpoint_dir_name: "str | None" = None,
) -> dict:
    """Phase 11 -- local training loop.

    THIN WRAPPER around _run_local_training_round() (extracted
    2026-09-01 alongside Phase 12 so both phases share the exact same,
    already real-full-scale-verified per-hospital mechanics --
    BEHAVIOR OF THIS FUNCTION IS UNCHANGED by that extraction; every
    real-run number and test written against it before the extraction
    still holds). This function's own job is just: build the shared
    `model_config`, create ONE fresh seeded set of initial weights,
    and call the shared per-round helper exactly once.

    FULL END-TO-END joint training (LSTM + autoencoder + GNN + heads,
    src/training/local_model.py's `LocalPatientModel`), per hospital,
    never pooling patients across hospitals -- same privacy boundary
    every phase since Phase 3 has held. DECIDED 2026-08-30, confirmed
    with the user: NOT heads-only. Directly grounded in Phase 9's own
    docstring ("ONE model... what a genuine federated deployment would
    eventually FedAvg") and TabularAutoencoder's decoder, built
    anticipating exactly this reconstruction-loss objective (see
    src/training/local_model.py and src/training/losses.py for the
    full reasoning).

    Only consumes phase3_result (labels), phase4_result/phase5_result
    (Phase 4/5's RAW normalized tensors, not their frozen embeddings --
    this phase trains those encoders, so it cannot work from
    already-embedded output the way Phase 7 does), and phase8_result
    (graphs) -- NOT phase1_summary, partition_result, phase7_result,
    or phase9_result.

    EVERY HOSPITAL STARTS FROM THE SAME INITIAL WEIGHTS ("global model"
    for this round) -- not independent random initializations per
    hospital. This matters beyond just this phase: FedAvg (Phase 12)
    averages weight TENSORS directly, which is only a coherent
    operation when every hospital's model started from the same point
    in weight-space. The shared starting point is saved to
    `checkpoints/phase11_local/global_initial_weights.pt`. Phase 12's
    round 1 can reuse this run's own checkpoints directly (see
    run_phase12_fedavg()'s docstring) rather than re-training from
    scratch.

    `train_patient_ids`/`checkpoint_dir_name`, added 2026-09-11: see
    `_run_local_training_round()`'s docstring for what split-aware
    training actually changes (a train-only subgraph for the gradient-
    computing forward/backward pass, with a separate frozen full-graph
    pass afterward for val/test predictions). `checkpoint_dir_name`
    defaults to `"phase11_local"` when `train_patient_ids is None`
    (byte-for-byte the original location) but to
    `"phase11_local_splitaware"` when a split IS given -- SAFE BY
    DEFAULT, per "keep all previous full-cohort Phase 11-13 results
    intact... separate, clearly identifiable output directories": a
    caller cannot silently overwrite the original full-cohort
    checkpoints just by forgetting to pass a different directory name.
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE11_LOCAL_TRAINING"):
        import torch

        from src.modalities.tabular.autoencoder import AutoencoderConfig
        from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig
        from src.training.local_model import LocalModelConfig, LocalPatientModel

        seed = cfg.get("run.seed", 42)
        ts_input_dim = len(phase4_result["feature_names"])
        tab_input_dim = len(phase5_result["feature_names"])

        model_config = LocalModelConfig(
            lstm=LSTMEncoderConfig(
                input_dim=ts_input_dim, hidden_dim=cfg.get("modalities.timeseries.hidden_dim", 128),
                num_layers=cfg.get("modalities.timeseries.num_layers", 2),
                dropout=cfg.get("modalities.timeseries.dropout", 0.2),
            ),
            autoencoder=AutoencoderConfig(
                input_dim=tab_input_dim, hidden_dim=cfg.get("modalities.tabular.hidden_dim", 128),
                embedding_dim=cfg.get("modalities.tabular.embedding_dim", 64),
                dropout=cfg.get("modalities.tabular.dropout", 0.1),
            ),
            gnn_hidden_dim=cfg.get("gnn.hidden_dim", 128), gnn_n_layers=cfg.get("gnn.layers", 2),
            gnn_type=cfg.get("gnn.type", "graphsage"), gnn_dropout=cfg.get("gnn.dropout", 0.2),
            head_hidden_dim=cfg.get("multitask.head_hidden_dim", 64),
            head_dropout=cfg.get("multitask.head_dropout", 0.1),
        )

        checkpoint_dir = Path(cfg.get("run.checkpoint_dir", "checkpoints")) / (
            checkpoint_dir_name or ("phase11_local_splitaware" if train_patient_ids is not None else "phase11_local")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        global_model = LocalPatientModel(model_config)
        initial_state_dict = {k: v.clone() for k, v in global_model.state_dict().items()}
        initial_checkpoint_path = checkpoint_dir / "global_initial_weights.pt"

        result = _run_local_training_round(
            cfg, phase3_result, phase4_result, phase5_result, phase8_result,
            model_config, initial_state_dict, checkpoint_dir, logger, round_tag="PHASE11",
            train_patient_ids=train_patient_ids,
        )

    result["global_initial_checkpoint_path"] = str(initial_checkpoint_path)
    result["lstm_input_dim"] = ts_input_dim
    result["tabular_input_dim"] = tab_input_dim
    return result


def _try_reuse_existing_local_checkpoints(checkpoint_dir: Path, n_patients_by_hid: dict, logger) -> "tuple[dict, dict] | None":
    """Attempts to load an EXISTING prior local-training round's
    per-hospital checkpoints for reuse (Phase 12's round 1, when
    `federated.reuse_existing_local_checkpoints_for_round_1` is true).
    Returns (state_dicts, weights) if at least one usable checkpoint
    was found, or None if the directory doesn't exist / has none at
    all -- the caller falls back to a fresh round in that case, never
    crashes.

    Each loaded checkpoint is independently re-checked for NaN/Inf --
    a prior run (like the user's own real Phase 11 run, which predates
    the 2026-08-31 NaN-safety fix) could in principle have saved a
    corrupted checkpoint before that fix existed. A bad checkpoint is
    EXCLUDED from aggregation and logged, never silently included.
    """
    if not checkpoint_dir.exists():
        return None
    checkpoint_paths = sorted(checkpoint_dir.glob("hospital_*_local_weights.pt"))
    if not checkpoint_paths:
        return None

    import torch

    state_dicts: dict = {}
    weights: dict = {}
    n_excluded_nan = 0
    n_excluded_unknown = 0
    for path in checkpoint_paths:
        hid = path.name[len("hospital_"):-len("_local_weights.pt")]
        state = torch.load(path, weights_only=True)
        if any(not torch.isfinite(v).all() for v in state.values()):
            n_excluded_nan += 1
            logger.warning(f"[PHASE12] excluding reused checkpoint for hospital {hid}: contains NaN/Inf.")
            continue
        if hid not in n_patients_by_hid:
            n_excluded_unknown += 1
            logger.warning(
                f"[PHASE12] excluding reused checkpoint for hospital {hid}: not found in Phase 8's "
                f"current graph files (stale checkpoint from a different cohort/config?)."
            )
            continue
        state_dicts[hid] = state
        weights[hid] = n_patients_by_hid[hid]

    if n_excluded_nan or n_excluded_unknown:
        logger.warning(
            f"[PHASE12] excluded {n_excluded_nan} reused checkpoint(s) for NaN/Inf and "
            f"{n_excluded_unknown} for not matching the current cohort."
        )
    if not state_dicts:
        return None
    return state_dicts, weights


def run_phase12_fedavg(
    cfg, phase3_result: dict, phase4_result: dict, phase5_result: dict, phase8_result: dict,
    train_patient_ids: "set | None" = None, checkpoint_dir_name: "str | None" = None,
) -> dict:
    """Phase 12 -- FedAvg: multi-round federated averaging.

    Each round: every hospital trains locally starting from the
    CURRENT global model (round 1: a fresh seeded init, or the user's
    already-completed real Phase 11 checkpoints when reuse is enabled
    -- see below; rounds 2+: the PREVIOUS round's aggregated result),
    then their updated weights are combined via
    src/training/fedavg.py's `fedavg_aggregate()` -- weighted by each
    hospital's own patient count (standard McMahan et al. 2017 FedAvg,
    NOT an unweighted mean -- a 7,058-patient hospital, Phase 8's real
    max, should influence the global model more than a 20-patient one,
    matching how a hypothetical centralized model on the pooled data
    would naturally weight more evidence). The aggregated result
    becomes the next round's starting point.

    SCOPE, decided explicitly rather than silently assumed: PLAIN
    FedAvg only (weighted parameter averaging). No differential
    privacy, no secure aggregation -- several of this project's own
    reference papers explore those, but they're a substantial separate
    research direction, not folded into this phase.

    Reuses `_run_local_training_round()` (extracted from Phase 11
    2026-09-01 for exactly this purpose) for every FRESH round, so
    Phase 12 trains with the identical, already real-full-scale-
    verified per-hospital mechanics Phase 11 itself uses -- same loss
    masking, same NaN/Inf safety, same never-pool-across-hospitals
    boundary, same per-round training_manifest.json persistence.

    `federated.n_rounds` and
    `federated.reuse_existing_local_checkpoints_for_round_1` are both
    config-driven, DECIDED WITH THE USER (2026-09-01) rather than
    silently defaulted: this run uses 3 total rounds, and round 1
    reuses the user's own already-completed, already quality-checked
    real Phase 11 run (`checkpoints/phase11_local/`) instead of
    re-training from scratch -- confirmed 2026-08-31 to be healthy
    (zero NaN/Inf across all 207 real hospitals, a tight 1.05x
    weight-delta spread). If reuse is requested but no usable
    checkpoints are found, round 1 falls back to fresh training
    automatically, with a clear warning -- never a crash.

    Each hospital's per-patient-count weight for aggregation is
    derived directly from Phase 8's own graph files
    (`len(patient_ids)` per hospital) -- NOT from Phase 11's
    training_manifest.json, which may not exist (the user's own real
    run predates that fix). This means Phase 12 does not need
    Phase 11 to have run in this same invocation at all: round 1 reuse
    reads directly from whatever's already on disk under
    `checkpoints/phase11_local/`, exactly the user's real situation.

    `train_patient_ids`/`checkpoint_dir_name`, added 2026-09-11: see
    `_run_local_training_round()`'s docstring for what split-aware
    training changes at the per-round level. Two additional things
    change HERE, specifically for FedAvg's own correctness:

    1. **Round-1 reuse is redirected, never cross-contaminated.** When
       `train_patient_ids` is given, round-1 reuse (if enabled) looks
       under `checkpoints/phase11_local_splitaware/` instead of
       `checkpoints/phase11_local/` -- reusing the ORIGINAL, full-
       cohort-trained Phase 11 checkpoints for a split-aware Phase 12
       run would silently reintroduce exactly the leakage this whole
       effort exists to remove (those checkpoints were fit on val/test
       patients too). A split-aware round 1 can only ever reuse a
       PRIOR split-aware Phase 11 run, never the original one.
    2. **Aggregation weights use each hospital's TRAIN-only patient
       count, not its total population**, when split-aware --
       `hospital_results[hid]["n_train_patients"]` (added to
       `_run_local_training_round()`'s per-hospital output) rather than
       `n_patients`. FedAvg's weighting is meant to reflect how much
       evidence actually went into a hospital's update; under split-
       aware training that's the train-split count, not the hospital's
       full population -- using the full count would overweight
       hospitals whose train fraction happens to be small relative to
       their total size.

    `checkpoint_dir_name` defaults to `"phase12_fedavg"` when
    `train_patient_ids is None` (byte-for-byte the original location)
    but to `"phase12_fedavg_splitaware"` when a split IS given -- same
    safe-by-default reasoning as Phase 11's own `checkpoint_dir_name`.
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE12_FEDAVG"):
        import torch

        from src.modalities.tabular.autoencoder import AutoencoderConfig
        from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig
        from src.training.fedavg import fedavg_aggregate
        from src.training.local_model import LocalModelConfig, LocalPatientModel

        algorithm = cfg.get("federated.algorithm", "fedavg")
        if algorithm == "fedproto":
            raise NotImplementedError(
                "federated.algorithm='fedproto' is not implemented yet -- the config key and this "
                "dispatch point exist specifically so FedProto can be added later as a new branch "
                "here (alongside fedavg_aggregate()) without a pipeline rewrite, per the ablation-"
                "framework decisions confirmed 2026-09-10. Set federated.algorithm='fedavg' for now."
            )
        if algorithm != "fedavg":
            raise ValueError(f"federated.algorithm={algorithm!r} not recognized -- must be 'fedavg' or 'fedproto'.")

        seed = cfg.get("run.seed", 42)
        n_rounds = cfg.get("federated.n_rounds", 1)
        reuse_round1 = cfg.get("federated.reuse_existing_local_checkpoints_for_round_1", False)
        if not isinstance(n_rounds, int) or n_rounds < 1:
            raise ValueError(f"federated.n_rounds must be an integer >= 1, got {n_rounds!r}.")

        ts_input_dim = len(phase4_result["feature_names"])
        tab_input_dim = len(phase5_result["feature_names"])
        model_config = LocalModelConfig(
            lstm=LSTMEncoderConfig(
                input_dim=ts_input_dim, hidden_dim=cfg.get("modalities.timeseries.hidden_dim", 128),
                num_layers=cfg.get("modalities.timeseries.num_layers", 2),
                dropout=cfg.get("modalities.timeseries.dropout", 0.2),
            ),
            autoencoder=AutoencoderConfig(
                input_dim=tab_input_dim, hidden_dim=cfg.get("modalities.tabular.hidden_dim", 128),
                embedding_dim=cfg.get("modalities.tabular.embedding_dim", 64),
                dropout=cfg.get("modalities.tabular.dropout", 0.1),
            ),
            gnn_hidden_dim=cfg.get("gnn.hidden_dim", 128), gnn_n_layers=cfg.get("gnn.layers", 2),
            gnn_type=cfg.get("gnn.type", "graphsage"), gnn_dropout=cfg.get("gnn.dropout", 0.2),
            head_hidden_dim=cfg.get("multitask.head_hidden_dim", 64),
            head_dropout=cfg.get("multitask.head_dropout", 0.1),
        )

        base_checkpoint_dir = Path(cfg.get("run.checkpoint_dir", "checkpoints")) / (
            checkpoint_dir_name or ("phase12_fedavg_splitaware" if train_patient_ids is not None else "phase12_fedavg")
        )
        base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        phase11_checkpoint_dir = Path(cfg.get("run.checkpoint_dir", "checkpoints")) / (
            "phase11_local_splitaware" if train_patient_ids is not None else "phase11_local"
        )

        # Hospital -> weight for FedAvg aggregation. Derived from Phase
        # 8's graph files directly -- NOT from a training_manifest.json,
        # which may not exist (see docstring). When split-aware, this
        # is each hospital's TRAIN-ONLY patient count (not its full
        # population) -- see docstring point 2.
        n_patients_by_hid = {}
        for hid, path in phase8_result["hospital_graph_paths"].items():
            graph_data = np.load(path)
            if train_patient_ids is not None:
                n_patients_by_hid[hid] = int(sum(1 for pid in graph_data["patient_ids"] if int(pid) in train_patient_ids))
            else:
                n_patients_by_hid[hid] = len(graph_data["patient_ids"])

        current_state_dict = None
        round_summaries = []

        for round_idx in range(n_rounds):
            round_num = round_idx + 1
            reused = None
            if round_idx == 0 and reuse_round1:
                reused = _try_reuse_existing_local_checkpoints(phase11_checkpoint_dir, n_patients_by_hid, logger)

            with PhaseTimer(logger, f"PHASE12_ROUND_{round_num}"):
                if reused is not None:
                    state_dicts, weights = reused
                    round_source = "reused_phase11_checkpoints"
                    round_checkpoint_dir_note = str(phase11_checkpoint_dir)
                    n_hospitals_aggregated = len(state_dicts)
                    logger.info(
                        f"[PHASE12] round {round_num}: reusing {n_hospitals_aggregated} existing "
                        f"checkpoints from {phase11_checkpoint_dir} instead of training fresh."
                    )
                else:
                    if round_idx == 0:
                        if reuse_round1:
                            logger.warning(
                                f"[PHASE12] round {round_num}: federated.reuse_existing_local_checkpoints_"
                                f"for_round_1 is true but no usable checkpoints were found at "
                                f"{phase11_checkpoint_dir} -- running round {round_num} fresh instead."
                            )
                        torch.manual_seed(seed)
                        fresh_model = LocalPatientModel(model_config)
                        current_state_dict = {k: v.clone() for k, v in fresh_model.state_dict().items()}

                    round_checkpoint_dir = base_checkpoint_dir / f"round_{round_num}"
                    round_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    round_result = _run_local_training_round(
                        cfg, phase3_result, phase4_result, phase5_result, phase8_result,
                        model_config, current_state_dict, round_checkpoint_dir, logger,
                        round_tag=f"PHASE12_ROUND_{round_num}", train_patient_ids=train_patient_ids,
                    )
                    state_dicts = {
                        hid: torch.load(r["checkpoint_path"], weights_only=True)
                        for hid, r in round_result["hospital_results"].items()
                    }
                    weights = {
                        hid: r.get("n_train_patients", r["n_patients"])
                        for hid, r in round_result["hospital_results"].items()
                    }
                    round_source = "fresh_local_training"
                    round_checkpoint_dir_note = str(round_checkpoint_dir)
                    n_hospitals_aggregated = round_result["n_hospitals_trained"]

                aggregated_state_dict = fedavg_aggregate(state_dicts, weights)
                global_model_path = base_checkpoint_dir / f"global_model_round_{round_num}.pt"
                torch.save(aggregated_state_dict, global_model_path)
                current_state_dict = aggregated_state_dict

                logger.info(
                    f"[PHASE12] round {round_num}/{n_rounds} complete: aggregated {n_hospitals_aggregated} "
                    f"hospitals ({round_source}) -> {global_model_path}"
                )

            round_summaries.append({
                "round": round_num,
                "source": round_source,
                "n_hospitals_aggregated": n_hospitals_aggregated,
                "checkpoint_dir": round_checkpoint_dir_note,
                "global_model_path": str(global_model_path),
            })

        fedavg_manifest_path = base_checkpoint_dir / "fedavg_manifest.json"
        with open(fedavg_manifest_path, "w") as f:
            json.dump(
                {"n_rounds": n_rounds, "rounds": round_summaries,
                 "final_global_model_path": round_summaries[-1]["global_model_path"]},
                f, indent=2,
            )

    return {
        "n_rounds": n_rounds,
        "final_global_model_path": round_summaries[-1]["global_model_path"],
        "fedavg_manifest_path": str(fedavg_manifest_path),
        "round_summaries": round_summaries,
        "checkpoint_dir": str(base_checkpoint_dir),
    }


def _run_centralized_sequential_training(
    cfg, phase3_result: dict, phase4_result: dict, phase5_result: dict, phase8_result: dict,
    model_config, total_epochs: int, seed: int, checkpoint_dir: Path, logger,
    train_patient_ids: "set | None" = None,
) -> dict:
    """Phase 13's "centralized" baseline: ONE continuously-updated
    model -- never reset, never split into per-hospital copies -- that
    cycles through every hospital's own intra-hospital graph in a
    reshuffled order each epoch.

    DECIDED WITH THE USER (2026-09-03): sequential/continuous, NOT
    true cross-hospital data pooling. Every graph stays strictly
    intra-hospital (Phase 8's locked design, the project's core
    privacy boundary) -- this baseline isolates specifically the
    effect of federated ROUNDS/RESETS/AVERAGING, holding the
    privacy-preserving architecture itself constant, rather than
    measuring "what if we were allowed to pool raw data" (which would
    need a genuinely different, cross-hospital graph -- a real
    architectural change this project has never made and wasn't asked
    to make here). See run_phase13_baselines()'s own docstring for the
    "reset frequency" framing this and the other two baselines
    together are meant to isolate.

    EACH HOSPITAL GETS EXACTLY ONE GRADIENT STEP PER EPOCH (matching
    every other phase's "no mini-batching within a hospital" rule --
    the GNN needs each hospital's whole graph in one forward call) --
    `total_epochs` therefore gives every hospital the SAME cumulative
    gradient-step budget as the local and federated baselines (see
    run_phase13_baselines()'s docstring for why matching budgets
    matters for a fair comparison).

    ONE optimizer instance persists for the ENTIRE run, never reset
    between hospitals or epochs -- the natural, correct reading of
    "continuous, no resets": Adam's momentum/variance state carries
    over exactly as it would for any ordinary SGD/Adam loop being fed
    a sequence of different mini-batches (here, each hospital's own
    full graph is one such "batch").

    Hospital visit order is RESHUFFLED every epoch (seeded via
    `run.seed`, for reproducibility) -- a fixed order would bias the
    model toward whichever hospital always comes last in that epoch,
    standard practice for any SGD-style training loop.

    Each hospital's raw tensors/graph are RE-LOADED on every visit
    (via `_load_hospital_batch()`) rather than pre-loaded once and
    held in memory for the whole run: Phase 4's LSTM already needed
    real-data batching to avoid OOM holding the full 199,108-patient
    cohort at once, and pre-loading per-hospital COPIES for all ~207
    hospitals simultaneously would roughly double that memory
    footprint for no real benefit -- slicing/graph-loading itself is
    cheap numpy/npz work compared to the actual forward/backward pass
    through four chained neural network components.

    A non-finite (NaN/Inf) loss on any single hospital-visit skips
    just that one optimizer step (model NOT updated from it) and
    training continues -- unlike Phase 11/12, where a bad hospital is
    excluded from that round's checkpoint/aggregation entirely, there
    is no equivalent "exclude from this round" concept in a single
    continuous run; skipping the one bad step and moving on is the
    direct analogue.

    `train_patient_ids`, added 2026-09-11: same train-only-subgraph
    technique as `_run_local_training_round()`'s own split-aware path
    (see that function's docstring for the full design rationale and
    the correction that led to it) -- each hospital's per-epoch
    gradient step trains on a subgraph containing ONLY that hospital's
    train-split patients, so a non-train patient's raw data is never
    present in any tensor the (continuously-updated, never-reset)
    model's forward/backward pass touches. After ALL epochs complete,
    one additional frozen (`eval()`, `no_grad()`) forward pass per
    hospital, over its FULL graph, produces predictions for val/test
    patients against the final, fixed model.
    """
    import torch

    from src.training.local_model import LocalPatientModel
    from src.training.losses import compute_multitask_loss

    sepsis_cohort = cfg.get("multitask.sepsis_cohort")
    mortality_weight = cfg.get("multitask.mortality_weight", 1.0)
    sepsis_weight = cfg.get("multitask.sepsis_weight", 1.0)
    loss_type = cfg.get("multitask.loss", "weighted_bce")
    learning_rate = cfg.get("training.learning_rate", 0.001)
    weight_decay = cfg.get("training.weight_decay", 0.0001)
    optimizer_name = cfg.get("training.optimizer", "adam")
    reconstruction_weight = cfg.get("training.reconstruction_weight", 0.0)
    max_hospitals = cfg.get("training.max_hospitals", None)

    if optimizer_name not in ("adam", "sgd"):
        raise ValueError(f"training.optimizer={optimizer_name!r} not recognized -- must be 'adam' or 'sgd'.")

    ts_patient_ids = [int(pid) for pid in phase4_result["patient_ids"]]
    ts_row_of_patient = {pid: i for i, pid in enumerate(ts_patient_ids)}
    ts_array = phase4_result["normalized_array"]
    ts_seq_length = phase4_result["sequence_length"]

    tab_patient_ids = [int(pid) for pid in phase5_result["patient_ids"]]
    tab_row_of_patient = {pid: i for i, pid in enumerate(tab_patient_ids)}
    tab_array = phase5_result["normalized_array"]

    hospital_graph_paths = phase8_result["hospital_graph_paths"]
    hospital_ids = sorted(hospital_graph_paths.keys())
    if max_hospitals is not None:
        hospital_ids = hospital_ids[:max_hospitals]

    torch.manual_seed(seed)
    model = LocalPatientModel(model_config)
    initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial_state_dict, checkpoint_dir / "global_initial_weights.pt")

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Determine the non-empty hospital list ONCE (doesn't change across
    # epochs) via a cheap emptiness check -- see docstring for why full
    # batches are NOT pre-loaded here.
    non_empty_hospital_ids = []
    n_skipped_empty = 0
    for hid in hospital_ids:
        graph_data = np.load(hospital_graph_paths[hid])
        if len(graph_data["patient_ids"]) == 0:
            n_skipped_empty += 1
        else:
            non_empty_hospital_ids.append(hid)

    rng = random.Random(seed)
    model.train()
    epoch_history: list[dict] = []
    n_non_finite_step_events = 0
    n_no_label_step_events = 0

    for epoch in range(total_epochs):
        visit_order = list(non_empty_hospital_ids)
        rng.shuffle(visit_order)
        epoch_losses: list[float] = []
        for hid in visit_order:
            batch = _load_hospital_batch(
                hid, hospital_graph_paths, ts_row_of_patient, ts_array, ts_seq_length,
                tab_row_of_patient, tab_array, phase3_result, sepsis_cohort,
            )
            model_ts_x, model_ts_len, model_tab_x = batch["timeseries_x"], batch["timeseries_lengths"], batch["tabular_x"]
            model_edge_index, model_edge_weight = batch["edge_index"], batch["edge_weight"]
            model_mortality, model_sepsis = batch["mortality_labels"], batch["sepsis_labels"]
            if train_patient_ids is not None:
                train_row_mask = torch.tensor([pid in train_patient_ids for pid in batch["patient_ids"]])
                if int(train_row_mask.sum().item()) == 0:
                    continue  # nothing to train on for this hospital this epoch
                model_edge_index, model_edge_weight, train_indices = _build_train_only_subgraph(
                    batch["edge_index"], batch["edge_weight"], train_row_mask,
                )
                model_ts_x = batch["timeseries_x"][train_indices]
                model_ts_len = batch["timeseries_lengths"][train_indices]
                model_tab_x = batch["tabular_x"][train_indices]
                model_mortality = batch["mortality_labels"][train_indices]
                model_sepsis = batch["sepsis_labels"][train_indices]

            optimizer.zero_grad()
            out = model(model_ts_x, model_ts_len, model_tab_x, model_edge_index, model_edge_weight)
            total_loss, _components, _counts = compute_multitask_loss(
                out["mortality_logits"], out["sepsis_logits"], model_mortality, model_sepsis,
                mortality_weight, sepsis_weight, loss_type,
                reconstruction=out["tabular_reconstruction"], tabular_target=model_tab_x,
                reconstruction_weight=reconstruction_weight,
            )
            if total_loss is None:
                n_no_label_step_events += 1
                continue
            if not torch.isfinite(total_loss):
                n_non_finite_step_events += 1
                logger.warning(
                    f"[PHASE13_CENTRALIZED] epoch {epoch + 1}, hospital {hid}: non-finite loss -- "
                    f"this ONE step skipped (model not updated from it), training continues."
                )
                continue
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(total_loss.item())

        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
        epoch_history.append({
            "epoch": epoch + 1, "mean_loss": mean_epoch_loss, "n_hospitals_contributed": len(epoch_losses),
        })
        if (epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs:
            logger.info(
                f"[PHASE13_CENTRALIZED] epoch {epoch + 1}/{total_epochs}: mean_loss="
                f"{mean_epoch_loss if mean_epoch_loss is None else round(mean_epoch_loss, 4)}"
            )

    final_model_path = checkpoint_dir / "centralized_sequential_final.pt"
    torch.save(model.state_dict(), final_model_path)

    predictions_by_hospital: dict = {}
    if train_patient_ids is not None:
        # One frozen forward pass per hospital over its FULL graph
        # (train+val+test) against the final, fixed model -- read-only
        # inference, exactly "passing through frozen components."
        model.eval()
        with torch.no_grad():
            for hid in non_empty_hospital_ids:
                batch = _load_hospital_batch(
                    hid, hospital_graph_paths, ts_row_of_patient, ts_array, ts_seq_length,
                    tab_row_of_patient, tab_array, phase3_result, sepsis_cohort,
                )
                eval_out = model(
                    batch["timeseries_x"], batch["timeseries_lengths"], batch["tabular_x"],
                    batch["edge_index"], batch["edge_weight"],
                )
                is_train_row = ["train" if pid in train_patient_ids else "not_train" for pid in batch["patient_ids"]]
                predictions_by_hospital[hid] = {
                    "patient_ids": batch["patient_ids"],
                    "mortality_logits": eval_out["mortality_logits"].numpy().tolist(),
                    "sepsis_logits": eval_out["sepsis_logits"].numpy().tolist(),
                    "mortality_labels": batch["mortality_labels"].numpy().tolist(),
                    "sepsis_labels": batch["sepsis_labels"].numpy().tolist(),
                    "is_train_row": is_train_row,
                }

    manifest = {
        "total_epochs": total_epochs,
        "n_hospitals": len(non_empty_hospital_ids),
        "n_skipped_empty": n_skipped_empty,
        "n_non_finite_step_events": n_non_finite_step_events,
        "n_no_label_step_events": n_no_label_step_events,
        "loss_type": loss_type,
        "sepsis_cohort": sepsis_cohort,
        "reconstruction_weight": reconstruction_weight,
        "split_aware": train_patient_ids is not None,
        "epoch_history": epoch_history,
        "final_model_path": str(final_model_path),
    }
    manifest_path = checkpoint_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"[PHASE13_CENTRALIZED] complete: {total_epochs} epochs, {len(non_empty_hospital_ids)} hospitals, "
        f"{n_skipped_empty} empty, {n_non_finite_step_events} non-finite step(s) skipped -> {final_model_path}"
    )

    return {
        "final_model_path": str(final_model_path),
        "manifest_path": str(manifest_path),
        "n_hospitals": len(non_empty_hospital_ids),
        "n_skipped_empty": n_skipped_empty,
        "n_non_finite_step_events": n_non_finite_step_events,
        "epoch_history": epoch_history,
        "predictions_by_hospital": predictions_by_hospital,
    }


def run_phase13_baselines(
    cfg, phase3_result: dict, phase4_result: dict, phase5_result: dict, phase8_result: dict,
    train_patient_ids: "set | None" = None, checkpoint_dir_name: "str | None" = None,
) -> dict:
    """Phase 13 -- Baselines (local / centralized / federated).

    Produces the two comparison points Phase 12 doesn't already give
    us. The "reset frequency" spectrum these three baselines exist to
    compare (confirmed with the user 2026-09-03):

      LOCAL:        each hospital trains independently, NEVER synced
                     with any other hospital -- effectively infinite
                     resets / no sync at all.
      FEDERATED:     hospitals sync to a shared global model every
                     `training.local_epochs` epochs -- `federated.
                     n_rounds - 1` resets (Phase 11 + Phase 12, already
                     built and real-verified).
      CENTRALIZED:   ONE continuously-updated model, never reset --
                     zero resets, fully continuous (this phase, see
                     `_run_centralized_sequential_training()`).

    Federated needs no new work here: it's Phase 12's already-completed
    real result, referenced by reading `checkpoints/phase12_fedavg/
    fedavg_manifest.json` (not re-run) -- same "reuse already-completed
    real work" precedent as Phase 12's own round-1 checkpoint reuse.

    SCOPE, decided explicitly: this phase TRAINS the local and
    centralized baselines. It does NOT compute AUROC/AUPRC/accuracy or
    do any cross-baseline comparison -- that is Phase 14 (Evaluation)'s
    job, per the tracker's own separation of these two phases, and
    also requires a train/val/test split that doesn't exist anywhere
    in this project yet (explicitly deferred since Phase 11).

    FAIR EPOCH BUDGET, decided with the user (2026-09-03): both new
    baselines get the SAME TOTAL per-hospital epoch budget federated
    hospitals see cumulatively across all rounds --
    `training.local_epochs * federated.n_rounds` -- derived
    automatically from the existing config rather than a new,
    independently-settable value that could silently drift out of
    sync if `federated.n_rounds` is ever changed later. A local
    baseline trained for only one round's worth of epochs would look
    artificially worse than federated purely from getting less total
    training, not from federation itself helping.

    CENTRALIZED BASELINE DEFINITION, decided with the user
    (2026-09-03): sequential/continuous, NOT true cross-hospital data
    pooling -- see `_run_centralized_sequential_training()`'s own
    docstring for the full reasoning (keeps Phase 8's locked
    intra-hospital-only graph design, and this project's core privacy
    boundary, completely untouched).

    `train_patient_ids`/`checkpoint_dir_name`, added 2026-09-11: both
    new baselines are trained split-aware exactly the way their
    underlying helpers already support (`_run_local_training_round()`
    and `_run_centralized_sequential_training()`'s own
    `train_patient_ids` parameters -- train-only subgraphs, frozen
    full-graph inference for val/test predictions). The federated
    reference is ALSO redirected: when split-aware, it reads
    `checkpoints/phase12_fedavg_splitaware/fedavg_manifest.json`
    instead of the original `phase12_fedavg/fedavg_manifest.json` --
    comparing a split-aware local/centralized baseline against the
    ORIGINAL, full-cohort-trained federated result would not be an
    apples-to-apples "reset frequency" comparison (see this function's
    "reset frequency" framing above: all three points are meant to
    hold everything ELSE constant while varying only sync frequency).
    `checkpoint_dir_name` defaults to `"phase13_baselines"` when
    `train_patient_ids is None` (byte-for-byte the original location)
    but to `"phase13_baselines_splitaware"` when a split IS given --
    same safe-by-default reasoning as Phase 11/12's own
    `checkpoint_dir_name`.
    """
    logger = setup_logging(log_dir=cfg.get("run.log_dir"))

    with PhaseTimer(logger, "PHASE13_BASELINES"):
        import torch

        from src.modalities.tabular.autoencoder import AutoencoderConfig
        from src.modalities.timeseries.lstm_encoder import LSTMEncoderConfig
        from src.training.local_model import LocalModelConfig, LocalPatientModel

        seed = cfg.get("run.seed", 42)
        local_epochs_per_round = cfg.get("training.local_epochs", 5)
        n_rounds = cfg.get("federated.n_rounds", 1)
        total_epochs = local_epochs_per_round * n_rounds

        ts_input_dim = len(phase4_result["feature_names"])
        tab_input_dim = len(phase5_result["feature_names"])
        model_config = LocalModelConfig(
            lstm=LSTMEncoderConfig(
                input_dim=ts_input_dim, hidden_dim=cfg.get("modalities.timeseries.hidden_dim", 128),
                num_layers=cfg.get("modalities.timeseries.num_layers", 2),
                dropout=cfg.get("modalities.timeseries.dropout", 0.2),
            ),
            autoencoder=AutoencoderConfig(
                input_dim=tab_input_dim, hidden_dim=cfg.get("modalities.tabular.hidden_dim", 128),
                embedding_dim=cfg.get("modalities.tabular.embedding_dim", 64),
                dropout=cfg.get("modalities.tabular.dropout", 0.1),
            ),
            gnn_hidden_dim=cfg.get("gnn.hidden_dim", 128), gnn_n_layers=cfg.get("gnn.layers", 2),
            gnn_type=cfg.get("gnn.type", "graphsage"), gnn_dropout=cfg.get("gnn.dropout", 0.2),
            head_hidden_dim=cfg.get("multitask.head_hidden_dim", 64),
            head_dropout=cfg.get("multitask.head_dropout", 0.1),
        )

        base_checkpoint_dir = Path(cfg.get("run.checkpoint_dir", "checkpoints")) / (
            checkpoint_dir_name or ("phase13_baselines_splitaware" if train_patient_ids is not None else "phase13_baselines")
        )
        base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[PHASE13] fair epoch budget = training.local_epochs ({local_epochs_per_round}) x "
                    f"federated.n_rounds ({n_rounds}) = {total_epochs} total epochs/hospital")

        # --- Local-only baseline: each hospital trains independently
        # for the full total_epochs budget, no reset, no averaging. ---
        local_dir = base_checkpoint_dir / "local_only"
        local_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        local_init_model = LocalPatientModel(model_config)
        local_init_state = {k: v.clone() for k, v in local_init_model.state_dict().items()}
        local_result = _run_local_training_round(
            cfg, phase3_result, phase4_result, phase5_result, phase8_result,
            model_config, local_init_state, local_dir, logger, round_tag="PHASE13_LOCAL",
            local_epochs_override=total_epochs, train_patient_ids=train_patient_ids,
        )

        # --- Centralized-sequential baseline. ---
        centralized_dir = base_checkpoint_dir / "centralized_sequential"
        centralized_dir.mkdir(parents=True, exist_ok=True)
        centralized_result = _run_centralized_sequential_training(
            cfg, phase3_result, phase4_result, phase5_result, phase8_result,
            model_config, total_epochs, seed, centralized_dir, logger,
            train_patient_ids=train_patient_ids,
        )

        # --- Federated baseline: reference Phase 12's already-completed
        # real result, never re-run. Redirected to the split-aware
        # FedAvg manifest when split-aware -- see docstring. ---
        fedavg_dir_name = "phase12_fedavg_splitaware" if train_patient_ids is not None else "phase12_fedavg"
        fedavg_manifest_path = Path(cfg.get("run.checkpoint_dir", "checkpoints")) / fedavg_dir_name / "fedavg_manifest.json"
        federated_final_model_path = None
        if fedavg_manifest_path.exists():
            with open(fedavg_manifest_path) as f:
                fedavg_manifest = json.load(f)
            federated_final_model_path = fedavg_manifest.get("final_global_model_path")
            logger.info(f"[PHASE13] federated baseline reference: {federated_final_model_path} (Phase 12, not re-run)")
        else:
            logger.warning(
                f"[PHASE13] no Phase 12 FedAvg manifest found at {fedavg_manifest_path} -- the federated "
                f"baseline reference will be unavailable for this run. Run Phase 12 first for a complete "
                f"3-way comparison; Phase 13 itself still completes (local + centralized are independent)."
            )

        summary = {
            "total_epochs": total_epochs,
            "local_epochs_per_round": local_epochs_per_round,
            "n_rounds": n_rounds,
            "local_only": {
                "checkpoint_dir": str(local_dir), "manifest_path": local_result["manifest_path"],
                "n_hospitals_trained": local_result["n_hospitals_trained"],
            },
            "centralized_sequential": {
                "checkpoint_dir": str(centralized_dir), "manifest_path": centralized_result["manifest_path"],
                "final_model_path": centralized_result["final_model_path"],
                "n_hospitals": centralized_result["n_hospitals"],
            },
            "federated_reference": {
                "manifest_path": str(fedavg_manifest_path) if fedavg_manifest_path.exists() else None,
                "final_model_path": federated_final_model_path,
            },
        }
        baselines_manifest_path = base_checkpoint_dir / "baselines_manifest.json"
        with open(baselines_manifest_path, "w") as f:
            json.dump(summary, f, indent=2)

    return {
        "checkpoint_dir": str(base_checkpoint_dir),
        "manifest_path": str(baselines_manifest_path),
        "total_epochs": total_epochs,
        "local_only": local_result,
        "centralized_sequential": centralized_result,
        "federated_reference_model_path": federated_final_model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a mode config, e.g. configs/debug.yaml")
    parser.add_argument(
        "--phase", default=5, type=int,
        help="Run phases 1..N in sequence (1-5, 7, 8, 9, 10, 11, 12, 13 implemented; 6 is "
             "intentionally skipped -- see IMPLEMENTATION_TRACKER.md).",
    )
    parser.add_argument(
        "--start-phase", default=1, type=int,
        help="Skip phases before this one, reloading their outputs from run.output_dir instead of "
             "recomputing them -- requires those phases to have already been run and saved for this "
             "exact output_dir. E.g. --start-phase 8 --phase 9 reuses Phases 1-7's saved outputs and "
             "only actually runs 8-9. Defaults to 1 (full run, current behavior unchanged). See "
             "docs/phase_resumption.md.",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.phase < 1 or (args.phase > 5 and args.phase not in (7, 8, 9, 10, 11, 12, 13)):
        print(
            f"Phase '{args.phase}' is out of the implemented range (1-5, 7, 8, 9, 10, 11, 12, 13; 6 "
            "is intentionally skipped). See IMPLEMENTATION_TRACKER.md for current status.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.start_phase > args.phase:
        print(f"--start-phase ({args.start_phase}) cannot be greater than --phase ({args.phase}).", file=sys.stderr)
        sys.exit(1)
    if args.start_phase not in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13):
        print(
            f"--start-phase '{args.start_phase}' is not a valid resumption point (1, 2, 3, 4, 5, 7, "
            "8, 9, 10, 11, 12, 13; 6 is intentionally skipped).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        def will_run(n: int) -> bool:
            return args.start_phase <= n <= args.phase

        if args.start_phase > 2:
            # Must happen before anything else -- guards against
            # silently resuming into a cohort built by a DIFFERENT
            # config that happens to share the same run.output_dir
            # (e.g. debug.yaml vs full.yaml, neither of which overrides
            # it). See validate_run_identity()'s docstring.
            validate_run_identity(cfg)

        # A phaseN_result variable needs a value (via run OR load) if
        # phase N itself will actually run, OR if some downstream phase
        # that WILL run directly consumes it as an input -- loading a
        # phase's result is NOT the same as needing its dependencies,
        # so a phase being skipped-and-loaded contributes nothing to
        # any OTHER phase's dependency requirements. Direct-consumer
        # mapping here matches each run_phaseN() function's actual
        # parameter list exactly (see IMPLEMENTATION_TRACKER.md/
        # docs/phase_resumption.md for the full dependency graph).
        need_phase3_result = will_run(3) or any(will_run(m) for m in (4, 5, 7, 10, 11, 12, 13))
        # shard_manifests (part of phase3_result) is ONLY ever read by
        # Phase 4 -- Phase 5 and Phase 7 never touch it. Skipping its
        # reconstruction when Phase 4 won't run isn't just faster: the
        # underlying cache-freshness check needs the ORIGINAL raw
        # table to still exist on disk even on a cache hit, so this is
        # what lets --start-phase 7/8/9 work with the raw eICU data no
        # longer present at all, not merely quickly.
        need_shard_manifests = will_run(4)
        need_phase1_summary = (
            will_run(1) or any(will_run(m) for m in (2, 3, 5))
            # load_phase3_result() itself needs phase1_summary internally,
            # but only when it also needs to rebuild shard_manifests.
            or (need_phase3_result and not will_run(3) and need_shard_manifests)
        )
        need_partition_result = will_run(2) or any(will_run(m) for m in (3, 4, 5, 8))
        need_patient_df = will_run(3)  # only run_phase3 itself uses this; load_phase3_result doesn't
        need_phase4_result = will_run(4) or will_run(7) or will_run(11) or will_run(12) or will_run(13)
        need_phase5_result = will_run(5) or will_run(7) or will_run(11) or will_run(12) or will_run(13)
        need_phase7_result = will_run(7) or any(will_run(m) for m in (8, 9))
        need_phase8_result = will_run(8) or will_run(9) or will_run(11) or will_run(12) or will_run(13)
        # Phase 10 consumes Phase 9's OUTPUT (gnn_embeddings_path) directly --
        # it does NOT need phase7_result/phase8_result itself (those are
        # Phase 9's own inputs, already accounted for above only via
        # need_phase7_result/need_phase8_result's own will_run(9) term).
        need_phase9_result = will_run(9) or will_run(10)
        # Phase 11, 12, and 13 all need phase4_result/phase5_result/
        # phase8_result directly (see the flags above) but NOT
        # phase7_result or phase9_result -- all three recompute fusion
        # in-memory themselves and train their own fresh GNN/heads
        # rather than reusing Phase 9/10's forward-pass-only instances
        # (see run_phase11_local_training()'s docstring). Phase 12 and
        # 13 also do NOT need phase11_result/phase12_result as in-memory
        # values -- both read whatever's already on disk from a prior
        # phase's checkpoints/manifest directly (Phase 12's round-1
        # reuse; Phase 13's federated-baseline reference), so both work
        # standalone via --start-phase without the earlier phase having
        # run in the same invocation at all.

        phase1_summary = None
        if need_phase1_summary:
            phase1_summary = run_phase1_ingestion(cfg) if will_run(1) else load_phase1_summary(cfg)

        partition_result = patient_df = None
        if need_partition_result:
            if will_run(2):
                partition_result, patient_df, _hospital_df = run_phase2_hospital_partition(cfg, phase1_summary)
            else:
                partition_result = load_partition_result(cfg)
        if need_patient_df and patient_df is None:
            # partition_result was loaded (not run) above, but Phase 3
            # is about to actually run and needs patient_df itself --
            # cheap to re-read directly (patient.csv is small; unlike
            # the large sharded tables, it was never a bottleneck).
            patient_df = pd.read_csv(phase1_summary["table_paths"]["patient"])

        phase3_result = None
        if need_phase3_result:
            if will_run(3):
                phase3_result = run_phase3_cohort_construction(cfg, phase1_summary, partition_result, patient_df)
            else:
                phase3_result = load_phase3_result(cfg, phase1_summary, need_shard_manifests=need_shard_manifests)

        phase4_result = None
        if need_phase4_result:
            if will_run(4):
                phase4_result = run_phase4_timeseries_modality(cfg, phase3_result, partition_result)
            else:
                phase4_result = load_phase4_result(cfg)

        phase5_result = None
        if need_phase5_result:
            if will_run(5):
                phase5_result = run_phase5_tabular_modality(cfg, phase1_summary, phase3_result, partition_result)
            else:
                phase5_result = load_phase5_result(cfg)

        phase7_result = None
        if need_phase7_result:
            if will_run(7):  # Phase 6 (text) intentionally skipped -- see docs/text_modality_decision.md
                phase7_result = run_phase7_multimodal_fusion(cfg, phase3_result, phase4_result, phase5_result)
            else:
                phase7_result = load_phase7_result(cfg)

        phase8_result = None
        if need_phase8_result:
            if will_run(8):
                phase8_result = run_phase8_patient_graph_construction(cfg, phase7_result, partition_result)
            else:
                phase8_result = load_phase8_result(cfg)

        phase9_result = None
        if need_phase9_result:
            if will_run(9):
                phase9_result = run_phase9_gnn_forward_pass(cfg, phase7_result, phase8_result)
            else:
                phase9_result = load_phase9_result(cfg)

        if will_run(10):
            run_phase10_multitask_heads(cfg, phase9_result, phase3_result)

        if will_run(11):
            run_phase11_local_training(cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        if will_run(12):
            run_phase12_fedavg(cfg, phase3_result, phase4_result, phase5_result, phase8_result)

        if will_run(13):
            run_phase13_baselines(cfg, phase3_result, phase4_result, phase5_result, phase8_result)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
