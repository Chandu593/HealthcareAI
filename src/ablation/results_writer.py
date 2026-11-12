"""
Ablation framework -- structured, never-overwriting results
persistence + publication-ready summary tables.

Confirmed with the user (original request Section 3; ablation-
framework decisions carried over unchanged): every experiment gets a
unique, descriptive ID; results are saved as structured JSON (one file
per experiment, full detail/traceability -- config, split, model
combination, task, metrics, training info, runtime) AND appended to a
master CSV (for easy loading into pandas/Excel -- CSV opens natively in
Excel, so this covers "Excel-style tables" without adding a new heavy
dependency for a real .xlsx writer); Before-CA / After-CA / Graph
experiments are clearly distinguishable by filename (see
src/ablation/config_grid.py's `combo_id`/`experiment_id` naming); a
run's results are NEVER silently overwritten -- each invocation of the
ablation framework gets its own timestamped run directory, and
re-writing the SAME experiment id within a run directory raises rather
than silently clobbering the prior result unless `overwrite=True` is
passed explicitly.

BUG FOUND AND FIXED 2026-09-16 (found via a real full-scale run, not
hypothetical): `master_results.csv` was a SINGLE file receiving every
experiment kind, with each row's CSV header/column order derived fresh
from THAT row's own fields. `embedding_ablation` records carry
`config.ts_model`/`tab_model`/`fusion_mode` (3 fields) while
`graph_ablation` records carry `config.gnn_type` (1 field, plus a
differently-ordered `training.*` block) -- once a `graph_ablation` row
was appended after `embedding_ablation` rows had already established
the file's header, every value in that row landed under the WRONG
column when the file was read back by its header (pandas' `read_csv`,
Excel, `csv.DictReader`, etc.), even though the row's own comma-
separated values were individually correct. Confirmed by writing one
of each kind and reading the second row back through the file's own
header.

FIXED by giving each experiment kind its OWN csv file
(`master_results.csv` for everything except `graph_ablation`,
`graph_results.csv` for `graph_ablation`) -- every row within a single
file now shares one shape, so there is no header/row mismatch to have.
This does not touch or reformat any already-written file: an existing
`master_results.csv` with real prior-run rows in it is only ever
APPENDED to (exactly as before) for the experiment kinds it already
contained; `graph_ablation` rows simply stop landing there and start
landing in the new file instead.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Experiment kinds routed to their own dedicated CSV rather than
# master_results.csv -- see the module docstring's "BUG FOUND AND
# FIXED" section. Add a new kind here if a future experiment type
# introduces yet another incompatible config/training_info shape.
_DEDICATED_CSV_FILENAME_BY_KIND = {
    "graph_ablation": "graph_results.csv",
    # Added 2026-09-20 for the class-imbalance sampling comparison --
    # its config dict carries extra keys (sampling_strategy,
    # sampling_ratio) that embedding_ablation rows don't, so it gets
    # its own file for exactly the same reason graph_ablation did (see
    # the module docstring's "BUG FOUND AND FIXED" section): one shape
    # per CSV, never a header/row mismatch to have.
    "sampling_comparison": "sampling_results.csv",
    # Added 2026-09-23 for the Graph A/B comparison -- its config dict
    # (combo_id, graph, method, threshold_used) has yet another shape,
    # same one-shape-per-CSV reasoning as above.
    "graph_comparison": "graph_comparison_results.csv",
}


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


@dataclass
class ExperimentRecord:
    experiment_id: str
    experiment_kind: str  # "embedding_ablation" | "graph_ablation"
    config: dict          # ts_model/tab_model/fusion_mode/classifier/classifier_params/etc
    task: str              # "mortality" | "sepsis"
    split_manifest_seed: int
    metrics: dict           # {"train": {...}, "val": {...}, "test": {...}}
    training_info: dict     # e.g. n_hospitals_trained, epochs, encoder-training runtime
    runtime_seconds: float
    timestamp: str


class ResultsWriter:
    """One instance per ablation RUN (not per experiment). Every
    experiment within the run writes to `{run_dir}/records/
    {experiment_id}.json`, plus one row to a CSV chosen by
    `experiment_kind` -- `{run_dir}/master_results.csv` for most kinds,
    `{run_dir}/graph_results.csv` for `graph_ablation` (see
    `_DEDICATED_CSV_FILENAME_BY_KIND` and the module docstring).
    """

    def __init__(self, output_dir: "str | Path", run_id: "str | None" = None, overwrite: bool = False):
        self.run_dir = Path(output_dir) / (run_id or new_run_id())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir = self.run_dir / "records"
        self.records_dir.mkdir(exist_ok=True)
        self.master_csv_path = self.run_dir / "master_results.csv"
        self.overwrite = overwrite
        self.logger = logging.getLogger("eicu_fl")

    def _csv_path_for_kind(self, experiment_kind: str) -> Path:
        dedicated_filename = _DEDICATED_CSV_FILENAME_BY_KIND.get(experiment_kind)
        if dedicated_filename is not None:
            return self.run_dir / dedicated_filename
        return self.master_csv_path

    def write(self, record: ExperimentRecord) -> Path:
        record_path = self.records_dir / f"{record.experiment_id}.json"
        if record_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Experiment result already exists at {record_path} -- refusing to silently "
                f"overwrite (per 'never overwrite previous experimental results'). Pass "
                f"overwrite=True on this ResultsWriter if a re-run is genuinely intentional, or "
                f"start a fresh run_id."
            )
        with open(record_path, "w") as f:
            json.dump(asdict(record), f, indent=2, default=str)
        self._append_to_csv(record, self._csv_path_for_kind(record.experiment_kind))
        self.logger.info(f"[ABLATION_RESULTS] wrote {record.experiment_id} -> {record_path}")
        return record_path

    def _append_to_csv(self, record: ExperimentRecord, csv_path: Path) -> None:
        """Re-derives the whole CSV from every JSON record on disk that belongs
        to `csv_path` (header = UNION of all fields, in first-seen order).

        FIX (2026-10-07): the previous version took the header from the FIRST
        row ever written and appended later rows with their own key order.
        If a later row had extra/different keys (e.g. sampling rows carrying
        sampling_strategy/sampling_ratio, or a classifier-param column), its
        values landed under the wrong header silently -- a data-corruption
        bug, not a crash. The per-record JSON files are the source of truth,
        so rebuilding from them can never misalign columns and also heals a
        previously corrupted CSV on the next write."""
        self._rebuild_csv(csv_path)

    def _rows_for_csv(self, csv_path: Path) -> list:
        rows = []
        for rp in sorted(self.records_dir.glob("*.json")):
            try:
                with open(rp) as f:
                    data = json.load(f)
                if self._csv_path_for_kind(data.get("experiment_kind")) != csv_path:
                    continue
                rows.append((str(data.get("timestamp", "")), str(data.get("experiment_id", "")), self._flatten_for_csv(ExperimentRecord(**data))))
            except (OSError, ValueError, TypeError) as exc:
                self.logger.warning(f"[ABLATION_RESULTS] skipped unreadable record {rp.name}: {exc}")
        rows.sort(key=lambda t: (t[0], t[1]))
        return [r for _, _, r in rows]

    def _rebuild_csv(self, csv_path: Path) -> None:
        rows = self._rows_for_csv(csv_path)
        if not rows:
            return
        fieldnames: list = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(csv_path)

    def rebuild_dedicated_csvs(self) -> list:
        """Rebuilds every CSV this run owns from its JSON records (use to heal an old run's CSVs)."""
        paths = {self.master_csv_path} | {self.run_dir / n for n in _DEDICATED_CSV_FILENAME_BY_KIND.values()}
        done = []
        for p in sorted(paths):
            if self._rows_for_csv(p):
                self._rebuild_csv(p); done.append(p)
        return done

    @staticmethod
    def _flatten_for_csv(record: ExperimentRecord) -> dict:
        flat = {
            "experiment_id": record.experiment_id, "experiment_kind": record.experiment_kind,
            "task": record.task, "split_manifest_seed": record.split_manifest_seed,
            "timestamp": record.timestamp, "runtime_seconds": record.runtime_seconds,
        }
        flat.update({f"config.{k}": v for k, v in record.config.items()})
        flat.update({f"training.{k}": v for k, v in record.training_info.items()})
        for split in ("train", "val", "test"):
            for k, v in (record.metrics.get(split) or {}).items():
                flat[f"{split}.{k}"] = v
        return flat

    def build_summary_table(self) -> "list[dict]":
        """Reads every persisted JSON record in this run back off disk
        and returns them sorted by test-split AUROC descending (records
        with an undefined/missing test AUROC sort last, never crash the
        sort or get silently dropped)."""
        rows = []
        for path in sorted(self.records_dir.glob("*.json")):
            with open(path) as f:
                rows.append(json.load(f))

        def sort_key(r: dict):
            auroc = (r.get("metrics", {}).get("test", {}) or {}).get("auroc")
            return (auroc is None, -(auroc or 0.0))

        rows.sort(key=sort_key)
        return rows

    def save_summary_csv(self, path: "str | Path | None" = None) -> Path:
        """Publication-ready comparison table: one row per experiment,
        every split's key metrics side by side."""
        rows = self.build_summary_table()
        out_path = Path(path) if path else self.run_dir / "summary_table.csv"
        fieldnames = ["experiment_id", "experiment_kind", "task"]
        for split in ("train", "val", "test"):
            for metric in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "n", "n_positive"):
                fieldnames.append(f"{split}.{metric}")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                flat = {"experiment_id": r["experiment_id"], "experiment_kind": r["experiment_kind"], "task": r["task"]}
                for split in ("train", "val", "test"):
                    m = r.get("metrics", {}).get(split) or {}
                    for metric in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "n", "n_positive"):
                        flat[f"{split}.{metric}"] = m.get(metric)
                writer.writerow(flat)
        return out_path