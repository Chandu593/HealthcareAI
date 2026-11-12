"""
Standalone diagnostic for a COMPLETED local-training-style run --
checks quality using only what's already on disk (checkpoints +
training_manifest.json if present). Does NOT re-train anything, so
it's safe and fast to run against a real run that already finished.

Works against two manifest shapes, auto-detected:
  - PER-HOSPITAL (Phase 11, Phase 12's rounds, Phase 13's local-only
    baseline): manifest has a "hospitals" dict, one checkpoint file
    per hospital.
  - CENTRALIZED-SEQUENTIAL (Phase 13's centralized baseline, added
    2026-09-05): manifest has an "epoch_history" list instead, ONE
    continuously-updated model, ONE final checkpoint file.

Usage:
    python scripts/check_local_training_quality.py --checkpoint-dir checkpoints/phase11_local
    python scripts/check_local_training_quality.py --checkpoint-dir checkpoints/phase13_baselines/local_only
    python scripts/check_local_training_quality.py --checkpoint-dir checkpoints/phase13_baselines/centralized_sequential
    (path is run.checkpoint_dir + the phase-specific subdirectory from your config -- these are the
    defaults, resolved relative to wherever you ran run_pipeline.py from)

PER-HOSPITAL manifests (added 2026-08-31) additionally report:
  - how many hospitals' loss actually decreased vs. increased vs. stayed flat
  - the distribution of n_mortality_known / n_sepsis_known across hospitals
    (how many hospitals trained with little or no real label signal)
CENTRALIZED-SEQUENTIAL manifests (added 2026-09-05) additionally report:
  - the full epoch-by-epoch mean loss trajectory (not just whatever was logged)
  - whether improvement is accelerating or decelerating (a never-reset continuous
    model doesn't have the same "diminishing returns per round" signature FedAvg
    rounds do, so this is worth seeing explicitly rather than assumed)
Regardless of manifest shape, this always checks the checkpoint(s) themselves for
two problems a "run completed successfully" log line cannot rule out on its own:
  - NaN/Inf weights (a run predating the 2026-08-31 NaN-safety fix could
    have silently saved a corrupted checkpoint for any hospital whose
    loss went non-finite mid-training)
  - zero weight change from the shared initial checkpoint (would suggest
    that hospital's -- or for centralized, the model's -- training loop
    didn't actually run/update anything)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _report_per_hospital(manifest: dict) -> None:
    print("\n=== training_manifest.json summary ===")
    print(
        f"trained={manifest['n_hospitals_trained']} "
        f"skipped_empty={manifest['n_hospitals_skipped_empty']} "
        f"skipped_no_labels={manifest['n_hospitals_skipped_no_labels']} "
        f"skipped_non_finite={manifest.get('n_hospitals_skipped_non_finite', 'N/A -- predates this field')}"
    )

    loss_decreased, loss_increased, loss_flat = 0, 0, 0
    n_mortality_known_list, n_sepsis_known_list = [], []
    for h in manifest["hospitals"].values():
        history = h["loss_history"]
        if len(history) >= 2:
            if history[-1] < history[0]:
                loss_decreased += 1
            elif history[-1] > history[0]:
                loss_increased += 1
            else:
                loss_flat += 1
        n_mortality_known_list.append(h["n_mortality_known"])
        n_sepsis_known_list.append(h["n_sepsis_known"])

    print(f"\nLoss trend across {len(manifest['hospitals'])} hospitals:")
    print(f"  decreased: {loss_decreased}   increased: {loss_increased}   flat: {loss_flat}")
    if loss_increased:
        increasing = [
            (hid, h) for hid, h in manifest["hospitals"].items()
            if len(h["loss_history"]) >= 2 and h["loss_history"][-1] > h["loss_history"][0]
        ]
        increasing.sort(key=lambda row: row[1]["n_patients"])  # smallest hospital first
        all_n = sorted(h["n_patients"] for h in manifest["hospitals"].values())
        print("  hospitals with increasing loss, smallest first (tests whether this is really a")
        print("  small/sparse-hospital effect, rather than just asserting it):")
        for hid, h in increasing:
            n = h["n_patients"]
            pct_of_hospitals = sum(1 for x in all_n if x <= n) / len(all_n) * 100
            history = h["loss_history"]
            n_mort, n_sep = h["n_mortality_known"], h["n_sepsis_known"]
            mort_pct = n_mort / n * 100 if n else 0.0
            sep_pct = n_sep / n * 100 if n else 0.0
            print(
                f"    hospital {hid}: n_patients={n} ({pct_of_hospitals:.0f}th percentile by size), "
                f"loss {history[0]:.4f} -> {history[-1]:.4f}, full history={[round(x, 4) for x in history]}"
            )
            print(
                f"      label coverage: mortality_known={n_mort}/{n} ({mort_pct:.0f}%), "
                f"sepsis_known={n_sep}/{n} ({sep_pct:.0f}%)"
            )

        all_mort_pct = sorted(
            h["n_mortality_known"] / h["n_patients"] * 100
            for h in manifest["hospitals"].values() if h["n_patients"] > 0
        )
        all_sep_pct = sorted(
            h["n_sepsis_known"] / h["n_patients"] * 100
            for h in manifest["hospitals"].values() if h["n_patients"] > 0
        )
        print(
            f"  for comparison, median label coverage across all {len(manifest['hospitals'])} "
            f"hospitals: mortality={all_mort_pct[len(all_mort_pct) // 2]:.0f}%, "
            f"sepsis={all_sep_pct[len(all_sep_pct) // 2]:.0f}%"
        )

        median_n = all_n[len(all_n) // 2]
        all_below_median_size = all(h["n_patients"] <= median_n for _, h in increasing)
        if all_below_median_size:
            print(
                f"  -> ALL {len(increasing)} increasing-loss hospitals are at or below the median "
                f"hospital size ({median_n} patients) -- consistent with the small/sparse-hospital "
                f"hypothesis BY RAW SIZE, not just asserted."
            )
        else:
            print(
                "  -> NOT all increasing-loss hospitals are small BY RAW PATIENT COUNT -- compare "
                "each one's label-coverage percentages above against the population medians instead; "
                "a hospital can have many raw patients but still have unusually sparse USABLE labels."
            )

    def _summary(name, values):
        values = sorted(values)
        n_zero = sum(1 for v in values if v == 0)
        print(
            f"  {name}: min={values[0]} median={values[len(values) // 2]} max={values[-1]} "
            f"(zero for {n_zero}/{len(values)} hospitals)"
        )

    print("\nLabel coverage per hospital:")
    _summary("n_mortality_known", n_mortality_known_list)
    _summary("n_sepsis_known", n_sepsis_known_list)


def _check_per_hospital_checkpoints(ckpt_dir: Path, initial_state: dict, sample_size: int) -> None:
    hospital_checkpoints = sorted(ckpt_dir.glob("hospital_*_local_weights.pt"))
    print(f"Found {len(hospital_checkpoints)} hospital checkpoints in {ckpt_dir}")

    sample = hospital_checkpoints[:sample_size]
    print(f"\n=== Weight-delta and NaN/Inf check ({len(sample)} of {len(hospital_checkpoints)} hospitals) ===")
    nan_hospitals, zero_delta_hospitals = [], []
    deltas: list[tuple[str, float]] = []
    for ckpt_path in sample:
        state = torch.load(ckpt_path, weights_only=True)
        has_bad = any(not torch.isfinite(v).all() for v in state.values())
        total_delta = sum((state[k] - initial_state[k]).norm().item() for k in initial_state if k in state)
        if has_bad:
            nan_hospitals.append(ckpt_path.name)
        if total_delta == 0.0:
            zero_delta_hospitals.append(ckpt_path.name)
        deltas.append((ckpt_path.name, total_delta))

    if len(sample) <= 20:
        for name, total_delta in deltas:
            flag = " *** NaN/Inf DETECTED ***" if name in nan_hospitals else ""
            print(f"  {name}: total_weight_delta_from_initial={total_delta:.4f}{flag}")
    else:
        finite_deltas = sorted(d for _, d in deltas if d > 0.0)
        if finite_deltas:
            n = len(finite_deltas)
            print(
                f"  delta distribution (excluding zero/NaN, n={n}): "
                f"min={finite_deltas[0]:.4f} median={finite_deltas[n // 2]:.4f} max={finite_deltas[-1]:.4f}"
            )
            outlier_low, outlier_high = finite_deltas[0], finite_deltas[-1]
            if finite_deltas[n // 2] > 0:
                spread_ratio = outlier_high / outlier_low if outlier_low > 0 else float("inf")
                tightness = "  (tight, consistent across hospitals)" if spread_ratio < 3 else "  (wide spread -- worth a closer look at the extremes)"
                print(f"  max/min ratio: {spread_ratio:.2f}x{tightness}")

    if nan_hospitals:
        print(f"\n!!! {len(nan_hospitals)} checkpoint(s) contain NaN/Inf weights: {nan_hospitals}")
        print("    These are corrupted -- do not feed them into FedAvg. Re-run with the 2026-08-31")
        print("    NaN-safety fix so these hospitals get cleanly skipped instead of saved.")
    if zero_delta_hospitals:
        print(f"\n!!! {len(zero_delta_hospitals)} checkpoint(s) show ZERO change from the initial weights: "
              f"{zero_delta_hospitals}")
        print("    Training may not have actually run for these hospitals -- worth investigating.")
    if not nan_hospitals and not zero_delta_hospitals:
        print(f"\nAll {len(sample)} sampled checkpoints look healthy: finite weights, genuine change from initial.")


def _report_centralized_sequential(manifest: dict) -> None:
    print("\n=== training_manifest.json summary (centralized-sequential) ===")
    print(
        f"total_epochs={manifest['total_epochs']} n_hospitals={manifest['n_hospitals']} "
        f"n_skipped_empty={manifest['n_skipped_empty']} "
        f"n_non_finite_step_events={manifest.get('n_non_finite_step_events', 'N/A')} "
        f"n_no_label_step_events={manifest.get('n_no_label_step_events', 'N/A')}"
    )

    history = manifest["epoch_history"]
    losses = [(h["epoch"], h["mean_loss"]) for h in history if h["mean_loss"] is not None]
    print(f"\nFull epoch-by-epoch mean loss ({len(losses)}/{len(history)} epochs had a valid mean):")
    for epoch, loss in losses:
        print(f"  epoch {epoch:>3}: {loss:.4f}")

    if len(losses) >= 2:
        first_epoch, first_loss = losses[0]
        last_epoch, last_loss = losses[-1]
        net_change = last_loss - first_loss
        net_pct = (net_change / first_loss * 100) if first_loss else float("nan")
        print(f"\nNet change (epoch {first_epoch} -> {last_epoch}): {first_loss:.4f} -> {last_loss:.4f} ({net_pct:+.1f}%)")

        n_increases = sum(1 for i in range(1, len(losses)) if losses[i][1] > losses[i - 1][1])
        print(f"epochs with an increase from the previous epoch: {n_increases}/{len(losses) - 1}")

        mid = len(losses) // 2
        first_half_change = losses[mid][1] - losses[0][1]
        second_half_change = losses[-1][1] - losses[mid][1]
        print(f"first-half change: {first_half_change:+.4f}   second-half change: {second_half_change:+.4f}")
        if abs(second_half_change) > abs(first_half_change) * 1.1:
            print("  -> improvement is ACCELERATING in the second half -- unusual for a converging model "
                  "(FedAvg rounds typically show the OPPOSITE, diminishing movement round-over-round). "
                  "This run likely has NOT plateaued and may improve further with more epochs.")
        elif abs(second_half_change) < abs(first_half_change) * 0.9:
            print("  -> improvement is DECELERATING in the second half -- the more typical convergence pattern.")
        else:
            print("  -> improvement rate is roughly steady across both halves.")

    contributed = [h["n_hospitals_contributed"] for h in history]
    print(
        f"\nhospitals contributing per epoch: min={min(contributed)} max={max(contributed)} "
        f"(out of {manifest['n_hospitals']} non-empty hospitals)"
    )


def _check_single_final_model(manifest: dict, initial_state: dict, ckpt_dir: Path) -> None:
    final_model_path = Path(manifest["final_model_path"])
    if not final_model_path.exists():
        fallback = ckpt_dir / final_model_path.name
        final_model_path = fallback if fallback.exists() else None
    print("\n=== Weight-delta and NaN/Inf check (single continuously-trained model) ===")
    if final_model_path is None:
        print(f"  ERROR: final model not found at {manifest['final_model_path']!r} or under {ckpt_dir}")
        return

    state = torch.load(final_model_path, weights_only=True)
    has_bad = any(not torch.isfinite(v).all() for v in state.values())
    total_delta = sum((state[k] - initial_state[k]).norm().item() for k in initial_state if k in state)
    flag = " *** NaN/Inf DETECTED ***" if has_bad else ""
    print(f"  {final_model_path.name}: total_weight_delta_from_initial={total_delta:.4f}{flag}")

    if has_bad:
        print("\n!!! the final model contains NaN/Inf weights -- do not use this checkpoint.")
    elif total_delta == 0.0:
        print("\n!!! the final model shows ZERO change from the initial weights -- training may not have run.")
    else:
        print("\nFinal model looks healthy: finite weights, genuine change from initial.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. checkpoints/phase11_local")
    parser.add_argument(
        "--sample", type=int, default=1000,
        help="PER-HOSPITAL manifests only: how many hospital checkpoints to load and weight-delta-check, "
             "taken in sorted-filename order. Defaults to 1000 -- comfortably above any realistic "
             "hospital count (208 in this dataset), so a plain run checks EVERY checkpoint by default "
             "rather than an arbitrary alphabetically-first slice (real eICU hospitalid values are not "
             "sequential, so 'first N' is not a representative sample). The manifest-based loss/label "
             "stats always cover every hospital regardless of this setting. Ignored for "
             "centralized-sequential manifests, which have only one final model to check.",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    manifest_path = ckpt_dir / "training_manifest.json"
    initial_path = ckpt_dir / "global_initial_weights.pt"

    if not initial_path.exists():
        print(f"ERROR: {initial_path} not found -- is --checkpoint-dir correct?")
        return
    initial_state = torch.load(initial_path, weights_only=True)

    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    is_centralized_sequential = manifest is not None and "epoch_history" in manifest

    if is_centralized_sequential:
        _report_centralized_sequential(manifest)
        _check_single_final_model(manifest, initial_state, ckpt_dir)
    else:
        if manifest is not None:
            _report_per_hospital(manifest)
        else:
            print(
                f"\nNo training_manifest.json at {manifest_path} -- this run predates that fix "
                f"(added 2026-08-31). Loss-curve and label-coverage numbers aren't recoverable for this "
                f"specific run; only the weight-delta/NaN checks below are still possible."
            )
        _check_per_hospital_checkpoints(ckpt_dir, initial_state, args.sample)


if __name__ == "__main__":
    main()
