"""Three-modality suite entry point (additive; does not touch run_pipeline.py / run_ablation.py / run_paper.py).

  python run_three_modality.py --config configs/three_modality.yaml --stage {incident_label,tcd_features,tcd_encoder,gru_inputs,gru_encoder,tabular_inputs,tabular_ae,
                                                                              cross_attention,nonfl_config1,nonfl_config2,nonfl_config3,nonfl_config4,nonfl_report,
                                                                              nonfl_tune,fl_tune,fl_config1,fl_config2,fl_config3,fl_config4,fl_personalized,fl_report}
         [--run-id ID] [--force-recompute] [--inspect-cache] [--reuse-phase1to3] [--set KEY=VALUE ...]
         [--incident-run-id ID] [--feature-run-id ID]          (tcd_encoder only: pick specific completed upstream runs)

Stages (each writes only under <run.output_dir>/three_modality/, never overwrites a completed run, reuses a VERIFIED identical run unless --force-recompute):
  incident_label : the incident sepsis-proxy label (ICU hours 24-96 by default) for every patient in the PERMANENT split manifest.
  tcd_features   : ONE unified Temporal Clinical Dynamics feature vector per patient from the first 24 ICU hours (+ train-only preprocessing, availability
                   report, real-data cutoff audit).
  tcd_encoder    : the MLP encoder (focal multitask, train/val only) -> ONE embedding per patient, saved like the GRU / autoencoder embeddings.
  gru_inputs     : fresh 24 x 1-hour sequences (value + observation mask) of the first 24 ICU hours, train-only scaling, fixed length, cutoff audit.
  gru_encoder    : the existing GRU encoder, pooled focal multitask training (separate alpha/gamma/weights per task) -> ONE 128-d embedding per patient.
  tabular_inputs : the paper track's clean static feature set (no APACHE scores, no labs, no label-ingredient groups), train-only preprocessing, typed columns.
  tabular_ae     : the existing autoencoder, pooled training: reconstruction loss (MSE continuous / BCE binary) + focal multitask heads -> ONE 64-d embedding per patient.
  cross_attention: the three-token cross-attention (GRU -> Tabular -> TCD) over the FROZEN saved embeddings, aligned by patient ID; centrally pooled training; ONE fused embedding per
                   patient in the TCD patient order (outputs/three_modality/cross_attention/). Needs --tcd-encoder-run-id / --gru-encoder-run-id / --tabular-ae-run-id.
  nonfl_config1..4 : Non-FL, hospital-wise Configs 1-4 (one independent model per hospital and task; outputs/three_modality/non_federated/). Need --cross-attention-run-id.
  nonfl_report     : consolidates the completed Non-FL runs of one cross-attention run (outputs/three_modality/reports/).
  nonfl_tune       : validation-only hyper-parameter tuning (outputs/three_modality/tuning/); apply with --hparams-run-id on nonfl_config1-4. Never loads test data.
  fl_config1..4    : FEDERATED downstream Configs 1-4 on the frozen, CENTRALLY pretrained embeddings (clients = hospitals; outputs/three_modality/federated/). FedAvg / FedProx / FedAdam /
                     SCAFFOLD for LR, MLP, GCN; federated-histogram GBDT (classifier gbdt_fed) for trees -- never parameter-averaged. Not end-to-end federated learning.
  fl_personalized  : separate personalisation experiment (FedPer, FedRep, FedAvg + local fine-tuning; MLP / GCN).
  fl_tune          : validation-only FL hyper-parameter tuning (outputs/three_modality/fl_tuning/); apply with --fl-hparams-run-id. Never loads test labels.
  fl_report        : FL vs Non-FL comparison on identical test patients, hospital-macro and pooled in separate tables (outputs/three_modality/reports/).
The split manifest is only LOADED, never built or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.splitting.patient_split import load_split_manifest
from src.paper.features import _resolve_table
from src.paper.overrides import apply_set_overrides
from src.three_modality import artifacts
from src.three_modality.config import get_gru_cfg, get_incident_sepsis_cfg, get_tabular_cfg, get_tcd_cfg
from src.three_modality.downstream_config import CLASSIFIERS, get_cross_attention_cfg, get_downstream_cfg, get_tuning_cfg
from src.three_modality.incident_sepsis import build_incident_sepsis_label, current_identity, list_runs
from src.utils.config import load_config
from src.utils.logging_setup import close_logging, setup_logging

STAGES = ("incident_label", "tcd_features", "tcd_encoder", "gru_inputs", "gru_encoder", "tabular_inputs", "tabular_ae", "cross_attention",
          "nonfl_config1", "nonfl_config2", "nonfl_config3", "nonfl_config4", "nonfl_report", "nonfl_tune",
          "fl_tune", "fl_config1", "fl_config2", "fl_config3", "fl_config4", "fl_personalized", "fl_report")
_NONFL_DEFAULT_CLASSIFIERS = "lr,mlp,lightgbm,gbdt"


def _under_project_root(cfg, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(cfg.get("run.output_dir", "outputs")).parent / path


def _inspect(base: Path, ident: str, extra: str = "") -> None:
    runs = list_runs(base)
    print(f"current identity {ident[:12]}{extra}  ({len(runs)} completed run(s) under {base})")
    for r in runs:
        c = r.get("counts", {})
        print(f"  {r['run_id']}  created {r.get('created_utc', '?')[:19]}  {json.dumps({k: c[k] for k in list(c)[:6]}) if c else ''}  "
              f"matches-current={'YES' if r.get('identity') == ident else 'no'}")


def _pick_run(base: Path, ident: str, explicit: "str | None", what: str, hint: str) -> Path:
    if explicit:
        d = base / explicit
        if not (d / "manifest.json").is_file():
            raise FileNotFoundError(f"{what} run '{explicit}' is not a completed run under {base}")
        return d
    hit = artifacts.find_reusable_run(base, ident)
    if not hit:
        raise FileNotFoundError(f"no verified {what} run matches the CURRENT configuration under {base}. {hint}")
    return Path(hit["run_dir"])


def _single_or_explicit(base: Path, explicit: "str | None", what: str, flag: str) -> Path:
    if explicit:
        d = base / explicit
        if not (d / "manifest.json").is_file():
            raise FileNotFoundError(f"{what} run {explicit!r} not found under {base} (no manifest.json)")
        return d
    runs = artifacts.list_runs(base)
    if len(runs) == 1:
        return Path(runs[0]["run_dir"])
    names = ", ".join(r["run_id"] for r in runs) or "none"
    raise FileNotFoundError(f"{len(runs)} completed {what} runs under {base} ({names}); choose one explicitly with {flag}")


def _run_cross_attention_stage(args, out_root: Path, ca_base: Path, patient_path, xcfg: dict, tcfg: dict) -> int:
    from src.three_modality import cross_attention3 as ca
    runs = {"tcd": _single_or_explicit(out_root / "tcd" / "encoder", args.tcd_encoder_run_id, "TCD encoder", "--tcd-encoder-run-id"),
            "gru": _single_or_explicit(out_root / "gru" / "encoder", args.gru_encoder_run_id, "GRU encoder", "--gru-encoder-run-id"),
            "tabular": _single_or_explicit(out_root / "tabular" / "encoder", args.tabular_ae_run_id, "tabular autoencoder", "--tabular-ae-run-id")}
    tman = json.loads((runs["tcd"] / "manifest.json").read_text(encoding="utf-8"))
    runs["incident"] = out_root / "incident_sepsis" / tman["incident_label_run"]
    bases = {"tcd_features": str(out_root / "tcd" / "features"), "gru_inputs": str(out_root / "gru" / "inputs"), "tabular_inputs": str(out_root / "tabular" / "inputs")}
    runs = {k: str(v) for k, v in runs.items()}
    if args.inspect_cache:
        enc = {k: json.loads((Path(runs[k]) / "manifest.json").read_text(encoding="utf-8")) for k in ca.TOKEN_ORDER}
        iman = json.loads((Path(runs["incident"]) / "manifest.json").read_text(encoding="utf-8"))
        _inspect(ca_base, ca.cross_attention_identity(enc, iman, xcfg, tcfg["window_minutes"]), f"  (encoders {', '.join(enc[k]['run_id'] for k in ca.TOKEN_ORDER)})")
        return 0
    patients = pd.read_csv(patient_path, usecols=["patientunitstayid", "hospitaldischargestatus", "unitdischargeoffset"])
    res = ca.build_cross_attention(runs, bases, patients, xcfg, tcfg, ca_base, run_id=args.run_id, force=args.force_recompute)
    m = res["manifest"]
    print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "embedding_shape": m.get("embedding_shape"), "token_order": m.get("token_order"),
                      "encoder_runs": {k: v["run_id"] for k, v in m["encoder_runs"].items()}, "selected_checkpoint": m.get("selected_checkpoint"), "availability": m.get("availability"),
                      "training_population": m.get("training_population", {}).get("landmark_patients_by_split")}, indent=2))
    return 0


def _run_nonfl_stage(args, out_root: Path, ca_base: Path, nonfl_base: Path, reports_base: Path, dcfg: dict, tucfg: dict) -> int:
    from src.three_modality import hw_data, hw_tune, non_federated as nf, nonfl_report as nr
    ca_dir = _single_or_explicit(ca_base, args.cross_attention_run_id, "cross_attention", "--cross-attention-run-id")
    if args.stage == "nonfl_report":
        res = nr.build_nonfl_report(nonfl_base, reports_base, ca_dir.name, include_partial=args.include_partial, run_id=args.run_id,
                                    n_bootstrap=int(dcfg["reporting"]["n_bootstrap"]), top_hospitals=tuple(dcfg["reporting"]["top_hospitals"]), seed=int(dcfg["reporting"]["seed"]))
        print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "nonfl_runs_used": res["manifest"]["nonfl_runs_used"], "n_variants": res["manifest"]["n_variants"],
                          "n_comparisons": res["manifest"]["n_comparisons"]}, indent=2))
        return 0
    if args.stage == "nonfl_tune":
        fams = tuple(f.strip() for f in (args.tune_families or ",".join(tucfg["families"])).split(",") if f.strip())
        tasks_t = ("mortality", "sepsis") if args.tasks == "both" else (args.tasks,)
        cohort = hw_data.load_cohort(ca_dir, out_root, ("fused",))
        res = hw_tune.run_tuning(cohort, dcfg, out_root / "tuning", families=fams, tasks=tasks_t, n_hospitals=int(args.tune_hospitals or tucfg["n_hospitals"]), quick=bool(tucfg["quick"]),
                                 run_id=args.run_id, force=args.force_recompute)
        print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "best_hparams": res["best_hparams"]}, indent=2))
        return 0
    config = int(args.stage[-1])
    sets = ("fused", "gru", "tabular", "tcd") if args.embedding_set == "all" else (args.embedding_set,)
    if config != 1 and sets != ("fused",):
        raise SystemExit("--embedding-set applies to nonfl_config1 only (Configs 2-4 use the fused embedding)")
    clfs = tuple(c.strip() for c in args.classifiers.split(",") if c.strip())
    bad = [c for c in clfs if c not in CLASSIFIERS]
    if bad:
        raise SystemExit(f"unknown classifier(s) {bad}; choose from {CLASSIFIERS}")
    graphs = ("A", "B") if args.graph == "both" else (args.graph,)
    tasks = ("mortality", "sepsis") if args.tasks == "both" else (args.tasks,)
    cohort = hw_data.load_cohort(ca_dir, out_root, sets)
    hp = hw_tune.load_best_hparams(out_root / "tuning", args.hparams_run_id) if args.hparams_run_id else None
    res = nf.run_nonfl(config, cohort, dcfg, nonfl_base, embedding_sets=sets, classifiers=clfs, graphs=graphs, tasks=tasks, max_hospitals=args.max_hospitals,
                       run_id=args.run_id, force=args.force_recompute, hparams=hp, hparams_source=args.hparams_run_id)
    summ = json.loads((Path(res["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    brief = {v: {"trained": s["n_trained"], "skipped": s["n_skipped"], "failed": s.get("n_failed", 0), "auroc_macro": s["metrics"]["auroc"]["macro_mean"],
                 "auroc_weighted": s["metrics"]["auroc"]["sample_weighted_mean"], "auprc_macro": s["metrics"]["auprc"]["macro_mean"]} for v, s in summ.items()}
    print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "partial": bool(args.max_hospitals), "variants": brief}, indent=2))
    return 0


def _run_fl_stage(args, out_root: Path, ca_base: Path, nonfl_base: Path, reports_base: Path, dcfg: dict, flcfg: dict) -> int:
    """fl_* stages (federated downstream learning on frozen, centrally pretrained embeddings)."""
    from src.three_modality import fl_report as fr, fl_tune, federated as fed, hw_data
    from src.three_modality.fl_config import FL_CLASSIFIERS, NN_ALGORITHMS, PERSONALIZATION_METHODS, PERSONALIZED_MODELS, apply_fl_hparams
    ca_dir = _single_or_explicit(ca_base, args.cross_attention_run_id, "cross_attention", "--cross-attention-run-id")
    fed_base, tune_base = out_root / "federated", out_root / "fl_tuning"
    if args.stage == "fl_report":
        res = fr.build_fl_report(nonfl_base, fed_base, reports_base, ca_dir.name, include_partial=args.include_partial, run_id=args.run_id,
                                 n_bootstrap=int(dcfg["reporting"]["n_bootstrap"]), seed=int(dcfg["reporting"]["seed"]))
        print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "fl_runs_used": res["manifest"]["fl_runs_used"], "nonfl_runs_used": res["manifest"]["nonfl_runs_used"],
                          "n_comparisons": res["manifest"]["n_comparisons"], "identical_test_patient_checks_failed": res["manifest"]["n_identical_test_patient_checks_failed"]}, indent=2))
        return 0
    algs = tuple(a.strip() for a in args.algorithms.split(",") if a.strip())
    bad = [a for a in algs if a not in NN_ALGORITHMS]
    if bad:
        raise SystemExit(f"unknown algorithm(s) {bad}; parameter-averaging algorithms are {NN_ALGORITHMS} (trees use --classifiers gbdt_fed, never FedAvg)")
    if args.scope:
        flcfg = dict(flcfg, scope=args.scope)
    hp = fl_tune.load_fl_hparams(tune_base, args.fl_hparams_run_id) if args.fl_hparams_run_id else None
    flcfg = apply_fl_hparams(flcfg, hp)
    tasks = ("mortality", "sepsis") if args.tasks == "both" else (args.tasks,)
    graphs = ("A", "B") if args.graph == "both" else (args.graph,)
    clf_arg = None if args.classifiers == _NONFL_DEFAULT_CLASSIFIERS else args.classifiers
    clfs = tuple(c.strip() for c in (clf_arg or ",".join(FL_CLASSIFIERS)).split(",") if c.strip())
    bad = [c for c in clfs if c not in FL_CLASSIFIERS]
    if bad:
        raise SystemExit(f"unknown FL classifier(s) {bad}; choose from {FL_CLASSIFIERS}")
    if args.stage == "fl_tune":
        fams = tuple(f.strip() for f in (args.tune_families or ",".join(flcfg["tuning"]["families"])).split(",") if f.strip())
        cohort = hw_data.load_cohort(ca_dir, out_root, ("fused",))
        res = fl_tune.run_fl_tuning(cohort, dcfg, flcfg, tune_base, families=fams, algorithms=algs, tasks=tasks, max_hospitals=args.max_hospitals, quick=bool(flcfg["tuning"]["quick"]),
                                    run_id=args.run_id, force=args.force_recompute, resume=args.resume_existing_run)
        print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "best_hparams": res["best_hparams"]}, indent=2))
        return 0
    if args.stage == "fl_personalized":
        methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
        models = tuple(m.strip() for m in args.models.split(",") if m.strip())
        cohort = hw_data.load_cohort(ca_dir, out_root, ("fused",))
        res = fed.run_fl_personalized(cohort, dcfg, flcfg, fed_base, methods=methods, models=models, graph=args.pers_graph, tasks=tasks, max_hospitals=args.max_hospitals, run_id=args.run_id,
                                      force=args.force_recompute, resume=args.resume_existing_run, hparams_source=args.fl_hparams_run_id)
    else:
        config = int(args.stage[-1])
        sets = ("fused", "gru", "tabular", "tcd") if args.embedding_set == "all" else (args.embedding_set,)
        if config != 1 and sets != ("fused",):
            raise SystemExit("--embedding-set applies to fl_config1 only (Configs 2-4 use the fused embedding)")
        cohort = hw_data.load_cohort(ca_dir, out_root, sets)
        res = fed.run_fl(config, cohort, dcfg, flcfg, fed_base, embedding_sets=sets, classifiers=clfs, graphs=graphs, tasks=tasks, algorithms=algs, c4_gcn_algorithm=args.c4_gcn_algorithm,
                         max_hospitals=args.max_hospitals, run_id=args.run_id, force=args.force_recompute, resume=args.resume_existing_run, hparams_source=args.fl_hparams_run_id)
    if res.get("cache_hit"):
        print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": True}, indent=2))
        return 0
    summ = json.loads((Path(res["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    brief = {v: {"paired_hospitals_evaluated": s["counts"]["n_paired_evaluated"], "fl_only_clients_evaluated": s["counts"]["n_fl_only_evaluated"], "failed": s.get("n_failed", 0),
                 "paired_auroc_macro": s["paired_nonfl"]["metrics"]["auroc"]["macro_mean"], "paired_auroc_weighted": s["paired_nonfl"]["metrics"]["auroc"]["sample_weighted_mean"],
                 "val_auroc_hist_at_best": s["training"].get("val_auroc_hist_at_best"), "seconds": s["training"].get("seconds")} for v, s in summ.items()}
    print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": False, "partial": bool(args.max_hospitals), "variants": brief}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--run-id", default=None, help="explicit run id (an existing directory with this id is an error, never overwritten)")
    ap.add_argument("--force-recompute", action="store_true", help="ignore a verified identical run and create a NEW run directory")
    ap.add_argument("--inspect-cache", action="store_true", help="list completed runs of this stage and whether each matches the CURRENT parameters/inputs; computes nothing")
    ap.add_argument("--reuse-phase1to3", action="store_true", help="resolve table paths from the saved Phase-1 summary instead of data.root")
    ap.add_argument("--incident-run-id", default=None, help="tcd_encoder: use this completed incident_label run instead of the one matching the current config")
    ap.add_argument("--feature-run-id", default=None, help="tcd_encoder: use this completed tcd_features run instead of the one matching the current config")
    ap.add_argument("--tcd-encoder-run-id", default=None, help="cross_attention: the completed TCD encoder run to use (required when more than one exists)")
    ap.add_argument("--gru-encoder-run-id", default=None, help="cross_attention: the completed GRU encoder run to use (required when more than one exists)")
    ap.add_argument("--tabular-ae-run-id", default=None, help="cross_attention: the completed tabular autoencoder run to use (required when more than one exists)")
    ap.add_argument("--cross-attention-run-id", default=None, help="nonfl_*: the completed cross_attention run to use (required when more than one exists)")
    ap.add_argument("--embedding-set", default="fused", choices=("fused", "gru", "tabular", "tcd", "all"), help="nonfl_config1: which frozen embedding(s) to classify (default fused)")
    ap.add_argument("--graph", default="both", choices=("A", "B", "both"), help="nonfl_config2/3/4: Graph A (kNN + cosine threshold) and/or Graph B (MiniBatchKMeans candidates)")
    ap.add_argument("--tasks", default="both", choices=("mortality", "sepsis", "both"))
    ap.add_argument("--classifiers", default="lr,mlp,lightgbm,gbdt", help="nonfl_config1/4: comma-separated subset of lr,mlp,lightgbm,gbdt")
    ap.add_argument("--max-hospitals", type=int, default=None, help="nonfl_*: smoke test on the first N hospitals only; the run is marked partial and excluded from the report by default")
    ap.add_argument("--hparams-run-id", default=None, help="nonfl_config1-4: apply the validation-tuned hyper-parameters (best_hparams.json) of this nonfl_tune run")
    ap.add_argument("--tune-hospitals", type=int, default=None, help="nonfl_tune: hospitals per task used for tuning (default 60; the GCN uses at most ~40)")
    ap.add_argument("--tune-families", default=None, help="nonfl_tune: comma-separated subset of lr,mlp,lightgbm,gbdt,gcn")
    ap.add_argument("--include-partial", action="store_true", help="nonfl_report: also include partial (--max-hospitals) runs")
    ap.add_argument("--algorithms", default="fedavg,fedprox,fedadam,scaffold", help="fl_*: parameter-averaging algorithms (subset of fedavg,fedprox,fedadam,scaffold); trees use --classifiers gbdt_fed")
    ap.add_argument("--c4-gcn-algorithm", default="fedavg", choices=("fedavg", "fedprox", "fedadam", "scaffold"), help="fl_config4: the algorithm of the stage-1 federated GCN (embedding generator)")
    ap.add_argument("--scope", default=None, choices=("all", "nonfl_matched"), help="fl_*: clients = every hospital with a training patient (all, default) or only the hospitals the Non-FL regime trains (control)")
    ap.add_argument("--fl-hparams-run-id", default=None, help="fl_*: apply the validation-tuned FL hyper-parameters (best_hparams.json) of this fl_tune run")
    ap.add_argument("--resume-existing-run", action="store_true", help="fl_*: resume the interrupted run with this --run-id (identity checked; completed runs are never resumed or overwritten)")
    ap.add_argument("--methods", default=",".join(("fedper", "fedrep", "finetune")), help="fl_personalized: comma-separated subset of fedper,fedrep,finetune")
    ap.add_argument("--models", default="mlp,gcn", help="fl_personalized: comma-separated subset of mlp,gcn")
    ap.add_argument("--pers-graph", default="A", choices=("A", "B"), help="fl_personalized: the hospital-local graph used by the GCN")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config value, e.g. three_modality.tcd.encoder.embedding_dim=32")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        apply_set_overrides(cfg.raw, args.set)
        block = get_incident_sepsis_cfg(cfg)
        tcfg = get_tcd_cfg(cfg)
        gcfg = get_gru_cfg(cfg)
        acfg = get_tabular_cfg(cfg)
        xcfg = get_cross_attention_cfg(cfg)
        dcfg = get_downstream_cfg(cfg)
        tucfg = get_tuning_cfg(cfg)
        from src.three_modality.fl_config import get_fl_cfg
        flcfg = get_fl_cfg(cfg)
    except ValueError as exc:
        ap.error(str(exc))
    out_root = Path(cfg.get("run.output_dir", "outputs")) / "three_modality"
    inc_base, feat_base, enc_base = out_root / "incident_sepsis", out_root / "tcd" / "features", out_root / "tcd" / "encoder"
    gin_base, genc_base = out_root / "gru" / "inputs", out_root / "gru" / "encoder"
    tin_base, tae_base = out_root / "tabular" / "inputs", out_root / "tabular" / "encoder"
    ca_base, nonfl_base, reports_base = out_root / "cross_attention", out_root / "non_federated", out_root / "reports"
    setup_logging(str(out_root / "logs"))                    # own log directory: never mixed into the legacy / paper pipeline.log
    try:
        if args.stage.startswith("nonfl_"):
            return _run_nonfl_stage(args, out_root, ca_base, nonfl_base, reports_base, dcfg, tucfg)
        if args.stage.startswith("fl_"):
            return _run_fl_stage(args, out_root, ca_base, nonfl_base, reports_base, dcfg, flcfg)
        if args.reuse_phase1to3:
            from run_pipeline import load_phase1_summary
            src = load_phase1_summary(cfg)
        else:
            src = {"data_root": cfg.data_root}
        mpath = _under_project_root(cfg, (cfg.get("ablation.split", {}) or {}).get("manifest_path", "outputs/ablation/split_manifest.json"))
        manifest = load_split_manifest(mpath)                # raises FileNotFoundError with guidance if absent -- never builds one here
        patient_path = _resolve_table(src, "patient")
        if patient_path is None:
            raise FileNotFoundError("patient table not found under data.root / the Phase-1 summary")
        if args.stage == "cross_attention":
            return _run_cross_attention_stage(args, out_root, ca_base, patient_path, xcfg, tcfg)
        if args.stage == "incident_label":
            if args.inspect_cache:
                ident, mh, _ = current_identity(src, manifest, block)
                _inspect(inc_base, ident, f"  split manifest {mh[:12]}")
                return 0
            patients = pd.read_csv(patient_path, usecols=["patientunitstayid", "hospitalid", "unitdischargeoffset", "unitdischargestatus"])
            res = build_incident_sepsis_label(src, patients, manifest, block, inc_base, run_id=args.run_id, force=args.force_recompute)
            f = res["report"]["flow"]
            print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "initial_eligible_cohort": f["initial_eligible_cohort"],
                              "excluded_prevalent": f["excluded_prevalent"], "censored_or_excluded_followup": f["followup"]["total_excluded_or_censored"],
                              "incident_positive": f["incident_positive"], "incident_negative": f["incident_negative"]}, indent=2))
            return 0
        from src.three_modality import tcd_features as tf
        fident, mh, _ = tf.tcd_features_identity(src, manifest, tcfg)
        if args.stage == "tcd_features":
            if args.inspect_cache:
                _inspect(feat_base, fident, f"  split manifest {mh[:12]}")
                return 0
            patients = pd.read_csv(patient_path, usecols=["patientunitstayid", "hospitalid", "unitdischargeoffset"])
            res = tf.build_tcd_features(src, patients, manifest, tcfg, feat_base, run_id=args.run_id, force=args.force_recompute)
            m = res["manifest"]
            print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "patients": m["counts"], "shapes": m["shapes"],
                              "cutoff_audit_passed": m.get("cutoff_audit_passed")}, indent=2))
            return 0
        if args.stage in ("tabular_inputs", "tabular_ae"):
            from src.three_modality import tabular_inputs as ti
            fr = _pick_run(feat_base, fident, args.feature_run_id, "tcd_features", "Run --stage tcd_features first.")
            fman = json.loads((fr / "manifest.json").read_text(encoding="utf-8"))
            tident = ti.tabular_inputs_identity(src, fman, acfg, tcfg)
            if args.stage == "tabular_inputs":
                if args.inspect_cache:
                    _inspect(tin_base, tident, f"  (tcd features {fr.name})")
                    return 0
                res = ti.build_tabular_inputs(src, pd.read_csv(patient_path), fr, acfg, tcfg, tin_base, run_id=args.run_id, force=args.force_recompute)
                m = res["manifest"]
                print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "patients": m["counts"], "shape": m["shape"], "kept": m["kept"]}, indent=2))
                return 0
            from src.three_modality import tabular_ae as ta
            iident, _, _ = current_identity(src, manifest, block)
            xr = _pick_run(tin_base, tident, None, "tabular_inputs", "Run --stage tabular_inputs first.")
            ir = _pick_run(inc_base, iident, args.incident_run_id, "incident_label", "Run --stage incident_label first (same censoring policy), or pass --incident-run-id.")
            if args.inspect_cache:
                eident = ta.tabular_ae_identity(json.loads((xr / "manifest.json").read_text()), json.loads((ir / "manifest.json").read_text()), acfg, tcfg["window_minutes"])
                _inspect(tae_base, eident, f"  (inputs {xr.name}, incident label {ir.name})")
                return 0
            patients = pd.read_csv(patient_path, usecols=["patientunitstayid", "hospitaldischargestatus", "unitdischargeoffset"])
            res = ta.build_tabular_ae(xr, ir, patients, acfg, tcfg, tae_base, run_id=args.run_id, force=args.force_recompute)
            m = res["manifest"]
            print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "embedding_shape": m.get("embedding_shape"), "best_epoch": m.get("best_epoch"),
                              "inputs_run": m.get("inputs_run"), "incident_label_run": m.get("incident_label_run")}, indent=2))
            return 0
        if args.stage in ("gru_inputs", "gru_encoder"):
            from src.three_modality import gru_inputs as gi
            fr = _pick_run(feat_base, fident, args.feature_run_id, "tcd_features", "Run --stage tcd_features first.")
            fman = json.loads((fr / "manifest.json").read_text(encoding="utf-8"))
            xident = gi.gru_inputs_identity(fman, gcfg, tcfg)
            if args.stage == "gru_inputs":
                if args.inspect_cache:
                    _inspect(gin_base, xident, f"  (tcd features {fr.name})")
                    return 0
                res = gi.build_gru_inputs(src, fr, gcfg, tcfg, gin_base, run_id=args.run_id, force=args.force_recompute)
                m = res["manifest"]
                print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "patients": m["counts"], "shapes": m["shapes"],
                                  "cutoff_audit_passed": m.get("cutoff_audit_passed")}, indent=2))
                return 0
            from src.three_modality import gru_encoder as ge
            iident, _, _ = current_identity(src, manifest, block)
            xr = _pick_run(gin_base, xident, None, "gru_inputs", "Run --stage gru_inputs first.")
            ir = _pick_run(inc_base, iident, args.incident_run_id, "incident_label", "Run --stage incident_label first (same censoring policy), or pass --incident-run-id.")
            if args.inspect_cache:
                eident = ge.gru_encoder_identity(json.loads((xr / "manifest.json").read_text()), json.loads((ir / "manifest.json").read_text()), gcfg, tcfg["window_minutes"])
                _inspect(genc_base, eident, f"  (inputs {xr.name}, incident label {ir.name})")
                return 0
            patients = pd.read_csv(patient_path, usecols=["patientunitstayid", "hospitaldischargestatus", "unitdischargeoffset"])
            res = ge.build_gru_encoder(xr, ir, patients, gcfg, tcfg, genc_base, run_id=args.run_id, force=args.force_recompute)
            m = res["manifest"]
            print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "embedding_shape": m.get("embedding_shape"), "best_epoch": m.get("best_epoch"),
                              "inputs_run": m.get("inputs_run"), "incident_label_run": m.get("incident_label_run")}, indent=2))
            return 0
        # ---- tcd_encoder ----
        from src.three_modality import tcd_encoder as te
        iident, _, _ = current_identity(src, manifest, block)
        if args.inspect_cache:
            try:
                fr = _pick_run(feat_base, fident, args.feature_run_id, "tcd_features", "")
                ir = _pick_run(inc_base, iident, args.incident_run_id, "incident_label", "")
                eident = te.tcd_encoder_identity(json.loads((fr / "manifest.json").read_text()), json.loads((ir / "manifest.json").read_text()), tcfg)
                _inspect(enc_base, eident, f"  (features {fr.name}, incident label {ir.name})")
            except FileNotFoundError as exc:
                print(f"cannot compute the current encoder identity: {exc}")
            return 0
        fr = _pick_run(feat_base, fident, args.feature_run_id, "tcd_features", "Run --stage tcd_features first.")
        ir = _pick_run(inc_base, iident, args.incident_run_id, "incident_label", "Run --stage incident_label first (same censoring policy), or pass --incident-run-id.")
        patients = pd.read_csv(patient_path, usecols=["patientunitstayid", "hospitaldischargestatus", "unitdischargeoffset"])
        res = te.build_tcd_encoder(fr, ir, patients, tcfg, enc_base, run_id=args.run_id, force=args.force_recompute)
        m = res["manifest"]
        print(json.dumps({"run_id": res["run_id"], "run_dir": res["run_dir"], "cache_hit": res["cache_hit"], "embedding_shape": m.get("embedding_shape"), "best_epoch": m.get("best_epoch"),
                          "feature_run": m.get("feature_run"), "incident_label_run": m.get("incident_label_run")}, indent=2))
    finally:
        close_logging()
    return 0


if __name__ == "__main__":
    sys.exit(main())
