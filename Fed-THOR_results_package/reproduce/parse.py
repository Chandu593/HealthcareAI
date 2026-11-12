import re, json
from pathlib import Path
ROW = re.compile(r"([0-9.]+)±([0-9.]+) \[([0-9.]+), ([0-9.]+)\]")
def na(x): return None if x.strip() in ("n/a", "") else float(x)
def _default():
    here = Path(__file__).resolve()
    for c in (here.parents[2] / "outputs" / "paper" / "summary_all_runs.md", Path("/mnt/user-data/uploads/summary_all_runs.md")):
        if c.exists(): return c
    raise FileNotFoundError("summary_all_runs.md not found; run scripts/paper_collect_results.py first")
def load(path=None):
    path = path or _default()
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 13 or c[0] in ("run", "---"): continue
        m = ROW.match(c[5])
        if not m: continue
        rows[(c[0], c[1], c[2])] = dict(run=c[0], exp=c[1], arm=c[2], seeds=int(c[3]), feats=int(c[4]), auroc=float(m[1]), sd=float(m[2]), lo=float(m[3]), hi=float(m[4]),
                                        auprc=na(c[6]), perhosp=na(c[7]), local=na(c[8]), val=na(c[9]), brier=na(c[10]), ece=na(c[11]), n=int(c[12]))
    return rows
if __name__ == "__main__":
    R = load(); print(len(R), "rows")
    for k in [("final_mortality_strict","mortality/no_apache_scores","fedavg_mlp"), ("final_mortality_allstays","mortality/no_apache_scores","fedavg_mlp"),
              ("final_sepsis24","sepsis/sepsis_clean","fedavg_mlp"), ("final_sepsis6h","sepsis/sepsis_clean","lightgbm_central"), ("fed_r3b","mortality/no_apache_scores","fedavg_mlp")]:
        print(k, R.get(k))
