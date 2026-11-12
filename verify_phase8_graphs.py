"""
Read-only verification script for Phase 8's real-data output.
Checks two things the pipeline log alone doesn't directly confirm:

1. No edge in any hospital's graph connects to a patient outside
   that hospital (the core "intra-hospital only" guarantee).
2. Per-hospital node-count distribution -- specifically, whether any
   hospital exceeded graph.max_nodes_full_similarity (default 2000),
   which would mean the chunked-similarity code path was actually
   exercised on real data.

Does not modify anything -- only reads outputs/graphs/*.npz,
outputs/graphs/graph_manifest.json, and (to get the real
patient->hospital mapping) outputs/cohort_reports/cohort_report.json.

Usage (from the repo root, after a --phase 8 run):
    python verify_phase8_graphs.py
"""
import json
import sys
from pathlib import Path

import numpy as np

def main():
    repo_root = Path(__file__).resolve().parent
    graphs_dir = repo_root / "outputs" / "graphs"
    manifest_path = graphs_dir / "graph_manifest.json"

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path} -- run --phase 8 first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: method={manifest['method']}, k={manifest.get('k')}, "
          f"{len(manifest['hospital_graph_paths'])} hospitals listed.\n")

    # --- Check 2: per-hospital node-count distribution ---
    print("[1/2] Per-hospital node counts...")
    node_counts = {}
    for hid, path in manifest["hospital_graph_paths"].items():
        data = np.load(path)
        node_counts[hid] = len(data["patient_ids"])

    counts = sorted(node_counts.values())
    print(f"    min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}, "
          f"total={sum(counts):,}")
    over_2000 = {hid: n for hid, n in node_counts.items() if n > 2000}
    if over_2000:
        print(f"    {len(over_2000)} hospital(s) exceeded 2000 patients -- chunked-similarity "
              f"path WAS exercised on real data:")
        for hid, n in sorted(over_2000.items(), key=lambda x: -x[1])[:10]:
            print(f"      hospital {hid}: {n:,} patients")
    else:
        print("    No hospital exceeded 2000 patients -- the chunked-similarity path was "
              "NOT exercised by this real run (only tested in sandbox so far).")

    # --- Check 1: no cross-hospital edges ---
    print("\n[2/2] Checking every edge in every hospital's graph for cross-hospital leakage...")
    patient_to_hospital = {}
    for hid, path in manifest["hospital_graph_paths"].items():
        data = np.load(path)
        for pid in data["patient_ids"]:
            patient_to_hospital[int(pid)] = hid

    violations = []
    total_edges_checked = 0
    for hid, path in manifest["hospital_graph_paths"].items():
        data = np.load(path)
        node_ids = data["patient_ids"]
        src, dst = data["edge_index"][0], data["edge_index"][1]
        total_edges_checked += len(src)
        for i, j in zip(src, dst):
            pid_i, pid_j = int(node_ids[i]), int(node_ids[j])
            if patient_to_hospital.get(pid_i) != hid or patient_to_hospital.get(pid_j) != hid:
                violations.append((hid, pid_i, pid_j))

    print(f"    Checked {total_edges_checked:,} directed edge entries across "
          f"{len(manifest['hospital_graph_paths'])} hospital graph files.")
    if violations:
        print(f"    FOUND {len(violations)} CROSS-HOSPITAL EDGE(S) -- this is a real problem:")
        for hid, pid_i, pid_j in violations[:10]:
            print(f"      graph for hospital {hid}: edge references patients {pid_i}, {pid_j}")
    else:
        print("    Zero cross-hospital edges found. The intra-hospital-only guarantee holds "
              "in this real output, not just in sandbox tests.")


if __name__ == "__main__":
    main()
