# import csv
# from src.ablation.graph_comparison_runner import select_best_flat_config_from_sampling_results

# with open("outputs/ablation/run_20260928T082030Z/sampling_results.csv", newline="") as f:
#     rows = list(csv.DictReader(f))

# best = select_best_flat_config_from_sampling_results(rows, task="mortality")
# print(best)
# {'ts_model': ..., 'tab_model': ..., 'fusion_mode': ..., 'source_classifier': ..., 'source_sampling_strategy': ..., ...}
import pandas as pd
d = pd.read_csv(r"outputs\three_modality\incident_sepsis\inc_20261030T102343Z_9f777dd6\incident_sepsis_labels.csv")
print(d[d.status == "incident_positive"].event_hour.value_counts().sort_index().head(12))
