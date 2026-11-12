# Text Modality Decision

## What the architecture doc says

The source architecture (`Federated_Multimodal_Graph_Learning_Framework.pdf`,
Step 1) lists:

> Unstructured data: Clinical notes (if available) or other free-text
> information.
> Text Encoder — Model: MiniBERT (or ClinicalBERT if available) — Input:
> Clinical notes

The "(if available)" qualifier is already in the source document. It is
not available here.

## What was checked

- The 12 tables confirmed/plausible for this project (see
  `schema_registry.py`) contain **no free-text clinical notes**. eICU's
  public release famously omits nursing/physician notes (unlike
  MIMIC-III's `NOTEEVENTS`), for de-identification reasons.
- None of this project's 10 reference PDFs describe using real clinical
  text with eICU either — grepped for "clinical note", "free-text",
  "nursingnote", "note text" across all of them, zero hits.
- The closest things to text in this dataset are short, semi-structured
  strings: `diagnosis.diagnosisstring` (pipe-delimited hierarchical
  category, e.g. `"cardiovascular|chest pain / ASHD|coronary artery
  disease"`) and (if confirmed available) `treatment.treatmentstring`
  in the same format. These are categorical taxonomies, not prose.

## Decision (per master-spec Section 4)

Per the master spec's own contingency ("implement the modality interface
so the text branch can be disabled/configured... clearly document the
limitation"):

- **Default: `modalities.text.enabled: false`.** The text branch is
  structurally present (interface exists so Phase 7 fusion doesn't need
  a special case) but does not run.
- **Optional pseudo-text mode:** `modalities.text.enabled: true` runs a
  MiniBERT-scale encoder (`prajjwal1/bert-tiny` is a genuine, tiny BERT
  — not a stand-in name) over `diagnosis.diagnosisstring` (and
  `treatment.treatmentstring` if available), tokenized as short category
  phrases. This must be labeled everywhere in outputs/reports as
  **"diagnosis-string pseudo-text," not "clinical notes"** — reusing the
  word "text" for a categorical taxonomy would misrepresent what the
  model is actually seeing, which matters for a research write-up.
- The ablation matrix (Section 23: time-series only / tabular only /
  text only / combinations) still runs correctly with text disabled —
  those variants that include "text" should be understood as "text
  (diagnosis-string pseudo-text)" in any results table, with a footnote,
  rather than silently implying clinical notes were used.

**This default is a reasonable call, not a locked one** — flagged in the
main chat response for confirmation before Phase 6 is actually built,
since it does change what "multimodal" means for this specific project.
