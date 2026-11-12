# Data Setup

This repository never bundles eICU data. You keep the CSV files on your
own machine and point the pipeline at them via an environment variable.

## 1. Locate your eICU CSVs

You should have (at minimum) these files somewhere on disk, from the
eICU Collaborative Research Database export:

```
patient.csv
hospital.csv
apachePatientResult.csv
apachePredVar.csv
diagnosis.csv
lab.csv
medication.csv
admissionDrug.csv
infusionDrug.csv
microLab.csv
```

Optionally (auto-detected if present, ignored if not):
```
medication_imputed.csv
medication_imputed1.csv
treatment.csv
```

File names are matched case-insensitively and `.csv.gz` is also
supported — see `src/data/ingestion/discovery.py`.

## 2. Point the pipeline at them

macOS / Linux:
```bash
export EICU_DATA_ROOT=/absolute/path/to/your/eicu/csvs
```

Windows (PowerShell):
```powershell
$env:EICU_DATA_ROOT = "C:\absolute\path\to\your\eicu\csvs"
```

Windows (cmd.exe):
```cmd
set EICU_DATA_ROOT=C:\absolute\path\to\your\eicu\csvs
```

## 3. Run Phase 1 to validate

```bash
python run_pipeline.py --config configs/debug.yaml
```

This reads only what's needed to validate schema and produce
`outputs/data_reports/` — it does not load the multi-GB tables (`lab`,
`medication`, `diagnosis`, `infusionDrug`) fully into memory; see
`src/data/ingestion/data_report.py` for the chunked-reading strategy.

## Testing without real data

`scripts/generate_mock_eicu.py` creates a tiny synthetic dataset with the
correct schema so you (or CI) can exercise the pipeline without the real
eICU files:

```bash
python scripts/generate_mock_eicu.py --out /tmp/mock_eicu --n-hospitals 3 --n-patients-per-hospital 20
export EICU_DATA_ROOT=/tmp/mock_eicu
python run_pipeline.py --config configs/debug.yaml
```
