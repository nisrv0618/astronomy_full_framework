# Full-dataset Astronomy Triage Framework

This research prototype processes all rows in `processed.parquet` in batches for profiling and inference. Model fitting uses a capped sample of known-class rows to manage runtime and memory; this is disclosed in run metadata and the report. The dataset is simulated and results are not evidence of production readiness for Rubin Observatory.

## Windows / VS Code
1. Extract this folder.
2. Copy `processed.parquet` beside `astronomy_pipeline.py`.
3. Open the folder in VS Code (`File > Open Folder`).
4. Open `Terminal > New Terminal`.
5. Install dependencies:

```powershell
python -m pip install pandas numpy pyarrow scikit-learn matplotlib seaborn pillow
```

6. Run:

```powershell
python astronomy_pipeline.py
```

Outputs are written to `astronomy_full_outputs/`. The script checks that the full source row count matches the number of rows scanned and scored. It can take time and disk space; the row-level outputs are compressed CSVs.

## Main outputs
- `all_observation_results.csv.gz`: one row per source observation, including scores and route.
- `pruning_history.csv.gz`: stage-wise history for every observation.
- `noise_assessment_results.csv`: quality-flagged records retained for review.
- `human_review_queue.csv`: immediate review items.
- `human_review_template.csv`: editable reviewer decisions (`RELEVANT`, `NOISE`, `UNCERTAIN`, `REVISIT`, or `PENDING`).
- `metrics.json`, `results_report.txt`, `run_metadata.json`.
- `figures/`: separate PNGs; `astronomy_dashboard.png`: combined dashboard.

## Retrieve one observation
Use an ID from `all_observation_results.csv.gz`:

```powershell
python retrieve_history.py ASTRO_000000001
```

This exports that observation's stage history to a CSV.

## Import human decisions
Edit `review_decision` and `reviewer_notes` in `astronomy_full_outputs/human_review_template.csv`, then run:

```powershell
python import_human_decisions.py
```

The importer validates decisions and saves `human_decisions.csv`. It does not pretend that pending rows have been reviewed.

## Method caveats
- `class` is used only for retrospective evaluation and plots; it is not an input feature.
- Existing `pred_*` and `iforest_output` columns are not used as model features.
- Isolation Forest scores are not probabilities. The held-out class is a controlled simulation of unknown-class evaluation, not a real-world unknown-label guarantee.
- Quality rules flag excessive feature missingness, non-finite values, and duplicate feature vectors. They do not establish that an observation is astrophysical noise. Flagged rows are retained in outputs.
- Novelty percentile scores are relative ranks over the full scored dataset. The review queue selects the top 25% of non-quality-flagged OOD candidates by priority, configurable in the script.

## Dataset Attribution

This project uses the simulated LSST light-curve dataset published by:

Crispim Romão, M., Croon, D., and Godines, D. (2025). *LSST light curves for constant and variable sources, and for point-like and extended objects microlensing.*

- **Dataset repository:** https://zenodo.org/records/15005108
- **DOI:** https://doi.org/10.5281/zenodo.15005108
- **Associated research paper:** [Anomaly Detection to Identify Transients in LSST Time Series Data](https://arxiv.org/abs/2503.09699)

The original dataset contains approximately 600,000 simulated LSST light curves across six astronomical classes. This project uses a processed dataset containing 597,013 observations.

The original dataset and its scientific contributions belong to their respective authors. The processing pipeline, analysis workflow, and project implementation are documented in this repository.