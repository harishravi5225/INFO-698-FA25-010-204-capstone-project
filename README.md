
Final submission package
========================

Included files:
- agent.py
- model.py
- run.py

Simulation outputs (from earlier runs) included from folder 'output_abm/':
baseline_agents_summary.csv, baseline_feed_events.csv, baseline_heatmap.png, baseline_larval_counts.csv, baseline_transfers.csv, gini_vs_foragers.png, latency_vs_foragers.png, param_sweep_results.csv, route_comparison_baseline.csv, route_perms_baseline.png

Optional earlier zip artifacts included if present:
- output_abm_all_results.zip
- full_submission_package.zip

How to run
----------
1. Place the CSV data files in the same folder as these scripts (examples used in the project):
   - ED_FL_3nests1noC2.csv
   - ALL_FL_minmaj_final3noC2.csv
  

2. Install required Python packages (recommended):
   pip install numpy pandas matplotlib seaborn mesa

3. Run the main script:
   python run.py

   This will create/overwrite the folder 'output_abm/' with CSV and PNG outputs:
   - baseline_feed_events.csv
   - baseline_larval_counts.csv
   - baseline_heatmap.png
   - param_sweep_results.csv
   - route_comparison.csv

Notes
-----
- The three Python files are the modular implementation of the ABM (agent.py, model.py, run.py).
- The model reads CSVs by filename; do not include absolute paths in the scripts.
- If you want the full single-file script or publication-ready figures, tell me and I will add them.


# INFO-698-FA25-010-204-capstone-project
