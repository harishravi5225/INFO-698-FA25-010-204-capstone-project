# run.py
"""
Comprehensive ABM runner with required metrics.
Runs parameter sweeps and generates detailed analysis.

Key features:
- Multiple nest sizes (20, 30, 50, 80)
- Multiple scenarios (forager ratios, wasp-to-cell ratios)
- Route efficiency comparisons (observed vs greedy vs random vs biased vs TSP)
- Feeding outcome metrics (unfed, multi-fed, center vs periphery)
- Error tracking (hungry-ignored, redundant feeds)
- Full behavioral audit logs
"""

import os
import sys

# Quick option: run single test or full analysis
# Usage: python run.py [quick|full]

if len(sys.argv) > 1 and sys.argv[1] == "quick":
    print("Running quick test (1 nest size, baseline scenario)...")
    from model import WaspModel
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    
    OUT = "output_abm"
    os.makedirs(OUT, exist_ok=True)
    
    m = WaspModel(nest_size=20, model_seed=42)
    print(f"Model created: {len(m.agents)} agents, {len(m.larvae)} larvae")
    m.run(steps=400)
    
    print(f"\nResults:")
    print(f"  Total feeds: {sum(c.feed_count for c in m.larvae)}")
    print(f"  Gini: {m.compute_gini():.4f}")
    print(f"  Unfed larvae: {sum(1 for c in m.larvae if c.feed_count == 0)}")
    print(f"  Center avg feeds: {m.compute_center_vs_periphery_feeding()['center_avg']:.2f}")
    print(f"  Periphery avg feeds: {m.compute_center_vs_periphery_feeding()['periphery_avg']:.2f}")
    print(f"  Bouts completed: {len(m.bouts_log)}")
    
    # Quick save
    df_cells = pd.DataFrame([
        {
            "cell_id": c.cell_id, "x": c.pos[0], "y": c.pos[1],
            "feeds": c.feed_count, "distance_from_center": c.distance_from_center
        }
        for c in m.larvae
    ])
    df_cells.to_csv(f"{OUT}/quick_test_larval_counts.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df_cells.x, df_cells.y, c=df_cells.feeds, cmap="magma", s=100, edgecolor="black")
    plt.colorbar(label="Feeding count")
    plt.title("Quick Test: Feeding Heatmap (nest=20)")
    plt.savefig(f"{OUT}/quick_test_heatmap.png", dpi=150)
    plt.close()
    
    print(f"\nQuick test outputs saved to {OUT}/")
else:
    print("Running comprehensive parameter sweep analysis...")
    print("This will test multiple nest sizes and scenarios.\n")
    from analysis import parameter_sweep_analysis, create_summary_report
    
    all_metrics, all_routes = parameter_sweep_analysis()
    create_summary_report(all_metrics, all_routes)
