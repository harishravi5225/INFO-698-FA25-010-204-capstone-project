"""
Comprehensive analysis and metrics generation for the ABM model.
Implements all required metrics:
- Route efficiency comparisons
- Larval feeding outcomes
- Feeding rate vs distance
- Hungry vs ignored events
- Redundancy tracking
- Center vs periphery analysis
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model import WaspModel

OUT = "output_abm"
os.makedirs(OUT, exist_ok=True)

def run_single_simulation(nest_size, wasp_ratio=0.9, forager_frac=0.06, 
                          steps=600, seed=None, scenario_name="default"):
    """Run a single simulation and collect metrics"""
    print(f"\n{'='*60}")
    print(f"Running: nest_size={nest_size}, wasp_ratio={wasp_ratio}, forager_frac={forager_frac}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    m = WaspModel(
        nest_size=int(nest_size),
        wasp_to_cell_ratio=wasp_ratio,
        forager_frac=forager_frac,
        model_seed=seed
    )
    
    print(f"Agents created: {len(m.agents)}")
    print(f"Larvae: {len(m.larvae)}")
    print(f"Running {steps} steps...")
    m.run(steps=steps)
    
    return m

def analyze_single_run(model, scenario_name, nest_size):
    """Generate comprehensive metrics for a single run"""
    
    # Basic statistics
    total_feeds = sum(c.feed_count for c in model.larvae)
    gini = model.compute_gini()
    
    # Larval feeding outcomes
    unfed_count = sum(1 for c in model.larvae if c.feed_count == 0)
    multi_fed = sum(1 for c in model.larvae if c.feed_count > 1)
    
    # Center vs periphery
    center_periph = model.compute_center_vs_periphery_feeding()
    
    # Feeding by distance
    feeding_by_dist = model.compute_feeding_by_distance()
    
    # Error metrics
    hungry_ignored = len(model.hungry_ignored_events)
    redundant_feeds = len(model.redundant_feed_events)
    
    # Bouts analysis
    total_bouts = len(model.bouts_log)
    if total_bouts > 0:
        bouts_list = []
        for bout_entry in model.bouts_log:
            if isinstance(bout_entry, tuple):
                _, bout = bout_entry
            else:
                bout = bout_entry
            bouts_list.append(bout)
        
        avg_bout_duration = np.mean([b.get('end', model.model_time) - b.get('start', 0) 
                                      for b in bouts_list])
        avg_larvae_per_bout = np.mean([len(set(f[1] for f in b.get('feeds', []))) 
                                       for b in bouts_list])
    else:
        avg_bout_duration = 0
        avg_larvae_per_bout = 0
    
    metrics = {
        "scenario": scenario_name,
        "nest_size": nest_size,
        "total_agents": len(model.agents),
        "total_feeds": total_feeds,
        "avg_feeds_per_larva": total_feeds / len(model.larvae) if model.larvae else 0,
        "gini_coefficient": gini,
        "unfed_larvae": unfed_count,
        "multi_fed_larvae": multi_fed,
        "center_avg_feeds": center_periph.get("center_avg", 0),
        "periphery_avg_feeds": center_periph.get("periphery_avg", 0),
        "hungry_ignored_events": hungry_ignored,
        "redundant_feed_events": redundant_feeds,
        "total_bouts": total_bouts,
        "avg_bout_duration": avg_bout_duration,
        "avg_larvae_per_bout": avg_larvae_per_bout,
        "total_transfers": len(model.global_transfers),
        "model_time": model.model_time
    }
    
    return metrics, model

def compare_routes(model, scenario_name, nest_size):
    """Compare route efficiency across different strategies"""
    
    # Extract a representative feeding route from first bout
    if not model.bouts_log:
        return None
    
    # Handle both tuple and direct info formats
    first_bout_entry = model.bouts_log[0]
    if isinstance(first_bout_entry, tuple):
        _, first_bout = first_bout_entry
    else:
        first_bout = first_bout_entry
    
    observed_route = [f[1] for f in first_bout.get('feeds', [])]
    
    if len(observed_route) < 3:
        return None
    
    # Remove duplicates while preserving order
    unique_route = []
    for cell_id in observed_route:
        if cell_id not in unique_route:
            unique_route.append(cell_id)
    
    if len(unique_route) < 2:
        return None
    
    start_cell = unique_route[0]
    cell_list = unique_route
    
    # Compute different route strategies
    results = {"scenario": scenario_name, "nest_size": nest_size}
    
    # Observed route
    obs_eff = model.route_efficiency(cell_list)
    results["observed_path_length"] = obs_eff.get("path_length", 0)
    results["observed_direction_score"] = obs_eff.get("direction_score", 0)
    
    # Greedy nearest-neighbor
    greedy = model.greedy_route(start_cell, cell_list)
    greedy_eff = model.route_efficiency(greedy)
    results["greedy_path_length"] = greedy_eff.get("path_length", 0)
    results["greedy_direction_score"] = greedy_eff.get("direction_score", 0)
    
    # Random walk
    random_route = model.random_walk_route(start_cell, cell_list)
    random_eff = model.route_efficiency(random_route)
    results["random_path_length"] = random_eff.get("path_length", 0)
    
    # Biased random walk
    biased_route = model.biased_random_walk_route(start_cell, cell_list)
    biased_eff = model.route_efficiency(biased_route)
    results["biased_path_length"] = biased_eff.get("path_length", 0)
    results["biased_direction_score"] = biased_eff.get("direction_score", 0)
    
    # Compute TSP (optional)
    try:
        from python_tsp.heuristics import solve_tsp_local_search
        coords = {c.cell_id: c.pos for c in model.larvae}
        coords_list = [coords[cid] for cid in unique_route if cid in coords]
        
        if len(coords_list) > 2:
            n = len(coords_list)
            dist_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_mat[i, j] = np.hypot(coords_list[i][0] - coords_list[j][0],
                                              coords_list[i][1] - coords_list[j][1])
            perm, d = solve_tsp_local_search(dist_mat)
            results["tsp_path_length"] = float(d)
    except:
        results["tsp_path_length"] = None
    
    return results

def parameter_sweep_analysis():
    """Run comprehensive parameter sweep across nest sizes and configurations"""
    
    all_metrics = []
    all_routes = []
    
    # Test different nest sizes
    nest_sizes = [20, 30, 50, 80]
    scenarios = [
        {"name": "baseline", "ratio": 0.9, "forager": 0.06},
        {"name": "low_forager", "ratio": 0.9, "forager": 0.03},
        {"name": "high_forager", "ratio": 0.9, "forager": 0.10},
        {"name": "low_ratio", "ratio": 0.7, "forager": 0.06},
        {"name": "high_ratio", "ratio": 0.95, "forager": 0.06},
    ]
    
    for scenario in scenarios:
        for nest_size in nest_sizes:
            try:
                model = run_single_simulation(
                    nest_size=nest_size,
                    wasp_ratio=scenario["ratio"],
                    forager_frac=scenario["forager"],
                    steps=400,
                    seed=42,
                    scenario_name=scenario["name"]
                )
                
                # Collect metrics
                metrics, _ = analyze_single_run(model, scenario["name"], nest_size)
                all_metrics.append(metrics)
                
                # Collect route comparisons
                routes = compare_routes(model, scenario["name"], nest_size)
                if routes:
                    all_routes.append(routes)
                
                # Save individual run outputs
                save_run_outputs(model, scenario["name"], nest_size)
                
            except Exception as e:
                print(f"Error in scenario {scenario['name']}, nest_size {nest_size}: {e}")
                continue
    
    # Save summary files
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(f"{OUT}/metrics_summary.csv", index=False)
        print(f"\nSaved metrics summary to {OUT}/metrics_summary.csv")
    
    if all_routes:
        df_routes = pd.DataFrame(all_routes)
        df_routes.to_csv(f"{OUT}/route_comparison.csv", index=False)
        print(f"Saved route comparison to {OUT}/route_comparison.csv")
    
    return all_metrics, all_routes

def save_run_outputs(model, scenario_name, nest_size):
    """Save detailed outputs for a single run"""
    
    prefix = f"{OUT}/{scenario_name}_n{nest_size}"
    
    # Larval summary
    df_cells = pd.DataFrame([
        {
            "cell_id": c.cell_id,
            "x": c.pos[0],
            "y": c.pos[1],
            "feeds": c.feed_count,
            "last_fed": c.last_fed,
            "hunger": c.hunger_timer,
            "distance_from_center": c.distance_from_center,
            "is_periphery": c.cell_id in model.periphery_ids
        }
        for c in model.larvae
    ])
    df_cells.to_csv(f"{prefix}_larval_counts.csv", index=False)
    
    # Feed events with bout tracking
    if len(model.global_feed_events) > 0:
        df_feeds = pd.DataFrame(
            model.global_feed_events,
            columns=["wasp_id", "cell_id", "amt", "time", "bout_id"]
        )
        df_feeds.to_csv(f"{prefix}_feed_events.csv", index=False)
    
    # Transfers
    if len(model.global_transfers) > 0:
        df_trans = pd.DataFrame(
            model.global_transfers,
            columns=["giver", "receiver", "amt", "time", "bout_id"]
        )
        df_trans.to_csv(f"{prefix}_transfers.csv", index=False)
    
    # Bouts summary
    with open(f"{prefix}_bouts.json", "w") as f:
        json.dump(model.bouts_log, f, default=str, indent=2)
    
    # Efficiency metrics
    with open(f"{prefix}_metrics.json", "w") as f:
        metrics = model.compute_efficiency_metrics()
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    create_heatmap(model, scenario_name, nest_size)

def create_heatmap(model, scenario_name, nest_size):
    """Create feeding heatmap and other visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    cells_data = pd.DataFrame([
        {"x": c.pos[0], "y": c.pos[1], "feeds": c.feed_count, "dist": c.distance_from_center}
        for c in model.larvae
    ])
    
    # Feeding heatmap
    ax = axes[0, 0]
    scatter = ax.scatter(cells_data.x, cells_data.y, c=cells_data.feeds, cmap="magma", s=100)
    plt.colorbar(scatter, ax=ax, label="Feed count")
    ax.set_title(f"Feeding Heatmap ({scenario_name}, n={nest_size})")
    ax.set_aspect("equal")
    
    # Distance vs feeding
    ax = axes[0, 1]
    ax.scatter(cells_data.dist, cells_data.feeds, alpha=0.6, s=80)
    z = np.polyfit(cells_data.dist, cells_data.feeds, 1)
    p = np.poly1d(z)
    ax.plot(cells_data.dist.sort_values(), p(cells_data.dist.sort_values()), "r--", label="Trend")
    ax.set_xlabel("Distance from center")
    ax.set_ylabel("Feeding count")
    ax.set_title("Distance vs Feeding Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Center vs periphery comparison
    ax = axes[1, 0]
    center_vs_periph = model.compute_center_vs_periphery_feeding()
    categories = ["Center", "Periphery"]
    values = [center_vs_periph["center_avg"], center_vs_periph["periphery_avg"]]
    ax.bar(categories, values, color=["blue", "orange"], alpha=0.7)
    ax.set_ylabel("Average feeds per larva")
    ax.set_title("Center vs Periphery Feeding")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Feed distribution histogram
    ax = axes[1, 1]
    ax.hist(cells_data.feeds, bins=15, color="green", alpha=0.7, edgecolor="black")
    ax.axvline(cells_data.feeds.mean(), color="red", linestyle="--", label=f"Mean: {cells_data.feeds.mean():.2f}")
    ax.set_xlabel("Feed count per larva")
    ax.set_ylabel("Frequency")
    ax.set_title("Feed Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(f"{OUT}/{scenario_name}_n{nest_size}_analysis.png", dpi=150)
    plt.close()
    print(f"Saved visualization to {OUT}/{scenario_name}_n{nest_size}_analysis.png")

def create_summary_report(all_metrics, all_routes):
    """Create a comprehensive summary report"""
    
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE ABM ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Metrics summary
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        report.append("KEY FINDINGS - METRICS SUMMARY")
        report.append("-"*80)
        report.append(f"Total simulations: {len(df)}")
        report.append(f"Average Gini coefficient: {df['gini_coefficient'].mean():.4f}")
        report.append(f"Average unfed larvae: {df['unfed_larvae'].mean():.2f}")
        report.append(f"Average hungry-ignored events: {df['hungry_ignored_events'].mean():.2f}")
        report.append(f"Average redundant feeds: {df['redundant_feed_events'].mean():.2f}")
        report.append("")
        
        # By scenario
        report.append("Results by Scenario:")
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            report.append(f"\n  {scenario.upper()}:")
            report.append(f"    Nest sizes: {sorted(scenario_data['nest_size'].unique())}")
            report.append(f"    Avg feeds/larva: {scenario_data['avg_feeds_per_larva'].mean():.2f}")
            report.append(f"    Avg Gini: {scenario_data['gini_coefficient'].mean():.4f}")
    
    # Route comparison
    if all_routes:
        report.append("\n" + "="*80)
        report.append("ROUTE EFFICIENCY COMPARISON")
        report.append("-"*80)
        df_routes = pd.DataFrame(all_routes)
        
        if 'observed_path_length' in df_routes.columns and 'greedy_path_length' in df_routes.columns:
            report.append(f"Observed vs Greedy efficiency:")
            report.append(f"  Avg observed path: {df_routes['observed_path_length'].mean():.2f}")
            report.append(f"  Avg greedy path: {df_routes['greedy_path_length'].mean():.2f}")
            report.append(f"  Avg random path: {df_routes['random_path_length'].mean():.2f}")
            report.append(f"  Avg biased path: {df_routes['biased_path_length'].mean():.2f}")
            
            if 'tsp_path_length' in df_routes.columns:
                tsp_vals = df_routes['tsp_path_length'].dropna()
                if len(tsp_vals) > 0:
                    report.append(f"  Avg TSP optimal: {tsp_vals.mean():.2f}")
    
    report.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open(f"{OUT}/analysis_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to {OUT}/analysis_report.txt")

if __name__ == "__main__":
    print("Starting comprehensive ABM analysis...")
    all_metrics, all_routes = parameter_sweep_analysis()
    create_summary_report(all_metrics, all_routes)
    print(f"\nAll outputs saved to {OUT}/")
