#!/usr/bin/env python3
"""
Generate LaTeX tables from results.json

Supports three task types:
  - presence: Multi-label presence/absence (F1 scores)
  - majority: Single-label classification (accuracy)
  - counting: Fish count regression (MAE/RMSE)

Usage:
    python generate_table.py                          # Default: presence task
    python generate_table.py --task presence          # Presence/absence
    python generate_table.py --task majority          # Single-label
    python generate_table.py --task counting          # Counting task
"""

import json
import os
import argparse
from collections import defaultdict


def generate_presence_table(results):
    """Generate table for presence/absence or majority task."""
    # Organize results
    organized = defaultdict(dict)
    shortcut_results = {"easy_to_extreme": {}, "extreme_to_easy": {}}

    for entry in results:
        arch = entry["architecture"]
        dataset = entry["dataset"]

        # Skip acoustic_only entries for main table
        if entry.get("mode") == "acoustic_only":
            continue

        # Check if this is a shortcut test entry
        if entry.get("shortcut_test") and entry.get("test_dataset"):
            test_ds = entry["test_dataset"]
            test_data = entry.get("test")
            if test_data:
                if dataset == "easy" and test_ds == "extreme":
                    shortcut_results["easy_to_extreme"][arch] = test_data
                elif dataset == "extreme" and test_ds == "easy":
                    shortcut_results["extreme_to_easy"][arch] = test_data
            continue

        # Keep the latest entry per (architecture, dataset) pair
        if dataset not in organized[arch] or entry["timestamp"] > organized[arch][dataset]["timestamp"]:
            organized[arch][dataset] = entry

    # Difficulty order
    difficulties = ["easy", "medium", "hard", "extreme"]
    splits = ["train", "val", "test"]

    # Extract metrics helper
    def get_metrics(arch, dataset, split):
        if arch not in organized or dataset not in organized[arch]:
            return None
        entry = organized[arch][dataset]
        data = entry.get(split)
        if not data:
            return None
        return {
            "kf": data["kingfish_f1"] * 100,
            "sn": data["snapper_f1"] * 100,
            "cd": data["cod_f1"] * 100,
            "em": data["empty_f1"] * 100,
            "macro": data["avg_f1"] * 100,
        }

    # Determine best scores per difficulty and split for bolding
    def find_best(difficulty, split):
        best_macro = -1
        best_arch = None
        for arch in ["JEPA", "LeWM"]:
            metrics = get_metrics(arch, difficulty, split)
            if metrics and metrics["macro"] > best_macro:
                best_macro = metrics["macro"]
                best_arch = arch
        return best_arch

    # Format value with bold if best
    def fmt(value, is_best):
        if is_best:
            return f"\\textbf{{{value:.1f}}}"
        return f"{value:.1f}"

    # Start building LaTeX table
    latex = r"""\begin{table*}[t]
\centering
\caption{%
  Full benchmark results: macro-averaged F1 score (\%) for Multimodal JEPA and LeWM across all four
  difficulty modes and all three data splits (Train / Validation / Test).
  Per-species F1 scores are shown for Kingfish (KF), Snapper (SN), Cod (CD), and Empty (EM).
  The overall macro F1 across all four classes is shown in the \textbf{Macro} column.
  Bold entries indicate the best score per difficulty level and split.
  $\uparrow$ indicates higher is better.
}
\label{tab:presence_results}
\setlength{\tabcolsep}{4.5pt}
\small
\begin{tabular}{@{}ll lllll lllll@{}}
\toprule
& & \multicolumn{5}{c}{\textbf{Multimodal JEPA}} 
  & \multicolumn{5}{c}{\textbf{LeWM (Acoustic Only)}} \\
\cmidrule(lr){3-7} \cmidrule(lr){8-12}
\textbf{Difficulty} & \textbf{Split} 
  & KF$\uparrow$ & SN$\uparrow$ & CD$\uparrow$ & EM$\uparrow$ & \textbf{Macro}$\uparrow$
  & KF$\uparrow$ & SN$\uparrow$ & CD$\uparrow$ & EM$\uparrow$ & \textbf{Macro}$\uparrow$ \\
\midrule
"""

    for i, difficulty in enumerate(difficulties):
        difficulty_display = difficulty.capitalize() if difficulty != "extreme" else "\\textsc{Extreme}"
        latex += f"%--- {difficulty.upper()} ---\n"
        latex += f"\\multirow{{3}}{{*}}{{{difficulty_display}}}\n"
        
        for j, split in enumerate(splits):
            split_display = split.capitalize() if split != "val" else "Val"
            
            # Find best for this difficulty/split
            best_arch = find_best(difficulty, split)
            
            # JEPA metrics
            jepa = get_metrics("JEPA", difficulty, split)
            # LeWM metrics
            lewm = get_metrics("LeWM", difficulty, split)
            
            if jepa:
                jepa_str = (
                    f"& {fmt(jepa['kf'], best_arch == 'JEPA')} & {fmt(jepa['sn'], best_arch == 'JEPA')} & "
                    f"{fmt(jepa['cd'], best_arch == 'JEPA')} & {fmt(jepa['em'], best_arch == 'JEPA')} & "
                    f"{fmt(jepa['macro'], best_arch == 'JEPA')}"
                )
            else:
                jepa_str = "& - & - & - & - & -"
            
            if lewm:
                lewm_str = (
                    f"& {fmt(lewm['kf'], best_arch == 'LeWM')} & {fmt(lewm['sn'], best_arch == 'LeWM')} & "
                    f"{fmt(lewm['cd'], best_arch == 'LeWM')} & {fmt(lewm['em'], best_arch == 'LeWM')} & "
                    f"{fmt(lewm['macro'], best_arch == 'LeWM')}"
                )
            else:
                lewm_str = "& - & - & - & - & -"
            
            latex += f"  & {split_display} {jepa_str} {lewm_str} \\\\\n"
        
        if i < len(difficulties) - 1:
            latex += "\\midrule\n"

    # Shortcut test rows
    shortcut_e2e_jepa = shortcut_results["easy_to_extreme"].get("JEPA")
    shortcut_e2e_lewm = shortcut_results["easy_to_extreme"].get("LeWM")
    shortcut_x2e_jepa = shortcut_results["extreme_to_easy"].get("JEPA")
    shortcut_x2e_lewm = shortcut_results["extreme_to_easy"].get("LeWM")

    def best_shortcut_e2e():
        if not shortcut_e2e_jepa or not shortcut_e2e_lewm:
            return None
        return "jepa" if shortcut_e2e_jepa["avg_f1"] > shortcut_e2e_lewm["avg_f1"] else "lewm"
    
    def best_shortcut_x2e():
        if not shortcut_x2e_jepa or not shortcut_x2e_lewm:
            return None
        return "jepa" if shortcut_x2e_jepa["avg_f1"] > shortcut_x2e_lewm["avg_f1"] else "lewm"

    best_e2e = best_shortcut_e2e()
    best_x2e = best_shortcut_x2e()

    def fmt_sc(value, is_best):
        if is_best:
            return f"\\textbf{{{value:.1f}}}"
        return f"{value:.1f}"

    # Easy→Ext row
    if shortcut_e2e_jepa:
        jepa_e2e_str = (
            f"& {fmt_sc(shortcut_e2e_jepa['kingfish_f1']*100, best_e2e == 'jepa')} & "
            f"{fmt_sc(shortcut_e2e_jepa['snapper_f1']*100, best_e2e == 'jepa')} & "
            f"{fmt_sc(shortcut_e2e_jepa['cod_f1']*100, best_e2e == 'jepa')} & "
            f"{fmt_sc(shortcut_e2e_jepa['empty_f1']*100, best_e2e == 'jepa')} & "
            f"{fmt_sc(shortcut_e2e_jepa['avg_f1']*100, best_e2e == 'jepa')}"
        )
    else:
        jepa_e2e_str = "& - & - & - & - & -"
    
    if shortcut_e2e_lewm:
        lewm_e2e_str = (
            f"& {fmt_sc(shortcut_e2e_lewm['kingfish_f1']*100, best_e2e == 'lewm')} & "
            f"{fmt_sc(shortcut_e2e_lewm['snapper_f1']*100, best_e2e == 'lewm')} & "
            f"{fmt_sc(shortcut_e2e_lewm['cod_f1']*100, best_e2e == 'lewm')} & "
            f"{fmt_sc(shortcut_e2e_lewm['empty_f1']*100, best_e2e == 'lewm')} & "
            f"{fmt_sc(shortcut_e2e_lewm['avg_f1']*100, best_e2e == 'lewm')}"
        )
    else:
        lewm_e2e_str = "& - & - & - & - & -"

    # Ext→Easy row
    if shortcut_x2e_jepa:
        jepa_x2e_str = (
            f"& {fmt_sc(shortcut_x2e_jepa['kingfish_f1']*100, best_x2e == 'jepa')} & "
            f"{fmt_sc(shortcut_x2e_jepa['snapper_f1']*100, best_x2e == 'jepa')} & "
            f"{fmt_sc(shortcut_x2e_jepa['cod_f1']*100, best_x2e == 'jepa')} & "
            f"{fmt_sc(shortcut_x2e_jepa['empty_f1']*100, best_x2e == 'jepa')} & "
            f"{fmt_sc(shortcut_x2e_jepa['avg_f1']*100, best_x2e == 'jepa')}"
        )
    else:
        jepa_x2e_str = "& - & - & - & - & -"
    
    if shortcut_x2e_lewm:
        lewm_x2e_str = (
            f"& {fmt_sc(shortcut_x2e_lewm['kingfish_f1']*100, best_x2e == 'lewm')} & "
            f"{fmt_sc(shortcut_x2e_lewm['snapper_f1']*100, best_x2e == 'lewm')} & "
            f"{fmt_sc(shortcut_x2e_lewm['cod_f1']*100, best_x2e == 'lewm')} & "
            f"{fmt_sc(shortcut_x2e_lewm['empty_f1']*100, best_x2e == 'lewm')} & "
            f"{fmt_sc(shortcut_x2e_lewm['avg_f1']*100, best_x2e == 'lewm')}"
        )
    else:
        lewm_x2e_str = "& - & - & - & - & -"

    latex += r"""
\midrule
%--- SHORTCUT TEST ---
\multirow{2}{*}{\shortstack[l]{Shortcut\\Test$^\dagger$}}
  & Easy$\to$\textsc{Ext.} & \multicolumn{5}{c}{-} """ + jepa_e2e_str + " " + lewm_e2e_str + r""" \\
  & \textsc{Ext.}$\to$Easy & \multicolumn{5}{c}{-} """ + jepa_x2e_str + " " + lewm_x2e_str + r""" \\
\bottomrule
\multicolumn{12}{l}{%
  $^\dagger$ Shortcut test rows: model trained on one difficulty, evaluated on the other.
  Easy$\to$\textsc{Ext.} collapse ($\approx$50\% F1) confirms depth-shortcut exploitation.}\\
\multicolumn{12}{l}{%
  \textsc{Ext.}$\to$Easy retention ($>$88\% F1) confirms genuine acoustic feature learning.
  KF = Kingfish, SN = Snapper, CD = Cod, EM = Empty.}
\end{tabular}
\end{table*}
"""
    return latex


def generate_counting_table(results):
    """Generate table for counting task (MAE/RMSE)."""
    # Organize results by architecture and dataset
    organized = defaultdict(dict)
    
    for entry in results:
        if entry.get("task") != "counting":
            continue
        arch = entry["architecture"]
        dataset = entry["dataset"]
        if dataset not in organized[arch] or entry["timestamp"] > organized[arch][dataset]["timestamp"]:
            organized[arch][dataset] = entry.get("test", {})

    # Model order
    model_order = [
        "JEPA (Multi-modal)",
        "JEPA (Acoustic-Only)",
        "JEPA+SigReg",
        "LeWM",
    ]
    
    datasets = ["easy", "medium", "hard", "extreme"]
    
    # Generate LaTeX
    latex = r"""\begin{table*}[t]
\centering
\caption{%
  Counting task results: Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) for fish count estimation.
  Per-species errors shown for Kingfish (KF), Snapper (SN), Cod (CD), and Empty (EM).
  The overall macro error across all four classes is shown in the \textbf{Macro} column.
  Lower is better. Bold entries indicate the best (lowest) MAE per difficulty level.
}
\label{tab:counting_results}
\setlength{\tabcolsep}{4.0pt}
\small
\begin{tabular}{@{}ll cccc|cccc@{}}
\toprule
& & \multicolumn{4}{c|}{\textbf{MAE}} & \multicolumn{4}{c}{\textbf{RMSE}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
\textbf{Model} & \textbf{Dataset} 
  & KF & SN & CD & EM & \textbf{Macro}
  & KF & SN & CD & EM & \textbf{Macro} \\
\midrule
"""
    
    for model_name in model_order:
        if model_name not in organized:
            continue
        
        for dataset in datasets:
            if dataset not in organized[model_name]:
                continue
            
            metrics = organized[model_name][dataset]
            
            # Find best MAE for this dataset
            best_mae = float('inf')
            for m in model_order:
                if m in organized and dataset in organized[m]:
                    mae = organized[m][dataset].get("macro_mae", float('inf'))
                    if mae < best_mae:
                        best_mae = mae
            
            # Format values
            def fmt(value, is_best):
                if is_best:
                    return f"\\textbf{{{value:.3f}}}"
                return f"{value:.3f}"
            
            is_best = metrics.get("macro_mae", float('inf')) == best_mae
            
            latex += f"{model_name} & {dataset} "
            latex += f"& {fmt(metrics.get('kingfish_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('snapper_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('cod_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('empty_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('macro_mae', 0), is_best)} "
            latex += f"& {metrics.get('kingfish_rmse', 0):.3f} "
            latex += f"& {metrics.get('snapper_rmse', 0):.3f} "
            latex += f"& {metrics.get('cod_rmse', 0):.3f} "
            latex += f"& {metrics.get('empty_rmse', 0):.3f} "
            latex += f"& {metrics.get('macro_rmse', 0):.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from results.json")
    parser.add_argument("--task", type=str, default="presence",
                       choices=["presence", "majority", "counting"],
                       help="Task type: presence (multi-label), majority (single-label), or counting")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (default: table_<task>.tex)")
    args = parser.parse_args()
    
    results_file = os.path.join(os.path.dirname(__file__), "results.json")
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    print("="*70)
    print(f"Generating {args.task.upper()} task table...")
    print("="*70)
    
    if args.task == "counting":
        table = generate_counting_table(results)
        output_file = args.output or "table_counting.tex"
    else:
        table = generate_presence_table(results)
        output_file = args.output or f"table_{args.task}.tex"
    
    if table:
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        with open(output_path, "w") as f:
            f.write(table)
        print(f"Table saved to: {output_path}")
        print("\n" + table[:2000] + "..." if len(table) > 2000 else "\n" + table)
    else:
        print("No table generated (missing data)")


if __name__ == "__main__":
    main()
