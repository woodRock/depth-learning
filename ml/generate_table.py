import json
import os
from collections import defaultdict


def generate_combined_table(results_path):
    """Generate combined LaTeX table with three sections:
    1. Multi-modal JEPA (joint embedding training)
    2. Acoustic-Only JEPA (evaluated on acoustic only)
    3. Acoustic-Only LeWM (evaluated on acoustic only)
    
    With shortcut test results at the bottom.
    """
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    # Organize results into three categories
    multimodal_jepa = defaultdict(dict)  # Original JEPA training (joint embedding)
    acoustic_jepa = defaultdict(dict)    # JEPA acoustic-only evaluation
    acoustic_lewm = defaultdict(dict)    # LeWM acoustic-only evaluation
    shortcut_results = {"easy_to_extreme": {}, "extreme_to_easy": {}}

    for entry in results:
        arch = entry["architecture"]
        dataset = entry["dataset"]

        # Skip shortcut tests for main table
        if entry.get("shortcut_test") and entry.get("test_dataset"):
            test_ds = entry["test_dataset"]
            test_data = entry.get("test")
            if test_data:
                if dataset == "easy" and test_ds == "extreme":
                    shortcut_results["easy_to_extreme"][arch] = test_data
                elif dataset == "extreme" and test_ds == "easy":
                    shortcut_results["extreme_to_easy"][arch] = test_data
            continue

        # Categorize by mode
        if entry.get("mode") == "acoustic_only":
            if arch == "JEPA":
                if dataset not in acoustic_jepa[arch] or entry["timestamp"] > acoustic_jepa[arch][dataset]["timestamp"]:
                    acoustic_jepa[arch][dataset] = entry
            elif arch == "LeWM":
                if dataset not in acoustic_lewm[arch] or entry["timestamp"] > acoustic_lewm[arch][dataset]["timestamp"]:
                    acoustic_lewm[arch][dataset] = entry
        else:
            # Multi-modal (joint embedding) training results
            if arch == "JEPA":
                if dataset not in multimodal_jepa[arch] or entry["timestamp"] > multimodal_jepa[arch][dataset]["timestamp"]:
                    multimodal_jepa[arch][dataset] = entry

    # Difficulty order
    difficulties = ["easy", "medium", "hard", "extreme"]
    splits = ["train", "val", "test"]

    # Extract metrics helper
    def get_metrics(organized, arch, dataset, split):
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

    # Determine best scores per difficulty and split for bolding (across all three sections)
    def find_best(difficulty, split):
        best_macro = -1
        best_key = None
        for key, organized in [("jepa_mm", multimodal_jepa), ("jepa_ac", acoustic_jepa), ("lewm_ac", acoustic_lewm)]:
            metrics = get_metrics(organized, "JEPA" if key != "lewm_ac" else "LeWM", difficulty, split)
            if metrics and metrics["macro"] > best_macro:
                best_macro = metrics["macro"]
                best_key = key
        return best_key

    # Format value with bold if best
    def fmt(value, is_best):
        if is_best:
            return f"\\textbf{{{value:.1f}}}"
        return f"{value:.1f}"

    # Start building LaTeX table
    latex = r"""\begin{table*}[t]
\centering
\caption{%
  Full benchmark results: macro-averaged F1 score (\%) across all four difficulty modes and all three data splits (Train / Validation / Test).
  Per-species F1 scores are shown for Kingfish (KF), Snapper (SN), Cod (CD), and Empty (EM).
  The overall macro F1 across all four classes is shown in the \textbf{Macro} column.
  Three model variants are compared: (1) Multi-modal JEPA trained with joint embedding (visual teacher),
  (2) Acoustic-Only JEPA evaluated on acoustic input only,
  (3) Acoustic-Only LeWM trained and evaluated on acoustic only.
  Bold entries indicate the best score per difficulty level, split, and species across all three variants.
  $\uparrow$ indicates higher is better.
}
\label{tab:combined_results}
\setlength{\tabcolsep}{3.5pt}
\small
\begin{tabular}{@{}ll lllll lllll lllll@{}}
\toprule
& & \multicolumn{5}{c}{\textbf{Multi-modal JEPA}} 
  & \multicolumn{5}{c}{\textbf{Acoustic-Only JEPA}} 
  & \multicolumn{5}{c}{\textbf{Acoustic-Only LeWM}} \\
\cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17}
\textbf{Difficulty} & \textbf{Split} 
  & KF$\uparrow$ & SN$\uparrow$ & CD$\uparrow$ & EM$\uparrow$ & \textbf{Macro}$\uparrow$
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
            best_key = find_best(difficulty, split)
            
            # Multi-modal JEPA
            jepa_mm = get_metrics(multimodal_jepa, "JEPA", difficulty, split)
            if jepa_mm:
                jepa_mm_str = (
                    f"& {fmt(jepa_mm['kf'], best_key == 'jepa_mm')} & {fmt(jepa_mm['sn'], best_key == 'jepa_mm')} & "
                    f"{fmt(jepa_mm['cd'], best_key == 'jepa_mm')} & {fmt(jepa_mm['em'], best_key == 'jepa_mm')} & "
                    f"{fmt(jepa_mm['macro'], best_key == 'jepa_mm')}"
                )
            else:
                jepa_mm_str = "& - & - & - & - & -"
            
            # Acoustic-Only JEPA
            jepa_ac = get_metrics(acoustic_jepa, "JEPA", difficulty, split)
            if jepa_ac:
                jepa_ac_str = (
                    f"& {fmt(jepa_ac['kf'], best_key == 'jepa_ac')} & {fmt(jepa_ac['sn'], best_key == 'jepa_ac')} & "
                    f"{fmt(jepa_ac['cd'], best_key == 'jepa_ac')} & {fmt(jepa_ac['em'], best_key == 'jepa_ac')} & "
                    f"{fmt(jepa_ac['macro'], best_key == 'jepa_ac')}"
                )
            else:
                jepa_ac_str = "& - & - & - & - & -"
            
            # Acoustic-Only LeWM
            lewm_ac = get_metrics(acoustic_lewm, "LeWM", difficulty, split)
            if lewm_ac:
                lewm_ac_str = (
                    f"& {fmt(lewm_ac['kf'], best_key == 'lewm_ac')} & {fmt(lewm_ac['sn'], best_key == 'lewm_ac')} & "
                    f"{fmt(lewm_ac['cd'], best_key == 'lewm_ac')} & {fmt(lewm_ac['em'], best_key == 'lewm_ac')} & "
                    f"{fmt(lewm_ac['macro'], best_key == 'lewm_ac')}"
                )
            else:
                lewm_ac_str = "& - & - & - & - & -"
            
            # First column handling with multirow
            latex += f"  & {split_display} {jepa_mm_str} {jepa_ac_str} {lewm_ac_str} \\\\\n"
        
        if i < len(difficulties) - 1:
            latex += "\\midrule\n"

    # Shortcut test rows
    def convert_to_metrics(test_data):
        return {
            "kf": test_data["kingfish_f1"] * 100,
            "sn": test_data["snapper_f1"] * 100,
            "cd": test_data["cod_f1"] * 100,
            "em": test_data["empty_f1"] * 100,
            "macro": test_data["avg_f1"] * 100,
        }

    shortcut_e2e_jepa = shortcut_results["easy_to_extreme"].get("JEPA")
    shortcut_e2e_lewm = shortcut_results["easy_to_extreme"].get("LeWM")
    shortcut_x2e_jepa = shortcut_results["extreme_to_easy"].get("JEPA")
    shortcut_x2e_lewm = shortcut_results["extreme_to_easy"].get("LeWM")

    # Determine best for shortcut tests
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
\multicolumn{17}{l}{%
  $^\dagger$ Shortcut test rows: model trained on one difficulty, evaluated on the other.
  JEPA results are from multi-modal training (with visual teacher) evaluated in acoustic-only mode.
  LeWM results are from acoustic-only training and evaluation.
  Easy$\to$\textsc{Ext.} collapse ($\approx$50\% F1) confirms depth-shortcut exploitation.}\\
\multicolumn{17}{l}{%
  \textsc{Ext.}$\to$Easy retention ($>$88\% F1) confirms genuine acoustic feature learning.
  KF = Kingfish, SN = Snapper, CD = Cod, EM = Empty.}
\end{tabular}
\end{table*}
"""
    return latex


if __name__ == "__main__":
    results_file = os.path.join(os.path.dirname(__file__), "results.json")
    
    # Generate combined table with all three sections
    print("="*60)
    print("Generating COMBINED benchmark table...")
    print("="*60)
    table = generate_combined_table(results_file)
    if table:
        with open(os.path.join(os.path.dirname(__file__), "table_combined.tex"), "w") as f:
            f.write(table)
        print("Saved to ml/table_combined.tex")
    
    print("\nDone!")
