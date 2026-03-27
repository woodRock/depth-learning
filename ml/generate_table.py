import json
import os
from collections import defaultdict


def generate_latex_table(results_path):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    # Organize results by architecture and dataset
    organized = defaultdict(dict)
    shortcut_results = {"easy_to_extreme": {}, "extreme_to_easy": {}}
    
    for entry in results:
        arch = entry["architecture"]
        dataset = entry["dataset"]
        
        # Check if this is a shortcut test entry
        if entry.get("shortcut_test") and entry.get("test_dataset"):
            test_ds = entry["test_dataset"]
            test_data = entry.get("test")
            if test_data:
                if dataset == "easy" and test_ds == "extreme":
                    shortcut_results["easy_to_extreme"][arch] = test_data
                elif dataset == "extreme" and test_ds == "easy":
                    shortcut_results["extreme_to_easy"][arch] = test_data
            continue  # Skip shortcut entries for regular table
        
        # Keep the latest entry per (architecture, dataset) pair for regular results
        if dataset not in organized[arch] or entry["timestamp"] > organized[arch][dataset]["timestamp"]:
            organized[arch][dataset] = entry

    # Difficulty order
    difficulties = ["easy", "medium", "hard", "extreme"]
    splits = ["train", "val", "test"]

    # Extract metrics for each architecture/dataset/split
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
    def fmt(arch, value, best_arch):
        if best_arch is None:
            return f"{value:.1f}"
        if arch == best_arch:
            return f"\\textbf{{{value:.1f}}}"
        return f"{value:.1f}"

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
\label{tab:full_results}
\setlength{\tabcolsep}{4.5pt}
\small
\begin{tabular}{@{}ll lllll lllll@{}}
\toprule
& & \multicolumn{5}{c}{\textbf{Multimodal JEPA} \cite{assran2023self}} 
  & \multicolumn{5}{c}{\textbf{LeWM (Acoustic Only)} \cite{lecun2026le}} \\
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
                    f"& {fmt('JEPA', jepa['kf'], best_arch)} & {fmt('JEPA', jepa['sn'], best_arch)} & "
                    f"{fmt('JEPA', jepa['cd'], best_arch)} & {fmt('JEPA', jepa['em'], best_arch)} & "
                    f"{fmt('JEPA', jepa['macro'], best_arch)}"
                )
            else:
                jepa_str = "& - & - & - & - & -"
            
            if lewm:
                lewm_str = (
                    f"& {fmt('LeWM', lewm['kf'], best_arch)} & {fmt('LeWM', lewm['sn'], best_arch)} & "
                    f"{fmt('LeWM', lewm['cd'], best_arch)} & {fmt('LeWM', lewm['em'], best_arch)} & "
                    f"{fmt('LeWM', lewm['macro'], best_arch)}"
                )
            else:
                lewm_str = "& - & - & - & - & -"
            
            # First column handling with multirow
            if j == 0:
                latex += f"  & {split_display} {jepa_str} {lewm_str} \\\\\n"
            else:
                latex += f"  & {split_display} {jepa_str} {lewm_str} \\\\\n"
        
        if i < len(difficulties) - 1:
            latex += "\\midrule\n"

    # Shortcut test rows using pre-extracted shortcut_results
    def convert_to_metrics(test_data):
        return {
            "kf": test_data["kingfish_f1"] * 100,
            "sn": test_data["snapper_f1"] * 100,
            "cd": test_data["cod_f1"] * 100,
            "em": test_data["empty_f1"] * 100,
            "macro": test_data["avg_f1"] * 100,
        }

    shortcut_easy_to_ext = {arch: convert_to_metrics(data) 
                           for arch, data in shortcut_results["easy_to_extreme"].items()}
    shortcut_ext_to_easy = {arch: convert_to_metrics(data) 
                           for arch, data in shortcut_results["extreme_to_easy"].items()}

    # Determine best for shortcut tests
    def get_best_shortcut(shortcut_dict):
        best_macro = -1
        best_arch = None
        for arch in ["JEPA", "LeWM"]:
            if arch in shortcut_dict:
                if shortcut_dict[arch]["macro"] > best_macro:
                    best_macro = shortcut_dict[arch]["macro"]
                    best_arch = arch
        return best_arch

    best_easy_to_ext = get_best_shortcut(shortcut_easy_to_ext)
    best_ext_to_easy = get_best_shortcut(shortcut_ext_to_easy)

    def fmt_shortcut(arch, value, best_arch):
        if best_arch is None or arch not in shortcut_easy_to_ext:
            return f"{value:.1f}"
        if arch == best_arch:
            return f"\\textbf{{{value:.1f}}}"
        return f"{value:.1f}"

    # Easy→Ext row
    jepa_e2e = shortcut_easy_to_ext.get("JEPA")
    lewm_e2e = shortcut_easy_to_ext.get("LeWM")
    
    if jepa_e2e:
        jepa_e2e_str = (
            f"& {fmt_shortcut('JEPA', jepa_e2e['kf'], best_easy_to_ext)} & "
            f"{fmt_shortcut('JEPA', jepa_e2e['sn'], best_easy_to_ext)} & "
            f"{fmt_shortcut('JEPA', jepa_e2e['cd'], best_easy_to_ext)} & "
            f"{fmt_shortcut('JEPA', jepa_e2e['em'], best_easy_to_ext)} & "
            f"{fmt_shortcut('JEPA', jepa_e2e['macro'], best_easy_to_ext)}"
        )
    else:
        jepa_e2e_str = "& - & - & - & - & -"
    
    if lewm_e2e:
        lewm_e2e_str = (
            f"& {fmt_shortcut('LeWM', lewm_e2e['kf'], best_easy_to_ext)} & "
            f"{fmt_shortcut('LeWM', lewm_e2e['sn'], best_easy_to_ext)} & "
            f"{fmt_shortcut('LeWM', lewm_e2e['cd'], best_easy_to_ext)} & "
            f"{fmt_shortcut('LeWM', lewm_e2e['em'], best_easy_to_ext)} & "
            f"{fmt_shortcut('LeWM', lewm_e2e['macro'], best_easy_to_ext)}"
        )
    else:
        lewm_e2e_str = "& - & - & - & - & -"

    # Ext→Easy row
    jepa_x2e = shortcut_ext_to_easy.get("JEPA")
    lewm_x2e = shortcut_ext_to_easy.get("LeWM")
    
    if jepa_x2e:
        jepa_x2e_str = (
            f"& {fmt_shortcut('JEPA', jepa_x2e['kf'], best_ext_to_easy)} & "
            f"{fmt_shortcut('JEPA', jepa_x2e['sn'], best_ext_to_easy)} & "
            f"{fmt_shortcut('JEPA', jepa_x2e['cd'], best_ext_to_easy)} & "
            f"{fmt_shortcut('JEPA', jepa_x2e['em'], best_ext_to_easy)} & "
            f"{fmt_shortcut('JEPA', jepa_x2e['macro'], best_ext_to_easy)}"
        )
    else:
        jepa_x2e_str = "& - & - & - & - & -"
    
    if lewm_x2e:
        lewm_x2e_str = (
            f"& {fmt_shortcut('LeWM', lewm_x2e['kf'], best_ext_to_easy)} & "
            f"{fmt_shortcut('LeWM', lewm_x2e['sn'], best_ext_to_easy)} & "
            f"{fmt_shortcut('LeWM', lewm_x2e['cd'], best_ext_to_easy)} & "
            f"{fmt_shortcut('LeWM', lewm_x2e['em'], best_ext_to_easy)} & "
            f"{fmt_shortcut('LeWM', lewm_x2e['macro'], best_ext_to_easy)}"
        )
    else:
        lewm_x2e_str = "& - & - & - & - & -"

    latex += r"""
\midrule
%--- SHORTCUT TEST ---
\multirow{2}{*}{\shortstack[l]{Shortcut\\Test$^\dagger$}}
  & Easy$\to$\textsc{Ext.} """ + jepa_e2e_str + " " + lewm_e2e_str + r""" \\
  & \textsc{Ext.}$\to$Easy """ + jepa_x2e_str + " " + lewm_x2e_str + r""" \\
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


if __name__ == "__main__":
    results_file = os.path.join(os.path.dirname(__file__), "results.json")
    table = generate_latex_table(results_file)
    if table:
        print(table)
        # Also save to file
        with open(os.path.join(os.path.dirname(__file__), "table.tex"), "w") as f:
            f.write(table)
        print(f"\nLaTeX table saved to ml/table.tex")
