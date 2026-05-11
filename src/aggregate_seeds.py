"""
Aggregate headline numbers across seeds for the reproducibility table.

Produces a small LaTeX table written to runs/repro_table.tex.

Headline metrics:
  * Power experiment: rejection rate at n=1000, K_factor in {0, 0.25, 0.5, 0.75, 0.95}
  * Power experiment: rejection rate at n=5000 for K_factor=0.95 (the hardest cell)
  * Power experiment: max Type-I FPR across all (n, K_factor>=1.05) cells
  * Anchored decoding: total #violations across (K x prompts), separately for the
    exact KL and the audit lower bound
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


POWER_DIRS = [
    ("seed 42", "runs/power_full"),
    ("seed 43", "runs/power_full_s43"),
    ("seed 44", "runs/power_full_s44"),
]

ANCHORED_DIRS = [
    ("seed 42", "runs/anchored_tightness"),
    ("seed 43", "runs/anchored_tightness_s43"),
    ("seed 44", "runs/anchored_tightness_s44"),
]


def load_power(p: Path):
    return json.loads((p / "summary.json").read_text())


def load_anchored(p: Path):
    return json.loads((p / "summary.json").read_text())


def fmt_pct(x):
    return f"{100 * x:.2f}\\%"


def fmt_mean_std_pct(values):
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return f"{100 * m:.2f}\\% \\pm {100 * s:.2f}\\%"


def main():
    # ---- Power experiment ----
    power = {}  # power[seed_label] = summary["by_n"]
    for label, d in POWER_DIRS:
        power[label] = load_power(Path(d))["by_n"]

    rows_power = []
    # n=1000 across K_factors
    for kf in ["0.00", "0.25", "0.50", "0.75", "0.95"]:
        vals = [power[lbl]["1000"][kf]["reject_rate"] for lbl, _ in POWER_DIRS]
        rows_power.append(("n=1000", kf, vals))
    # n=5000 for hard cell
    for kf in ["0.75", "0.95"]:
        vals = [power[lbl]["5000"][kf]["reject_rate"] for lbl, _ in POWER_DIRS]
        rows_power.append(("n=5000", kf, vals))

    # Type-I aggregate: max FPR across all (n, K>=1.05) cells per seed
    typei_per_seed = []
    for lbl, _ in POWER_DIRS:
        s = power[lbl]
        max_fp = 0.0
        for n_str, by_kf in s.items():
            for kf, info in by_kf.items():
                if float(kf) >= 1.05:
                    max_fp = max(max_fp, info["reject_rate"])
        typei_per_seed.append(max_fp)

    # ---- Anchored decoding ----
    anch = {}
    for label, d in ANCHORED_DIRS:
        anch[label] = load_anchored(Path(d))["by_K"]

    rows_anch = []
    NPROMPT = 30
    for K_str in ["0.50", "1.00", "2.00", "5.00"]:
        v_exact = [anch[lbl][K_str]["kl_anchored_vs_safe"]["fraction_above_K"]
                   for lbl, _ in ANCHORED_DIRS]
        v_lb = [anch[lbl][K_str]["audit_lb"]["fraction_above_K"]
                for lbl, _ in ANCHORED_DIRS]
        util = [anch[lbl][K_str]["budget_utilization_median"]
                for lbl, _ in ANCHORED_DIRS]
        rows_anch.append((K_str, NPROMPT, v_exact, v_lb, util))

    # ---- Build LaTeX ----
    lines = []
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{seed 42} & \textbf{seed 43} & \textbf{seed 44} & \textbf{mean} & \textbf{std} \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Power experiment (rejection rate, vs.\ true KL)}} \\")
    for n_lbl, kf, vals in rows_power:
        cells = " & ".join(fmt_pct(v) for v in vals)
        m = mean(vals); s = pstdev(vals)
        lines.append(f"\\quad reject @ {n_lbl}, $K_{{\\mathrm{{factor}}}}={kf}$ & {cells} & {100*m:.2f}\\% & {100*s:.2f}\\% \\\\")
    cells = " & ".join(fmt_pct(v) for v in typei_per_seed)
    m = mean(typei_per_seed); s = pstdev(typei_per_seed)
    lines.append(f"\\quad Type-I FPR (max over $K_{{\\mathrm{{factor}}}}\\!\\ge\\!1.05$, all $n$) & {cells} & {100*m:.2f}\\% & {100*s:.2f}\\% \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Anchored decoding tightness ($n_{\mathrm{prompt}}=30$, $n_{\mathrm{audit}}=1000$, $\delta=0.05$)}} \\")
    for K_str, np_, v_exact, v_lb, util in rows_anch:
        cells_e = " & ".join(f"{int(round(v*np_))}/{np_}" for v in v_exact)
        cells_l = " & ".join(f"{int(round(v*np_))}/{np_}" for v in v_lb)
        cells_u = " & ".join(f"{100*u:.1f}\\%" for u in util)
        lines.append(f"\\quad exact violations,\\ $K={K_str}$ & {cells_e} & --- & --- \\\\")
        lines.append(f"\\quad audit-LB violations,\\ $K={K_str}$ & {cells_l} & --- & --- \\\\")
        lines.append(f"\\quad budget util.\\ (median),\\ $K={K_str}$ & {cells_u} & --- & --- \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = Path("runs/repro_table.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
