"""
Regime Support as Alternative Dependent Variable
=================================================

Re-estimates the KOFGIdf -> political-institution relationship using
the V-Dem "regime support groups size" measure (v2regsupgroupssize)
from the "Behind the Throne" project as the dependent variable,
replacing the original W4 measure.

Models:
  1. Baseline OLS + Country FE (HC1 robust SE)
  2. Baseline OLS + Country FE (country-clustered SE)
  3. Two-way FE (country + year) with clustered SE
  4. KOF sub-index decomposition (two-way FE, clustered SE)

Comparison table: W4 vs RegSup results side by side.
"""

import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from asean_globalization_analysis import DATA_DIR, build_panel
from asean_subindex_analysis import (
    _star, _build_twoway_X, load_kof_subindices, SUB_INDICES, SUB_LABELS,
)

# ── Constants ────────────────────────────────────────────────────────────────

DV = "v2regsupgroupssize"
DV_LABEL = "Regime Support"
OUT_DIR = DATA_DIR


# ── Panel Prep ───────────────────────────────────────────────────────────────

def build_regsup_panel() -> pd.DataFrame:
    """Build analysis panel with regime support as DV."""
    panel = build_panel()

    # Load sub-indices and merge (drop KOFGIdf to avoid duplicate)
    kof_sub = load_kof_subindices(os.path.join(DATA_DIR, "KOFGI_2019_index.xlsx"))
    kof_sub = kof_sub.drop(columns=["KOFGIdf"])
    panel = panel.merge(kof_sub, on=["country_label", "year"], how="left")

    panel = panel.dropna(subset=[DV, "KOFGIdf", "e_gdppc", "log_pop"])
    return panel


# ── Models ───────────────────────────────────────────────────────────────────

def run_baseline_hc1(panel: pd.DataFrame):
    """Model 1: RegSup ~ KOFGIdf + controls + Country FE (HC1 SE)."""
    country_dum = pd.get_dummies(panel["country_label"], drop_first=True,
                                 dtype=float)
    X = pd.concat([panel[["KOFGIdf", "e_gdppc", "log_pop"]], country_dum],
                  axis=1)
    X = sm.add_constant(X)
    y = panel[DV]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    print("\n" + "=" * 70)
    print(f"MODEL 1: Baseline OLS — {DV_LABEL} (HC1 robust SE)")
    print(f"{DV} ~ KOFGIdf + e_gdppc + log_pop + Country FE")
    print("=" * 70)
    print(model.summary())
    return model


def run_baseline_clustered(panel: pd.DataFrame):
    """Model 2: RegSup ~ KOFGIdf + controls + Country FE (clustered SE)."""
    country_dum = pd.get_dummies(panel["country_label"], drop_first=True,
                                 dtype=float)
    X = pd.concat([panel[["KOFGIdf", "e_gdppc", "log_pop"]], country_dum],
                  axis=1)
    X = sm.add_constant(X)
    y = panel[DV]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["country_label"]},
    )

    print("\n" + "=" * 70)
    print(f"MODEL 2: Baseline OLS — {DV_LABEL} (country-clustered SE)")
    print(f"{DV} ~ KOFGIdf + e_gdppc + log_pop + Country FE")
    print("=" * 70)
    print(model.summary())
    return model


def run_twoway_fe(panel: pd.DataFrame):
    """Model 3: RegSup ~ KOFGIdf + controls + Country FE + Year FE (clustered)."""
    X = _build_twoway_X(panel, ["KOFGIdf"])
    y = panel[DV]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["country_label"]},
    )

    print("\n" + "=" * 70)
    print(f"MODEL 3: Two-Way FE — {DV_LABEL} (country-clustered SE)")
    print(f"{DV} ~ KOFGIdf + e_gdppc + log_pop + Country FE + Year FE")
    print("=" * 70)
    print(model.summary())
    return model


def run_subindex_models(panel: pd.DataFrame) -> OrderedDict:
    """Model 4: Each KOF sub-index separately, two-way FE, clustered SE."""
    results = OrderedDict()

    for idx in SUB_INDICES:
        label = SUB_LABELS[idx]
        X = _build_twoway_X(panel, [idx])
        y = panel[DV]
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["country_label"]},
        )
        ci = model.conf_int().loc[idx]

        print(f"\n{'=' * 70}")
        print(f"MODEL 4 — {label} Globalization ({idx})")
        print(f"{DV} ~ {idx} + e_gdppc + log_pop + Country FE + Year FE")
        print(f"{'=' * 70}")
        print(model.summary())

        results[idx] = {
            "model": model,
            "label": label,
            "coef": model.params[idx],
            "se": model.bse[idx],
            "pval": model.pvalues[idx],
            "ci_lo": ci.iloc[0],
            "ci_hi": ci.iloc[1],
            "r2": model.rsquared,
            "nobs": int(model.nobs),
        }

    return results


# ── W4 Benchmark Models ─────────────────────────────────────────────────────

def run_w4_benchmarks(panel: pd.DataFrame) -> list:
    """Re-run the 3 core models with W4 for the comparison table."""
    models = []
    country_dum = pd.get_dummies(panel["country_label"], drop_first=True,
                                 dtype=float)

    # HC1
    X = pd.concat([panel[["KOFGIdf", "e_gdppc", "log_pop"]], country_dum],
                  axis=1)
    X = sm.add_constant(X)
    models.append(sm.OLS(panel["W4"], X).fit(cov_type="HC1"))

    # Clustered
    models.append(sm.OLS(panel["W4"], X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["country_label"]},
    ))

    # Two-way FE clustered
    X2 = _build_twoway_X(panel, ["KOFGIdf"])
    models.append(sm.OLS(panel["W4"], X2).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["country_label"]},
    ))

    return models


# ── Tables ───────────────────────────────────────────────────────────────────

def print_comparison_table(w4_models: list, rs_models: list) -> None:
    """Side-by-side comparison: W4 vs Regime Support across 3 specifications."""
    print("\n" + "=" * 95)
    print("COMPARISON TABLE — KOFGIdf Effect: W4 vs Regime Support")
    print("=" * 95)

    col_w = 15
    spec_labels = ["(1) HC1", "(2) Clustered", "(3) Two-Way FE"]
    header = (f"{'':>{col_w}}"
              + "".join(f"{l:>{col_w}}" for l in spec_labels)
              + " | "
              + "".join(f"{l:>{col_w}}" for l in spec_labels))
    dv_header = (f"{'':>{col_w}}"
                 + f"{'--- W4 ---':^{col_w * 3}}"
                 + " | "
                 + f"{'--- Regime Support ---':^{col_w * 3}}")
    print(dv_header)
    print(header)
    print("-" * len(header))

    # KOFGIdf row
    coef_row = f"{'KOFGIdf':>{col_w}}"
    se_row = f"{'':>{col_w}}"
    for m in w4_models:
        stars = _star(m.pvalues["KOFGIdf"])
        coef_row += f"{m.params['KOFGIdf']:>{col_w - len(stars)}.5f}{stars}"
        se_str = f"({m.bse['KOFGIdf']:.5f})"
        se_row += f"{se_str:>{col_w}}"
    coef_row += " | "
    se_row += " | "
    for m in rs_models:
        stars = _star(m.pvalues["KOFGIdf"])
        coef_row += f"{m.params['KOFGIdf']:>{col_w - len(stars)}.5f}{stars}"
        se_str = f"({m.bse['KOFGIdf']:.5f})"
        se_row += f"{se_str:>{col_w}}"
    print(coef_row)
    print(se_row)

    # Controls
    for var in ["e_gdppc", "log_pop"]:
        coef_row = f"{var:>{col_w}}"
        se_row = f"{'':>{col_w}}"
        for m in w4_models:
            stars = _star(m.pvalues[var])
            coef_row += f"{m.params[var]:>{col_w - len(stars)}.5f}{stars}"
            se_str = f"({m.bse[var]:.5f})"
            se_row += f"{se_str:>{col_w}}"
        coef_row += " | "
        se_row += " | "
        for m in rs_models:
            stars = _star(m.pvalues[var])
            coef_row += f"{m.params[var]:>{col_w - len(stars)}.5f}{stars}"
            se_str = f"({m.bse[var]:.5f})"
            se_row += f"{se_str:>{col_w}}"
        print(coef_row)
        print(se_row)

    print("-" * len(header))

    # Footer rows
    def ft(lbl, w4_vals, rs_vals):
        return (f"{lbl:>{col_w}}"
                + "".join(f"{v:>{col_w}}" for v in w4_vals)
                + " | "
                + "".join(f"{v:>{col_w}}" for v in rs_vals))

    print(ft("Country FE", ["Yes"] * 3, ["Yes"] * 3))
    print(ft("Year FE", ["No", "No", "Yes"], ["No", "No", "Yes"]))
    print(ft("SE type", ["HC1", "Cluster", "Cluster"],
             ["HC1", "Cluster", "Cluster"]))

    n_row = f"{'N':>{col_w}}"
    r2_row = f"{'R^2':>{col_w}}"
    for m in w4_models:
        n_row += f"{int(m.nobs):>{col_w}}"
        r2_row += f"{m.rsquared:>{col_w}.3f}"
    n_row += " | "
    r2_row += " | "
    for m in rs_models:
        n_row += f"{int(m.nobs):>{col_w}}"
        r2_row += f"{m.rsquared:>{col_w}.3f}"
    print(n_row)
    print(r2_row)
    print("=" * len(header))
    print("* p<0.10, ** p<0.05, *** p<0.01\n")


def print_subindex_table(results: OrderedDict) -> None:
    """Regression table for sub-index models."""
    print("\n" + "=" * 80)
    print(f"SUB-INDEX DECOMPOSITION — DV: {DV_LABEL} (Two-Way FE, Clustered SE)")
    print("=" * 80)

    col_w = 18
    header = (f"{'':>{col_w}}"
              + "".join(f"{r['label']:>{col_w}}" for r in results.values()))
    print(header)
    print("-" * len(header))

    # Sub-index coefficient row
    for idx, r in results.items():
        coef_row = f"{idx:>{col_w}}"
        se_row = f"{'':>{col_w}}"
        stars = _star(r["pval"])
        coef_row += f"{r['coef']:>{col_w - len(stars)}.5f}{stars}"
        se_str = f"({r['se']:.5f})"
        se_row += f"{se_str:>{col_w}}"
        # Pad remaining columns
        coef_row += f"{'':>{col_w}}" * (len(results) - 1)
        se_row += f"{'':>{col_w}}" * (len(results) - 1)
        print(coef_row)
        print(se_row)

    print("-" * len(header))

    n_row = f"{'N':>{col_w}}"
    r2_row = f"{'R^2':>{col_w}}"
    for r in results.values():
        n_row += f"{r['nobs']:>{col_w}}"
        r2_row += f"{r['r2']:>{col_w}.3f}"
    print(n_row)
    print(r2_row)
    print("=" * len(header))
    print("All models: Country FE + Year FE, country-clustered SE")
    print("* p<0.10, ** p<0.05, *** p<0.01\n")


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_coefficient_comparison(w4_models: list, rs_models: list) -> str:
    """Side-by-side coefficient comparison: W4 vs Regime Support."""
    spec_labels = ["Country FE\n(HC1)", "Country FE\n(Clustered)",
                   "Two-Way FE\n(Clustered)"]

    w4_coefs = [m.params["KOFGIdf"] for m in w4_models]
    w4_ci = [m.conf_int().loc["KOFGIdf"] for m in w4_models]
    w4_lo = [ci.iloc[0] for ci in w4_ci]
    w4_hi = [ci.iloc[1] for ci in w4_ci]

    rs_coefs = [m.params["KOFGIdf"] for m in rs_models]
    rs_ci = [m.conf_int().loc["KOFGIdf"] for m in rs_models]
    rs_lo = [ci.iloc[0] for ci in rs_ci]
    rs_hi = [ci.iloc[1] for ci in rs_ci]

    x = np.arange(len(spec_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    # W4
    err_w4_lo = [c - lo for c, lo in zip(w4_coefs, w4_lo)]
    err_w4_hi = [hi - c for c, hi in zip(w4_coefs, w4_hi)]
    ax.errorbar(x - width / 2, w4_coefs, yerr=[err_w4_lo, err_w4_hi],
                fmt="s", color="#2c3e50", capsize=6, markersize=9,
                linewidth=1.8, capthick=1.3, label="DV: W4")

    # Regime Support
    err_rs_lo = [c - lo for c, lo in zip(rs_coefs, rs_lo)]
    err_rs_hi = [hi - c for c, hi in zip(rs_coefs, rs_hi)]
    ax.errorbar(x + width / 2, rs_coefs, yerr=[err_rs_lo, err_rs_hi],
                fmt="o", color="#e74c3c", capsize=6, markersize=9,
                linewidth=1.8, capthick=1.3, label="DV: Regime Support")

    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(spec_labels, fontsize=10)
    ax.set_ylabel("KOFGIdf coefficient (95% CI)", fontsize=11)
    ax.set_title("KOFGIdf Effect: W4 vs Regime Support\nAcross Specifications",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig_regsup_coefficient_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_subindex_coefficients(results: OrderedDict) -> str:
    """Coefficient plot for sub-index models with regime support DV."""
    labels = [r["label"] for r in results.values()]
    coefs = [r["coef"] for r in results.values()]
    ci_lo = [r["ci_lo"] for r in results.values()]
    ci_hi = [r["ci_hi"] for r in results.values()]

    y_pos = np.arange(len(labels))
    err_lo = [c - lo for c, lo in zip(coefs, ci_lo)]
    err_hi = [hi - c for c, hi in zip(coefs, ci_hi)]

    colors = ["#2c3e50", "#e74c3c", "#3498db", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(len(labels)):
        ax.errorbar(coefs[i], y_pos[i],
                    xerr=[[err_lo[i]], [err_hi[i]]],
                    fmt="o", color=colors[i], capsize=5, markersize=8,
                    linewidth=1.5, capthick=1.2)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Coefficient on Regime Support (95% CI)", fontsize=11)
    ax.set_title("KOF Sub-Index Effects on Regime Support\n(Two-Way FE, Clustered SE)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig_regsup_subindex_coefficients.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Findings Summary ─────────────────────────────────────────────────────────

def print_findings_summary(w4_models: list, rs_models: list,
                           sub_results: OrderedDict) -> None:
    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY")
    print("=" * 70)

    spec_labels = ["HC1", "Clustered", "Two-Way FE"]

    print("\nKOFGIdf coefficient on W4 vs Regime Support:")
    for i, spec in enumerate(spec_labels):
        w_c = w4_models[i].params["KOFGIdf"]
        w_p = w4_models[i].pvalues["KOFGIdf"]
        r_c = rs_models[i].params["KOFGIdf"]
        r_p = rs_models[i].pvalues["KOFGIdf"]
        w_sig = "sig" + _star(w_p) if w_p < 0.10 else "n.s."
        r_sig = "sig" + _star(r_p) if r_p < 0.10 else "n.s."
        print(f"  {spec:<14s}  W4: {w_c:>+.5f} (p={w_p:.3f}, {w_sig})"
              f"   RegSup: {r_c:>+.5f} (p={r_p:.3f}, {r_sig})")

    print("\nSub-index decomposition (Two-Way FE, Clustered SE):")
    any_sig = False
    for idx, r in sub_results.items():
        status = "significant" + _star(r["pval"]) if r["pval"] < 0.10 \
            else "NOT significant"
        if r["pval"] < 0.10:
            any_sig = True
        print(f"  {r['label']:<16s} ({idx}): "
              f"beta = {r['coef']:>+.5f}, p = {r['pval']:.3f} -- {status}")

    if any_sig:
        print("\n  --> At least one sub-index is significant with the regime "
              "support DV!")
    else:
        print("\n  --> No sub-index reaches significance, same pattern as W4.")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("REGIME SUPPORT AS ALTERNATIVE DEPENDENT VARIABLE")
    print("=" * 70)
    print("\nLoading panel data...")
    panel = build_regsup_panel()
    print(f"Panel: {len(panel)} obs, "
          f"{panel['country_label'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")
    print(f"\n{DV} descriptive statistics:")
    print(panel[DV].describe().to_string())

    # Core models with regime support DV
    print("\n" + "=" * 70)
    print("REGIME SUPPORT MODELS")
    print("=" * 70)
    m1 = run_baseline_hc1(panel)
    m2 = run_baseline_clustered(panel)
    m3 = run_twoway_fe(panel)
    rs_models = [m1, m2, m3]

    # Sub-index decomposition
    print("\n" + "=" * 70)
    print("SUB-INDEX DECOMPOSITION")
    print("=" * 70)
    sub_results = run_subindex_models(panel)

    # W4 benchmarks for comparison
    print("\n" + "=" * 70)
    print("W4 BENCHMARK MODELS (for comparison)")
    print("=" * 70)
    w4_models = run_w4_benchmarks(panel)

    # Comparison table
    print_comparison_table(w4_models, rs_models)

    # Sub-index table
    print_subindex_table(sub_results)

    # Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_coefficient_comparison(w4_models, rs_models)
    plot_subindex_coefficients(sub_results)

    # Summary
    print_findings_summary(w4_models, rs_models, sub_results)
    print("Done.")


if __name__ == "__main__":
    main()
