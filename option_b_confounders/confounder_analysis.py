"""
Option B: Targeted Confounders Replacing Year FE
=================================================

Re-estimates the KOFGIdf → W relationship by progressively adding
temporal controls — from no time controls (baseline) through period
dummies, linear/quadratic trends, and full year FE — to pinpoint
exactly which temporal specification absorbs the globalization effect.

Seven models, all with country FE and country-clustered SE:
  1. Baseline (no temporal controls)
  2. Cold War dummy (post-1991)
  3. Post-AFC dummy (post-1997)
  4. Both Cold War + post-AFC
  5. Linear time trend
  6. Quadratic time trend
  7. Full year FE (upper-bound benchmark)
"""

import os
import sys
from collections import OrderedDict

# Allow imports from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from asean_globalization_analysis import build_panel
from asean_subindex_analysis import _star

# ── Constants ────────────────────────────────────────────────────────────────

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SPECS = [
    (1, "Baseline",     []),
    (2, "Cold War",     ["post_cw"]),
    (3, "Post-AFC",     ["post_afc"]),
    (4, "CW + AFC",     ["post_cw", "post_afc"]),
    (5, "Linear",       ["year_c"]),
    (6, "Quadratic",    ["year_c", "year_c2"]),
    (7, "Year FE",      "year_fe"),
]


# ── Panel Prep ───────────────────────────────────────────────────────────────

def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Add centered time trend variables."""
    panel = panel.copy()
    panel["year_c"] = (panel["year"] - 1993).astype(float)
    panel["year_c2"] = panel["year_c"] ** 2
    return panel


# ── Models ───────────────────────────────────────────────────────────────────

def run_all_models(panel: pd.DataFrame) -> OrderedDict:
    """Run all 7 model specifications and collect results."""
    results = OrderedDict()

    for num, label, temporal in MODEL_SPECS:
        country_dum = pd.get_dummies(panel["country_label"], drop_first=True,
                                     dtype=float)

        if temporal == "year_fe":
            year_dum = pd.get_dummies(panel["year"], prefix="yr",
                                      drop_first=True, dtype=float)
            X = pd.concat(
                [panel[["KOFGIdf", "e_gdppc", "log_pop"]],
                 country_dum, year_dum], axis=1,
            )
        else:
            iv_cols = ["KOFGIdf"] + temporal + ["e_gdppc", "log_pop"]
            X = pd.concat([panel[iv_cols], country_dum], axis=1)

        X = sm.add_constant(X)
        y = panel["W4"]
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["country_label"]},
        )

        # VIF for KOFGIdf
        kof_idx = list(X.columns).index("KOFGIdf")
        vif = variance_inflation_factor(X.values, kof_idx)

        ci = model.conf_int().loc["KOFGIdf"]
        col_label = f"({num}) {label}"

        print(f"\n{'=' * 70}")
        temporal_desc = (
            "full year dummies" if temporal == "year_fe"
            else ", ".join(temporal) if temporal
            else "none"
        )
        print(f"MODEL {num}: {label} (temporal: {temporal_desc})")
        print(f"W4 ~ KOFGIdf + [{temporal_desc}] + e_gdppc + log_pop"
              " + Country FE")
        print(f"{'=' * 70}")
        print(model.summary())

        results[col_label] = {
            "model": model,
            "num": num,
            "label": label,
            "temporal": temporal,
            "coef": model.params["KOFGIdf"],
            "se": model.bse["KOFGIdf"],
            "pval": model.pvalues["KOFGIdf"],
            "ci_lo": ci.iloc[0],
            "ci_hi": ci.iloc[1],
            "r2": model.rsquared,
            "nobs": int(model.nobs),
            "vif": vif,
        }

    return results


# ── Regression Table ─────────────────────────────────────────────────────────

def print_regression_table(results: OrderedDict) -> None:
    """Stargazer-style side-by-side table for all 7 models."""
    print("\n" + "=" * 105)
    print("REGRESSION TABLE — KOFGIdf Coefficient Progression")
    print("=" * 105)

    col_w = 14
    labels = list(results.keys())
    n_cols = len(labels)

    header = f"{'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    # KOFGIdf row
    coef_row = f"{'KOFGIdf':>{col_w}}"
    se_row = f"{'':>{col_w}}"
    for r in results.values():
        stars = _star(r["pval"])
        coef_row += f"{r['coef']:>{col_w - len(stars)}.5f}{stars}"
        se_str = f"({r['se']:.5f})"
        se_row += f"{se_str:>{col_w}}"
    print(coef_row)
    print(se_row)

    # Controls: e_gdppc, log_pop
    for var in ["e_gdppc", "log_pop"]:
        coef_row = f"{var:>{col_w}}"
        se_row = f"{'':>{col_w}}"
        for r in results.values():
            m = r["model"]
            c = m.params[var]
            s = m.bse[var]
            p = m.pvalues[var]
            stars = _star(p)
            coef_row += f"{c:>{col_w - len(stars)}.5f}{stars}"
            se_row += f"{'(' + f'{s:.5f}' + ')':>{col_w}}"
        print(coef_row)
        print(se_row)

    # Temporal controls
    temp_row = f"{'Temporal':>{col_w}}"
    for r in results.values():
        t = r["temporal"]
        if t == "year_fe":
            desc = "Year FE"
        elif t:
            desc = "+".join(t)
        else:
            desc = "None"
        temp_row += f"{desc:>{col_w}}"
    print("-" * len(header))
    print(temp_row)

    # Footer
    def ft(lbl, vals):
        return f"{lbl:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in vals)

    print(ft("Country FE", ["Yes"] * n_cols))
    print(ft("SE cluster", ["Country"] * n_cols))

    n_row = f"{'N':>{col_w}}"
    r2_row = f"{'R²':>{col_w}}"
    vif_row = f"{'VIF(KOFGIdf)':>{col_w}}"
    for r in results.values():
        n_row += f"{r['nobs']:>{col_w}}"
        r2_row += f"{r['r2']:>{col_w}.3f}"
        vif_row += f"{r['vif']:>{col_w}.2f}"
    print(n_row)
    print(r2_row)
    print(vif_row)
    print("=" * len(header))
    print("* p<0.10, ** p<0.05, *** p<0.01  (country-clustered SE)\n")


# ── VIF Table ────────────────────────────────────────────────────────────────

def print_vif_table(results: OrderedDict) -> None:
    """Print KOFGIdf VIF across all specifications."""
    print("=" * 70)
    print("VIF FOR KOFGIdf ACROSS SPECIFICATIONS")
    print("=" * 70)
    for label, r in results.items():
        sig = "***" if r["pval"] < 0.01 else (
            "**" if r["pval"] < 0.05 else (
                "*" if r["pval"] < 0.10 else " "))
        print(f"  {label:<20s}  VIF = {r['vif']:>8.2f}  "
              f"  β = {r['coef']:>+.5f}{sig}  "
              f"  p = {r['pval']:.3f}")
    print()


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_coefficient_progression(results: OrderedDict) -> str:
    """Coefficient progression plot: KOFGIdf across 7 specifications."""
    labels = [r["label"] for r in results.values()]
    coefs = [r["coef"] for r in results.values()]
    ci_lo = [r["ci_lo"] for r in results.values()]
    ci_hi = [r["ci_hi"] for r in results.values()]

    x = np.arange(len(labels))
    err_lo = [c - lo for c, lo in zip(coefs, ci_lo)]
    err_hi = [hi - c for c, hi in zip(coefs, ci_hi)]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shade significant models
    for i in range(len(labels)):
        if ci_lo[i] > 0 or ci_hi[i] < 0:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.08, color="#27ae60")

    ax.errorbar(x, coefs, yerr=[err_lo, err_hi],
                fmt="o", color="#2c3e50", capsize=6, markersize=9,
                linewidth=1.8, capthick=1.3)
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("KOFGIdf coefficient (95% CI)", fontsize=11)
    ax.set_title("KOFGIdf Effect on W: Sensitivity to Temporal Controls",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "fig_b1_coefficient_progression.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Findings Summary ─────────────────────────────────────────────────────────

def print_findings_summary(results: OrderedDict) -> None:
    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY")
    print("=" * 70)

    # Identify tipping point
    sig_models = []
    nonsig_models = []
    for label, r in results.items():
        if r["pval"] < 0.10:
            sig_models.append(r)
        else:
            nonsig_models.append(r)

    print("\nCoefficient trajectory:")
    for label, r in results.items():
        status = "significant" + _star(r["pval"]) if r["pval"] < 0.10 \
            else "NOT significant"
        print(f"  Model {r['num']} ({r['label']:<10s}): "
              f"β = {r['coef']:>+.5f}, p = {r['pval']:.3f} — {status}")

    # Find tipping point
    prev_sig = True
    tipping = None
    for label, r in results.items():
        currently_sig = r["pval"] < 0.10
        if prev_sig and not currently_sig:
            tipping = r
            break
        prev_sig = currently_sig

    if tipping:
        print(f"\nTipping point: Model {tipping['num']} ({tipping['label']})")
        print(f"  The KOFGIdf effect becomes insignificant when "
              f"'{tipping['label']}' temporal controls are added.")
        if tipping["num"] <= 4:
            print("  This suggests a specific historical confounder "
                  "(not just a smooth time trend) absorbs the effect.")
        elif tipping["num"] <= 6:
            print("  This suggests a smooth upward drift — not a specific "
                  "event — is the confound. KOFGIdf and W share a common "
                  "trend.")
        else:
            print("  The effect survives all targeted confounders and only "
                  "dies with full year FE. This suggests year FE may be "
                  "absorbing legitimate variation, not just confounds.")
    elif sig_models:
        print("\nThe KOFGIdf effect remains significant across all "
              "specifications — it is robust to temporal controls.")
    else:
        print("\nThe KOFGIdf effect is not significant in any specification.")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("OPTION B: Targeted Confounders Replacing Year FE")
    print("=" * 70)
    print("\nLoading panel data...")
    panel = build_panel()
    panel = prepare_panel(panel)
    print(f"Panel: {len(panel)} obs, "
          f"{panel['country_label'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")

    # Run all 7 models
    results = run_all_models(panel)

    # Regression table
    print_regression_table(results)

    # VIF table
    print_vif_table(results)

    # Plot
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_coefficient_progression(results)

    # Summary
    print_findings_summary(results)
    print("Done.")


if __name__ == "__main__":
    main()
