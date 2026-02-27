"""
Option A: Country-Specific Globalization Shocks
================================================

Tests whether discrete, country-specific globalization shocks affect
winning coalition size (W) in ASEAN-9. Three analyses:

  1. Pooled shock model with all 9 event dummies
  2. Pre/post mean comparisons per shock (±10-year windows)
  3. Event-study plots for Vietnam WTO, Myanmar opening, Indonesia AFC
"""

import os
import sys

# Allow imports from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind

from asean_globalization_analysis import (
    DATA_DIR, YEAR_MIN, YEAR_MAX, build_panel,
)
from asean_subindex_analysis import _star

# ── Constants ────────────────────────────────────────────────────────────────

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SHOCKS = [
    ("Vietnam",   "vnm_doimoi",    1986, "Đổi Mới"),
    ("Vietnam",   "vnm_wto",       2007, "WTO Accession"),
    ("Myanmar",   "mmr_sanctions", 1997, "Sanctions Tightening"),
    ("Myanmar",   "mmr_opening",   2011, "Post-Junta Opening"),
    ("Cambodia",  "khm_wto",       2004, "WTO Accession"),
    ("Laos",      "lao_asean",     1997, "ASEAN Membership"),
    ("Laos",      "lao_wto",       2013, "WTO Accession"),
    ("Indonesia", "idn_afc",       1997, "Asian Financial Crisis"),
    ("Thailand",  "tha_afc",       1997, "Asian Financial Crisis"),
]

SHOCK_COLS = [s[1] for s in SHOCKS]

EVENT_STUDIES = [
    ("Vietnam",   "vnm_wto",       2007, "Vietnam WTO Accession"),
    ("Myanmar",   "mmr_opening",   2011, "Myanmar Post-Junta Opening"),
    ("Indonesia", "idn_afc",       1997, "Indonesia Asian Financial Crisis"),
]


# ── Shock Coding ─────────────────────────────────────────────────────────────

def code_shocks(panel: pd.DataFrame) -> pd.DataFrame:
    """Add country-specific shock dummy columns to the panel."""
    panel = panel.copy()
    for country, col, year, _ in SHOCKS:
        panel[col] = (
            (panel["country_label"] == country) & (panel["year"] >= year)
        ).astype(float)
    return panel


# ── Analysis 1: Pooled Shock Model ──────────────────────────────────────────

def run_pooled_shock_model(panel: pd.DataFrame):
    """W4 ~ shock dummies + e_gdppc + log_pop + Country FE (no year FE)."""
    country_dum = pd.get_dummies(panel["country_label"], drop_first=True,
                                 dtype=float)
    X = pd.concat(
        [panel[SHOCK_COLS + ["e_gdppc", "log_pop"]], country_dum], axis=1,
    )
    X = sm.add_constant(X)
    y = panel["W4"]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["country_label"]},
    )

    print("\n" + "=" * 70)
    print("ANALYSIS 1 — Pooled Shock Model")
    print("W4 ~ shock dummies + e_gdppc + log_pop + Country FE")
    print("(country-clustered SE, no year FE)")
    print("=" * 70)
    print(model.summary())
    return model


def print_shock_regression_table(model) -> None:
    """Print a focused table of shock coefficients only."""
    print("\n" + "=" * 70)
    print("SHOCK COEFFICIENT SUMMARY")
    print("=" * 70)
    header = (f"{'Country':<14s} {'Event':<26s} {'Coef':>9s} {'SE':>9s}"
              f" {'p':>7s}  {'95% CI':>20s}")
    print(header)
    print("-" * len(header))

    ci = model.conf_int()
    for country, col, year, label in SHOCKS:
        c = model.params[col]
        s = model.bse[col]
        p = model.pvalues[col]
        lo = ci.loc[col].iloc[0]
        hi = ci.loc[col].iloc[1]
        stars = _star(p)
        event_str = f"{label} ({year})"
        print(f"{country:<14s} {event_str:<26s} {c:>+8.4f}{stars}"
              f" {s:>8.4f} {p:>7.3f}  [{lo:>+8.4f}, {hi:>+8.4f}]")

    print("-" * len(header))
    for var in ["e_gdppc", "log_pop"]:
        c = model.params[var]
        s = model.bse[var]
        p = model.pvalues[var]
        stars = _star(p)
        print(f"{'':14s} {var:<26s} {c:>+8.4f}{stars}"
              f" {s:>8.4f} {p:>7.3f}")
    print(f"\nN = {int(model.nobs)}    R² = {model.rsquared:.3f}"
          f"    Country FE: Yes    Year FE: No")
    print("* p<0.10, ** p<0.05, *** p<0.01  (country-clustered SE)\n")


# ── Analysis 2: Pre/Post Comparisons ────────────────────────────────────────

def run_prepost_comparisons(panel: pd.DataFrame) -> pd.DataFrame:
    """Symmetric-window pre/post mean comparisons for each shock."""
    results = []
    for country, col, year, label in SHOCKS:
        sub = panel[panel["country_label"] == country].copy()
        win_start = max(YEAR_MIN, year - 10)
        win_end = min(YEAR_MAX, year + 9)
        sub = sub[(sub["year"] >= win_start) & (sub["year"] <= win_end)]

        pre = sub[sub["year"] < year]["W4"]
        post = sub[sub["year"] >= year]["W4"]

        if len(pre) < 2 or len(post) < 2:
            continue

        t_stat, p_val = ttest_ind(post.values, pre.values, equal_var=False)
        results.append({
            "country": country,
            "event": f"{label} ({year})",
            "pre_mean": pre.mean(),
            "post_mean": post.mean(),
            "diff": post.mean() - pre.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
            "n_pre": len(pre),
            "n_post": len(post),
        })

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("ANALYSIS 2 — Pre/Post Mean Comparisons (±10-year windows)")
    print("=" * 70)
    header = (f"{'Country':<12s} {'Event':<26s} {'Pre':>6s} {'Post':>6s}"
              f" {'Diff':>7s} {'t':>7s} {'p':>7s} {'n_pre':>5s} {'n_post':>6s}")
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        stars = _star(r["p_value"])
        print(f"{r['country']:<12s} {r['event']:<26s} "
              f"{r['pre_mean']:>6.3f} {r['post_mean']:>6.3f} "
              f"{r['diff']:>+6.3f}{stars} {r['t_stat']:>6.2f} "
              f"{r['p_value']:>7.3f} {r['n_pre']:>5.0f} {r['n_post']:>6.0f}")
    print("-" * len(header))
    print("Welch's t-test (unequal variance). "
          "* p<0.10, ** p<0.05, *** p<0.01\n")
    return df


# ── Analysis 3: Event-Study Regressions ─────────────────────────────────────

def run_event_study(panel: pd.DataFrame, country: str, event_year: int,
                    col_name: str, label: str) -> dict:
    """DiD-style event study for a single shock on the full panel."""
    p = panel.copy()
    p["event_time"] = p["year"] - event_year
    treated = (p["country_label"] == country)

    tau_min = max(-10, YEAR_MIN - event_year)
    tau_max = min(10, YEAR_MAX - event_year)
    taus = [t for t in range(tau_min, tau_max + 1) if t != -1]

    # Create event-time dummies for the treated country
    tau_cols = []
    for t in taus:
        suffix = f"n{abs(t)}" if t < 0 else str(t)
        cname = f"tau_{suffix}"
        p[cname] = (treated & (p["event_time"] == t)).astype(float)
        tau_cols.append((t, cname))

    # Design matrix: tau dummies + controls + country FE (no year FE)
    country_dum = pd.get_dummies(p["country_label"], drop_first=True,
                                 dtype=float)
    X = pd.concat(
        [p[[c for _, c in tau_cols] + ["e_gdppc", "log_pop"]], country_dum],
        axis=1,
    )
    X = sm.add_constant(X)
    y = p["W4"]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    # Extract coefficients
    ci = model.conf_int()
    result_taus = []
    result_coefs = []
    result_ci_lo = []
    result_ci_hi = []

    for t, cname in tau_cols:
        result_taus.append(t)
        result_coefs.append(model.params[cname])
        result_ci_lo.append(ci.loc[cname].iloc[0])
        result_ci_hi.append(ci.loc[cname].iloc[1])

    # Insert reference period (τ = -1) as zero
    ref_idx = sum(1 for t in result_taus if t < -1)
    result_taus.insert(ref_idx, -1)
    result_coefs.insert(ref_idx, 0.0)
    result_ci_lo.insert(ref_idx, 0.0)
    result_ci_hi.insert(ref_idx, 0.0)

    return {
        "taus": result_taus,
        "coefs": result_coefs,
        "ci_lo": result_ci_lo,
        "ci_hi": result_ci_hi,
        "label": label,
        "country": country,
        "event_year": event_year,
    }


def plot_event_study(es: dict, filename: str) -> str:
    """Generate event-study coefficient plot."""
    taus = np.array(es["taus"])
    coefs = np.array(es["coefs"])
    ci_lo = np.array(es["ci_lo"])
    ci_hi = np.array(es["ci_hi"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(taus, ci_lo, ci_hi, alpha=0.18, color="#3498db")
    ax.plot(taus, coefs, "o-", color="#2c3e50", markersize=5, linewidth=1.5)
    ax.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=1.0,
               alpha=0.8, label="Event year")
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Years relative to event", fontsize=11)
    ax.set_ylabel("Estimated effect on W (vs. τ = −1)", fontsize=11)
    ax.set_title(
        f"Event Study: {es['label']}\n({es['event_year']}, HC1 robust SE)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Findings Summary ─────────────────────────────────────────────────────────

def print_findings_summary(model, prepost_df: pd.DataFrame,
                           event_studies: list) -> None:
    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY")
    print("=" * 70)

    # Pooled model
    print("\nAnalysis 1 — Pooled shock model:")
    sig_shocks = []
    for country, col, year, label in SHOCKS:
        p = model.pvalues[col]
        if p < 0.10:
            c = model.params[col]
            direction = "increases" if c > 0 else "decreases"
            sig_shocks.append(f"  {country} {label} ({year}): "
                              f"β = {c:+.4f}, p = {p:.3f} — "
                              f"{direction} W{_star(p)}")
    if sig_shocks:
        print("  Significant shocks (p < 0.10):")
        for s in sig_shocks:
            print(s)
    else:
        print("  No shock reaches significance at p < 0.10.")

    # Pre/post
    print("\nAnalysis 2 — Pre/post comparisons:")
    sig_pp = prepost_df[prepost_df["p_value"] < 0.10]
    if len(sig_pp):
        for _, r in sig_pp.iterrows():
            print(f"  {r['country']} {r['event']}: "
                  f"Δ = {r['diff']:+.3f}, p = {r['p_value']:.3f}")
    else:
        print("  No shock shows a significant pre/post difference (p < 0.10).")

    # Event studies
    print("\nAnalysis 3 — Event-study pre-trend assessment:")
    for es in event_studies:
        pre_taus = [i for i, t in enumerate(es["taus"]) if t < -1]
        if pre_taus:
            pre_coefs = [es["coefs"][i] for i in pre_taus]
            pre_ci_lo = [es["ci_lo"][i] for i in pre_taus]
            pre_ci_hi = [es["ci_hi"][i] for i in pre_taus]
            any_sig = any(lo > 0 or hi < 0
                         for lo, hi in zip(pre_ci_lo, pre_ci_hi))
            trend = "PRE-TREND DETECTED" if any_sig else "no pre-trend"
            print(f"  {es['label']}: {trend} "
                  f"(pre-period coefs range: "
                  f"[{min(pre_coefs):+.3f}, {max(pre_coefs):+.3f}])")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("OPTION A: Country-Specific Globalization Shocks")
    print("=" * 70)
    print("\nLoading panel data...")
    panel = build_panel()
    panel = code_shocks(panel)
    print(f"Panel: {len(panel)} obs, "
          f"{panel['country_label'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")

    # Analysis 1
    model = run_pooled_shock_model(panel)
    print_shock_regression_table(model)

    # Analysis 2
    prepost_df = run_prepost_comparisons(panel)

    # Analysis 3
    print("\n" + "=" * 70)
    print("ANALYSIS 3 — Event-Study Regressions")
    print("=" * 70)

    filenames = [
        "fig_a1_event_study_vnm_wto.png",
        "fig_a2_event_study_mmr_opening.png",
        "fig_a3_event_study_idn_afc.png",
    ]
    event_studies = []
    for (country, col, year, label), fname in zip(EVENT_STUDIES, filenames):
        print(f"\nRunning event study: {label} ({year})...")
        es = run_event_study(panel, country, year, col, label)
        plot_event_study(es, fname)
        event_studies.append(es)

    # Summary
    print_findings_summary(model, prepost_df, event_studies)
    print("Done.")


if __name__ == "__main__":
    main()
