"""
Preliminary Analysis: Effects of Globalization on ASEAN Member States
=====================================================================

Examines the relationship between de facto globalization (KOFGIdf) and
winning coalition size (W4) across 9 ASEAN member states (1970-2017),
controlling for GDP per capita and population.

Models:
  1. Baseline pooled OLS with country FE
  2. Interaction model (KOFGIdf × GDP per capita)
  3. Country-by-country OLS for heterogeneity
  4. Period interactions (pre/post 1997 AFC and 1991 Cold War)

Data sources:
  - W measure: NewWmeasure.csv (Bueno de Mesquita et al.)
  - KOF Globalisation Index: KOFGI_2019_index.xlsx (ETH Zurich)
  - V-Dem: vdem.RData (e_gdppc, e_pop)
"""

import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
YEAR_MIN, YEAR_MAX = 1970, 2017

VDEM_URL = (
    "https://raw.githubusercontent.com/vdeminstitute/vdemdata/"
    "master/data/vdem.RData"
)

# Mapping table bridging COW codes, ISO3 codes, and dataset-specific names
ASEAN_MAP = pd.DataFrame([
    {"country_label": "Cambodia",    "cow_code": 811, "iso3": "KHM"},
    {"country_label": "Indonesia",   "cow_code": 850, "iso3": "IDN"},
    {"country_label": "Laos",        "cow_code": 812, "iso3": "LAO"},
    {"country_label": "Malaysia",    "cow_code": 820, "iso3": "MYS"},
    {"country_label": "Myanmar",     "cow_code": 775, "iso3": "MMR"},
    {"country_label": "Philippines", "cow_code": 840, "iso3": "PHL"},
    {"country_label": "Singapore",   "cow_code": 830, "iso3": "SGP"},
    {"country_label": "Thailand",    "cow_code": 800, "iso3": "THA"},
    {"country_label": "Vietnam",     "cow_code": 816, "iso3": "VNM"},
])

ASEAN_COW = set(ASEAN_MAP["cow_code"])
ASEAN_ISO3 = set(ASEAN_MAP["iso3"])


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_w(path: str) -> pd.DataFrame:
    """Load W dataset filtered to ASEAN-9 and analysis window."""
    df = pd.read_csv(path)
    df["ccode"] = df["ccode"].astype(int)
    df = df[df["ccode"].isin(ASEAN_COW)].copy()
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df["W4"] = pd.to_numeric(df["W4"], errors="coerce")
    df = df.merge(ASEAN_MAP[["cow_code", "country_label"]],
                  left_on="ccode", right_on="cow_code")
    return df[["country_label", "year", "W4"]].copy()


def load_kof(path: str) -> pd.DataFrame:
    """Load KOF dataset filtered to ASEAN-9 and analysis window."""
    df = pd.read_excel(path, sheet_name="KOFGI_2019_data", engine="openpyxl")
    df = df[df["code"].isin(ASEAN_ISO3)].copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df = df.merge(ASEAN_MAP[["iso3", "country_label"]],
                  left_on="code", right_on="iso3")
    return df[["country_label", "year", "KOFGIdf"]].copy()


def load_vdem(path: str) -> pd.DataFrame:
    """Load V-Dem data filtered to ASEAN-9 and analysis window."""
    vdem_path = os.path.join(DATA_DIR, "vdem.RData")
    if not os.path.exists(vdem_path):
        print("Downloading V-Dem data (~33 MB)...")
        urllib.request.urlretrieve(VDEM_URL, vdem_path)
    result = pyreadr.read_r(vdem_path)
    df = result[list(result.keys())[0]]
    df = df[df["COWcode"].isin(ASEAN_COW)].copy()
    df["year"] = df["year"].astype(int)
    df["COWcode"] = df["COWcode"].astype(int)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df = df.merge(ASEAN_MAP[["cow_code", "country_label"]],
                  left_on="COWcode", right_on="cow_code")
    return df[["country_label", "year", "e_gdppc", "e_pop"]].copy()


def build_panel() -> pd.DataFrame:
    """Merge W, KOF, and V-Dem into a single analysis panel."""
    w = load_w(os.path.join(DATA_DIR, "NewWmeasure.csv"))
    kof = load_kof(os.path.join(DATA_DIR, "KOFGI_2019_index.xlsx"))
    vdem = load_vdem(DATA_DIR)

    panel = w.merge(kof, on=["country_label", "year"], how="inner")
    panel = panel.merge(vdem, on=["country_label", "year"], how="inner")
    panel["log_pop"] = np.log(panel["e_pop"])
    panel = panel.dropna(subset=["W4", "KOFGIdf", "e_gdppc", "log_pop"])
    panel["post_afc"] = (panel["year"] >= 1998).astype(float)
    panel["post_cw"] = (panel["year"] >= 1992).astype(float)
    panel = panel.sort_values(["country_label", "year"]).reset_index(drop=True)
    return panel


# ── Diagnostics ──────────────────────────────────────────────────────────────

def print_diagnostics(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PANEL DIAGNOSTICS")
    print("=" * 70)
    print(f"Observations: {len(panel)}")
    print(f"Countries:    {panel['country_label'].nunique()}")
    print(f"Year range:   {panel['year'].min()}-{panel['year'].max()}")
    print(f"\nObservations per country:")
    print(panel.groupby("country_label").size().to_string())
    print(f"\nVariable summaries:")
    print(panel[["W4", "KOFGIdf", "e_gdppc", "e_pop", "log_pop"]]
          .describe().round(3).to_string())
    print()


def print_collinearity_diagnostics(panel: pd.DataFrame) -> None:
    """VIF and within-country correlations for KOFGIdf vs e_gdppc."""
    print("\n" + "=" * 70)
    print("COLLINEARITY DIAGNOSTICS")
    print("=" * 70)

    # VIF for Model 1 regressors (including country FE)
    dummies = pd.get_dummies(panel["country_label"], drop_first=True,
                             dtype=float)
    X = pd.concat([panel[["KOFGIdf", "e_gdppc", "log_pop"]], dummies], axis=1)
    X = sm.add_constant(X)

    substantive = ["KOFGIdf", "e_gdppc", "log_pop"]
    print("\nVariance Inflation Factors (Model 1 specification):")
    for var in substantive:
        idx = list(X.columns).index(var)
        vif = variance_inflation_factor(X.values, idx)
        print(f"  {var:<12s}  VIF = {vif:.2f}")

    # Within-country Pearson correlations
    print("\nWithin-country correlation (KOFGIdf vs e_gdppc):")
    corrs = (panel.groupby("country_label")[["KOFGIdf", "e_gdppc"]]
             .corr().iloc[0::2, -1].droplevel(1))
    for country, r in corrs.items():
        print(f"  {country:<14s}  r = {r:+.3f}")
    print(f"\n  {'Mean':<14s}  r = {corrs.mean():+.3f}")
    print()


# ── Models ───────────────────────────────────────────────────────────────────

def run_model1(panel: pd.DataFrame):
    """Baseline pooled OLS with country FE:
    W4 ~ KOFGIdf + e_gdppc + log_pop + C(country)
    """
    dummies = pd.get_dummies(panel["country_label"], drop_first=True,
                             dtype=float)
    X = pd.concat([panel[["KOFGIdf", "e_gdppc", "log_pop"]], dummies], axis=1)
    X = sm.add_constant(X)
    y = panel["W4"]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    print("\n" + "=" * 70)
    print("MODEL 1: Baseline Pooled OLS with Country FE")
    print("W4 ~ KOFGIdf + e_gdppc + log_pop + Country FE")
    print("=" * 70)
    print(model.summary())
    return model


def run_model2(panel: pd.DataFrame):
    """Interaction model:
    W4 ~ KOFGIdf * e_gdppc + log_pop + C(country)
    """
    panel = panel.copy()
    panel["KOFGIdf_x_gdppc"] = panel["KOFGIdf"] * panel["e_gdppc"]

    dummies = pd.get_dummies(panel["country_label"], drop_first=True,
                             dtype=float)
    X = pd.concat([
        panel[["KOFGIdf", "e_gdppc", "KOFGIdf_x_gdppc", "log_pop"]],
        dummies
    ], axis=1)
    X = sm.add_constant(X)
    y = panel["W4"]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    print("\n" + "=" * 70)
    print("MODEL 2: Interaction Model (KOFGIdf × GDP per capita)")
    print("W4 ~ KOFGIdf * e_gdppc + log_pop + Country FE")
    print("=" * 70)
    print(model.summary())
    return model


def run_country_models(panel: pd.DataFrame) -> pd.DataFrame:
    """Country-by-country OLS: W4 ~ KOFGIdf + e_gdppc + log_pop."""
    results = []
    for country in sorted(panel["country_label"].unique()):
        sub = panel[panel["country_label"] == country]
        X = sm.add_constant(sub[["KOFGIdf", "e_gdppc", "log_pop"]])
        y = sub["W4"]
        res = sm.OLS(y, X).fit(cov_type="HC1")
        ci = res.conf_int().loc["KOFGIdf"]
        results.append({
            "country": country,
            "coef": res.params["KOFGIdf"],
            "se": res.bse["KOFGIdf"],
            "ci_low": ci.iloc[0],
            "ci_high": ci.iloc[1],
            "nobs": int(res.nobs),
            "r2": res.rsquared,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("MODEL 3: Country-by-Country OLS")
    print("W4 ~ KOFGIdf + e_gdppc + log_pop (per country)")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print()
    return results_df


def run_period_models(panel: pd.DataFrame) -> dict:
    """Period interaction models testing temporal heterogeneity in KOF effect.

    Two cutpoints:
      - 1997 Asian Financial Crisis (post_afc: year >= 1998)
      - 1991 End of Cold War (post_cw: year >= 1992)
    """
    cutpoints = [
        ("afc", "post_afc", "Asian Financial Crisis (pre/post 1997)"),
        ("cw",  "post_cw",  "End of Cold War (pre/post 1991)"),
    ]
    results = {}

    for key, dummy_col, label in cutpoints:
        p = panel.copy()
        inter_col = f"KOFGIdf_x_{dummy_col}"
        p[inter_col] = p["KOFGIdf"] * p[dummy_col]

        dummies = pd.get_dummies(p["country_label"], drop_first=True,
                                 dtype=float)
        X = pd.concat([
            p[["KOFGIdf", dummy_col, inter_col, "e_gdppc", "log_pop"]],
            dummies
        ], axis=1)
        X = sm.add_constant(X)
        y = p["W4"]
        model = sm.OLS(y, X).fit(cov_type="HC1")

        # Implied KOF effects by period
        b_kof = model.params["KOFGIdf"]
        b_inter = model.params[inter_col]
        V = model.cov_params()
        se_pre = model.bse["KOFGIdf"]
        se_post = np.sqrt(
            V.loc["KOFGIdf", "KOFGIdf"]
            + V.loc[inter_col, inter_col]
            + 2 * V.loc["KOFGIdf", inter_col]
        )

        print("\n" + "=" * 70)
        print(f"MODEL 4: Period Interaction — {label}")
        print(f"W4 ~ KOFGIdf + {dummy_col} + KOFGIdf×{dummy_col}"
              " + e_gdppc + log_pop + Country FE")
        print("=" * 70)
        print(model.summary())
        print(f"\nImplied KOF effect:")
        print(f"  Pre-period:  {b_kof:+.5f}  (SE = {se_pre:.5f})")
        print(f"  Post-period: {b_kof + b_inter:+.5f}  (SE = {se_post:.5f})")
        print(f"  Difference (interaction): {b_inter:+.5f}"
              f"  (p = {model.pvalues[inter_col]:.4f})")
        print()

        results[key] = {
            "model": model,
            "inter_col": inter_col,
            "dummy_col": dummy_col,
            "label": label,
            "pre_coef": b_kof,
            "pre_se": se_pre,
            "post_coef": b_kof + b_inter,
            "post_se": se_post,
            "inter_coef": b_inter,
            "inter_pval": model.pvalues[inter_col],
        }

    return results


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_baseline_coefficients(model, panel: pd.DataFrame) -> str:
    """Figure 1: Standardized coefficient plot for Model 1."""
    # Re-run with standardized variables for comparable coefficient magnitudes
    panel_z = panel.copy()
    for var in ["KOFGIdf", "e_gdppc", "log_pop"]:
        panel_z[var] = zscore(panel_z[var].values)

    dummies = pd.get_dummies(panel_z["country_label"], drop_first=True,
                             dtype=float)
    X = pd.concat([panel_z[["KOFGIdf", "e_gdppc", "log_pop"]], dummies],
                  axis=1)
    X = sm.add_constant(X)
    y = panel_z["W4"]
    model_z = sm.OLS(y, X).fit(cov_type="HC1")

    vars_plot = ["KOFGIdf", "e_gdppc", "log_pop"]
    labels = ["KOF de facto\nGlobalization", "GDP per capita", "log(Population)"]
    coefs = [model_z.params[v] for v in vars_plot]
    ci = model_z.conf_int()
    ci_lo = [ci.loc[v].iloc[0] for v in vars_plot]
    ci_hi = [ci.loc[v].iloc[1] for v in vars_plot]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = np.arange(len(vars_plot))
    err_lo = [c - lo for c, lo in zip(coefs, ci_lo)]
    err_hi = [hi - c for c, hi in zip(coefs, ci_hi)]

    ax.errorbar(coefs, y_pos, xerr=[err_lo, err_hi],
                fmt="o", color="#2c3e50", capsize=5, markersize=8,
                linewidth=1.5, capthick=1.2)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Standardized coefficient (95% CI)", fontsize=11)
    ax.set_title("Model 1: Determinants of W (Pooled OLS, Country FE)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(DATA_DIR, "fig1_baseline_coefficients.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_marginal_effects(model, panel: pd.DataFrame) -> str:
    """Figure 3: Marginal effect of KOFGIdf on W4 across GDP per capita."""
    b_kof = model.params["KOFGIdf"]
    b_inter = model.params["KOFGIdf_x_gdppc"]
    V = model.cov_params()

    gdp_range = np.linspace(panel["e_gdppc"].min(),
                            panel["e_gdppc"].max(), 300)

    me = b_kof + b_inter * gdp_range
    se = np.sqrt(
        V.loc["KOFGIdf", "KOFGIdf"]
        + gdp_range ** 2 * V.loc["KOFGIdf_x_gdppc", "KOFGIdf_x_gdppc"]
        + 2 * gdp_range * V.loc["KOFGIdf", "KOFGIdf_x_gdppc"]
    )
    ci_lo = me - 1.96 * se
    ci_hi = me + 1.96 * se

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gdp_range, me, color="#2c3e50", linewidth=2)
    ax.fill_between(gdp_range, ci_lo, ci_hi, alpha=0.18, color="#3498db")
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)

    # Mark each country's median GDP per capita as triangles on the zero line
    countries_sorted = sorted(
        panel["country_label"].unique(),
        key=lambda c: panel.loc[panel["country_label"] == c, "e_gdppc"].median()
    )
    median_gdps = []
    for country in countries_sorted:
        med = panel.loc[panel["country_label"] == country, "e_gdppc"].median()
        median_gdps.append((country, med))
        ax.plot(med, 0, marker="v", color="#e74c3c", markersize=6,
                alpha=0.7, zorder=5)

    # Add a text box listing countries and their median GDP per capita
    legend_text = "Median GDP p.c.:\n" + "\n".join(
        f"  {c}: {g:.1f}" for c, g in median_gdps
    )
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
            fontsize=6.5, verticalalignment="bottom",
            horizontalalignment="right", color="#555555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
            family="monospace")

    ax.set_xlabel("GDP per capita", fontsize=11)
    ax.set_ylabel("Marginal effect of KOFGIdf on W", fontsize=11)
    ax.set_title(
        "Model 2: Marginal Effect of De Facto Globalization on W\n"
        "by GDP per Capita (with 95% CI)",
        fontsize=12, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(DATA_DIR, "fig3_marginal_effect_kof.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_country_heterogeneity(results_df: pd.DataFrame) -> str:
    """Figure 2: Country-level KOFGIdf coefficients."""
    df = results_df.sort_values("coef").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    y_pos = np.arange(len(df))
    err_lo = df["coef"] - df["ci_low"]
    err_hi = df["ci_high"] - df["coef"]

    ax.errorbar(df["coef"], y_pos, xerr=[err_lo.values, err_hi.values],
                fmt="o", color="#2c3e50", capsize=5, markersize=8,
                linewidth=1.5, capthick=1.2)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["country"], fontsize=11)
    ax.set_xlabel("KOFGIdf coefficient on W (95% CI)", fontsize=11)
    ax.set_title(
        "Country-Level Heterogeneity:\n"
        "Effect of De Facto Globalization on W by Country",
        fontsize=12, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(DATA_DIR, "fig2_country_kof_coefficients.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_period_interactions(period_results: dict) -> str:
    """Figure 4: KOF effect on W before vs. after two structural breaks."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    panel_info = [
        ("afc", "Asian Financial Crisis (1997)",
         f"Pre ({YEAR_MIN}\u20131997)", f"Post (1998\u2013{YEAR_MAX})"),
        ("cw",  "End of Cold War (1991)",
         f"Pre ({YEAR_MIN}\u20131991)", f"Post (1992\u2013{YEAR_MAX})"),
    ]

    for ax, (key, title, pre_label, post_label) in zip(axes, panel_info):
        r = period_results[key]
        labels = [post_label, pre_label]
        coefs = [r["post_coef"], r["pre_coef"]]
        ses = [r["post_se"], r["pre_se"]]
        colors = ["#e74c3c", "#3498db"]
        y_pos = np.arange(2)

        for i in range(2):
            ci_lo = coefs[i] - 1.96 * ses[i]
            ci_hi = coefs[i] + 1.96 * ses[i]
            ax.errorbar(coefs[i], y_pos[i],
                        xerr=[[coefs[i] - ci_lo], [ci_hi - coefs[i]]],
                        fmt="o", color=colors[i], capsize=5, markersize=8,
                        linewidth=1.5, capthick=1.2)

        ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8,
                   alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("KOFGIdf coefficient on W (95% CI)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        p_val = r["inter_pval"]
        sig = ("***" if p_val < 0.001 else
               "**" if p_val < 0.01 else
               "*" if p_val < 0.05 else "n.s.")
        ax.text(0.98, 0.98,
                f"\u0394 = {r['inter_coef']:+.4f}\np = {p_val:.3f} {sig}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Model 4: Temporal Heterogeneity in the Effect of "
        "De Facto Globalization on W",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    path = os.path.join(DATA_DIR, "fig4_period_kof_effects.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading and merging datasets...")
    panel = build_panel()
    print_diagnostics(panel)
    print_collinearity_diagnostics(panel)

    # Run models
    m1 = run_model1(panel)
    m2 = run_model2(panel)
    country_results = run_country_models(panel)
    period_results = run_period_models(panel)

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_baseline_coefficients(m1, panel)
    plot_country_heterogeneity(country_results)
    plot_marginal_effects(m2, panel)
    plot_period_interactions(period_results)

    print("\nDone. Four coefficient plots saved to:", DATA_DIR)


if __name__ == "__main__":
    main()
