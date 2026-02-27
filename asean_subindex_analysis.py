"""
KOF Sub-Index Decomposition: Effects on Winning Coalition Size
==============================================================

Decomposes the composite KOFGIdf into four de facto sub-indices and
examines which dimension of globalization drives the W relationship
across 9 ASEAN member states (1970-2017).

Step 1: Separate pooled OLS (two-way FE, country-clustered SE) for each
        sub-index — Trade, Financial, Interpersonal, Informational.
Step 2: Horse-race model combining significant sub-indices.
Step 3: VIF diagnostics and within-country correlation matrix.

Data sources:
  - W measure: NewWmeasure.csv (Bueno de Mesquita et al.)
  - KOF Globalisation Index: KOFGI_2019_index.xlsx (ETH Zurich)
  - V-Dem: vdem.RData (e_gdppc, e_pop)
"""

import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from asean_globalization_analysis import (
    DATA_DIR, YEAR_MIN, YEAR_MAX, ASEAN_MAP, ASEAN_ISO3, load_w, load_vdem,
)

# ── Constants ────────────────────────────────────────────────────────────────

SUB_INDICES = ["KOFTrGIdf", "KOFFiGIdf", "KOFIpGIdf", "KOFInGIdf"]

SUB_LABELS = OrderedDict([
    ("KOFTrGIdf",  "Trade"),
    ("KOFFiGIdf",  "Financial"),
    ("KOFIpGIdf",  "Interpersonal"),
    ("KOFInGIdf",  "Informational"),
])

SUB_COLORS = {
    "KOFTrGIdf":  "#2c3e50",
    "KOFFiGIdf":  "#e74c3c",
    "KOFIpGIdf":  "#3498db",
    "KOFInGIdf":  "#27ae60",
}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_kof_subindices(path: str) -> pd.DataFrame:
    """Load KOF dataset with composite index and four de facto sub-indices."""
    df = pd.read_excel(path, sheet_name="KOFGI_2019_data", engine="openpyxl")
    df = df[df["code"].isin(ASEAN_ISO3)].copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df = df.merge(ASEAN_MAP[["iso3", "country_label"]],
                  left_on="code", right_on="iso3")
    keep = ["country_label", "year", "KOFGIdf"] + SUB_INDICES
    return df[keep].copy()


def build_subindex_panel() -> pd.DataFrame:
    """Merge W, KOF (with sub-indices), and V-Dem into analysis panel."""
    w = load_w(os.path.join(DATA_DIR, "NewWmeasure.csv"))
    kof = load_kof_subindices(os.path.join(DATA_DIR, "KOFGI_2019_index.xlsx"))
    vdem = load_vdem(DATA_DIR)

    panel = w.merge(kof, on=["country_label", "year"], how="inner")
    panel = panel.merge(vdem, on=["country_label", "year"], how="inner")
    panel["log_pop"] = np.log(panel["e_pop"])
    na_cols = ["W4", "e_gdppc", "log_pop"] + SUB_INDICES
    panel = panel.dropna(subset=na_cols)
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
    summary_cols = ["W4"] + SUB_INDICES + ["e_gdppc", "log_pop"]
    print(panel[summary_cols].describe().round(3).to_string())
    print()


# ── Step 1: Separate Sub-Index Models ────────────────────────────────────────

def _build_twoway_X(panel: pd.DataFrame, iv_cols: list) -> pd.DataFrame:
    """Build design matrix with two-way FE (country + year dummies)."""
    country_dum = pd.get_dummies(panel["country_label"], drop_first=True,
                                 dtype=float)
    year_dum = pd.get_dummies(panel["year"], prefix="yr", drop_first=True,
                              dtype=float)
    X = pd.concat(
        [panel[iv_cols + ["e_gdppc", "log_pop"]], country_dum, year_dum],
        axis=1,
    )
    X = sm.add_constant(X)
    return X


def run_subindex_models(panel: pd.DataFrame) -> OrderedDict:
    """Step 1: Run four separate models, one per KOF sub-index."""
    results = OrderedDict()

    for idx in SUB_INDICES:
        label = SUB_LABELS[idx]
        X = _build_twoway_X(panel, [idx])
        y = panel["W4"]
        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["country_label"]},
        )
        ci = model.conf_int().loc[idx]

        print("\n" + "=" * 70)
        print(f"STEP 1 — {label} Globalization ({idx})")
        print(f"W4 ~ {idx} + e_gdppc + log_pop + Country FE + Year FE")
        print(f"Clustered SE at country level (k = {panel['country_label'].nunique()})")
        print("=" * 70)
        print(model.summary())

        results[idx] = {
            "model": model,
            "label": label,
            "coef": model.params[idx],
            "se": model.bse[idx],
            "pval": model.pvalues[idx],
            "ci_low": ci.iloc[0],
            "ci_high": ci.iloc[1],
            "r2": model.rsquared,
            "nobs": int(model.nobs),
        }

    return results


# ── Step 2: Horse-Race Model ─────────────────────────────────────────────────

def run_horserace_model(panel: pd.DataFrame,
                        sub_results: OrderedDict) -> dict:
    """Step 2: Combine significant sub-indices in one model."""
    sig_indices = [idx for idx, r in sub_results.items() if r["pval"] < 0.10]

    if not sig_indices:
        print("\nNo sub-index individually significant at p < 0.10.")
        print("Running horse-race with all four sub-indices.\n")
        sig_indices = list(SUB_INDICES)
        all_used = True
    else:
        sig_labels = [SUB_LABELS[i] for i in sig_indices]
        print(f"\nSignificant sub-indices (p < 0.10): "
              f"{', '.join(sig_labels)}")
        all_used = False

    X = _build_twoway_X(panel, sig_indices)
    y = panel["W4"]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["country_label"]},
    )

    header = "all four (none individually significant)" if all_used \
        else ", ".join(SUB_LABELS[i] for i in sig_indices)

    print("\n" + "=" * 70)
    print(f"STEP 2 — Horse-Race Model ({header})")
    spec_vars = " + ".join(sig_indices)
    print(f"W4 ~ {spec_vars} + e_gdppc + log_pop + Country FE + Year FE")
    print("=" * 70)
    print(model.summary())

    return {
        "model": model,
        "indices_used": sig_indices,
        "all_used": all_used,
    }


# ── Step 3: Collinearity Diagnostics ─────────────────────────────────────────

def print_subindex_vifs(panel: pd.DataFrame) -> None:
    """Step 3a: VIF for each separate sub-index specification."""
    print("\n" + "=" * 70)
    print("STEP 3a — VARIANCE INFLATION FACTORS (per model)")
    print("=" * 70)

    substantive = ["e_gdppc", "log_pop"]
    header = f"{'Variable':<14s}"
    for idx in SUB_INDICES:
        header += f"  {SUB_LABELS[idx]:>12s}"
    print(header)
    print("-" * len(header))

    # Build each model's design matrix and compute VIFs
    vif_rows = {}
    for idx in SUB_INDICES:
        X = _build_twoway_X(panel, [idx])
        col_list = list(X.columns)
        vifs = {}
        for var in [idx] + substantive:
            vi = col_list.index(var)
            vifs[var] = variance_inflation_factor(X.values, vi)
        vif_rows[idx] = vifs

    # Print sub-index VIF row
    row = f"{'Sub-index':<14s}"
    for idx in SUB_INDICES:
        row += f"  {vif_rows[idx][idx]:>12.2f}"
    print(row)

    # Print control VIF rows
    for var in substantive:
        row = f"{var:<14s}"
        for idx in SUB_INDICES:
            row += f"  {vif_rows[idx][var]:>12.2f}"
        print(row)

    print(f"\n(Composite KOFGIdf VIF from earlier analysis: 13.66)")
    print()


def print_correlation_matrix(panel: pd.DataFrame) -> None:
    """Step 3b: Within-country correlation matrix of sub-indices + GDP."""
    print("=" * 70)
    print("STEP 3b — WITHIN-COUNTRY CORRELATION MATRIX")
    print("=" * 70)

    cols = SUB_INDICES + ["e_gdppc"]
    # Demean by country to get within-country variation
    demeaned = panel[cols + ["country_label"]].copy()
    for col in cols:
        demeaned[col] = demeaned.groupby("country_label")[col].transform(
            lambda x: x - x.mean()
        )
    corr = demeaned[cols].corr()

    # Pretty-print with short labels
    short = {c: SUB_LABELS.get(c, c)[:8] for c in cols}
    short["e_gdppc"] = "GDP_pc"

    header = f"{'':>14s}" + "".join(f"  {short[c]:>10s}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in cols:
        row = f"{short[r]:>14s}"
        for c in cols:
            row += f"  {corr.loc[r, c]:>10.3f}"
        print(row)
    print()


# ── Regression Table ─────────────────────────────────────────────────────────

def _star(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def print_regression_table(sub_results: OrderedDict,
                           horserace: dict) -> None:
    """Stargazer-style side-by-side regression table."""
    print("\n" + "=" * 78)
    print("REGRESSION TABLE — KOF Sub-Index Models")
    print("=" * 78)

    models = []
    col_headers = []
    for i, (idx, r) in enumerate(sub_results.items(), 1):
        models.append(r["model"])
        col_headers.append(f"({i}) {SUB_LABELS[idx][:5]}")
    models.append(horserace["model"])
    col_headers.append(f"({len(models)}) Horse")
    n_cols = len(models)
    col_w = 13

    # Header
    header = f"{'':>{col_w}}" + "".join(f"{h:>{col_w}}" for h in col_headers)
    print(header)
    print("-" * len(header))

    # Rows for sub-index coefficients
    for idx in SUB_INDICES:
        label = idx
        coef_row = f"{label:>{col_w}}"
        se_row = f"{'':>{col_w}}"
        for m in models:
            if idx in m.params.index:
                c = m.params[idx]
                s = m.bse[idx]
                p = m.pvalues[idx]
                coef_row += f"{c:>{col_w - len(_star(p))}.4f}{_star(p)}"
                se_row += f"{'(' + f'{s:.4f}' + ')':>{col_w}}"
            else:
                coef_row += f"{'':>{col_w}}"
                se_row += f"{'':>{col_w}}"
        print(coef_row)
        print(se_row)

    # Control rows
    for var, label in [("e_gdppc", "e_gdppc"), ("log_pop", "log_pop")]:
        coef_row = f"{label:>{col_w}}"
        se_row = f"{'':>{col_w}}"
        for m in models:
            c = m.params[var]
            s = m.bse[var]
            p = m.pvalues[var]
            coef_row += f"{c:>{col_w - len(_star(p))}.4f}{_star(p)}"
            se_row += f"{'(' + f'{s:.4f}' + ')':>{col_w}}"
        print(coef_row)
        print(se_row)

    # Footer
    print("-" * len(header))
    ft = lambda lbl, vals: f"{lbl:>{col_w}}" + "".join(
        f"{v:>{col_w}}" for v in vals
    )
    print(ft("Country FE", ["Yes"] * n_cols))
    print(ft("Year FE", ["Yes"] * n_cols))
    print(ft("SE cluster", ["Country"] * n_cols))

    n_row = f"{'N':>{col_w}}"
    r2_row = f"{'R²':>{col_w}}"
    for m in models:
        n_row += f"{int(m.nobs):>{col_w}}"
        r2_row += f"{m.rsquared:>{col_w}.3f}"
    print(n_row)
    print(r2_row)
    print("=" * len(header))
    print("* p<0.10, ** p<0.05, *** p<0.01  (country-clustered SE)")
    print()


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_subindex_coefficients(sub_results: OrderedDict) -> str:
    """Figure 5: Sub-index coefficient comparison plot."""
    indices = list(sub_results.keys())
    n = len(indices)

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(n)

    for i, idx in enumerate(indices):
        r = sub_results[idx]
        ci_lo = r["ci_low"]
        ci_hi = r["ci_high"]
        ax.errorbar(
            r["coef"], y_pos[i],
            xerr=[[r["coef"] - ci_lo], [ci_hi - r["coef"]]],
            fmt="o", color=SUB_COLORS[idx], capsize=5,
            markersize=8, linewidth=1.5, capthick=1.2,
        )

    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    labels = [f"{SUB_LABELS[idx]}\n({idx})" for idx in indices]
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Coefficient on W (95% CI, country-clustered SE)",
                  fontsize=11)
    ax.set_title("Step 1: KOF Sub-Index Effects on Winning Coalition Size",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(DATA_DIR, "fig5_subindex_coefficients.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Findings Summary ─────────────────────────────────────────────────────────

def print_findings_summary(sub_results: OrderedDict,
                           horserace: dict) -> None:
    print("\n" + "=" * 70)
    print("FINDINGS SUMMARY")
    print("=" * 70)

    # Step 1 summary
    print("\nStep 1 — Separate specifications:")
    for idx, r in sub_results.items():
        sig = _star(r["pval"])
        direction = "positive" if r["coef"] > 0 else "negative"
        status = f"significant{sig}" if sig else "not significant"
        print(f"  {SUB_LABELS[idx]:<16s} ({idx}): "
              f"β = {r['coef']:+.4f}, p = {r['pval']:.3f} — "
              f"{direction}, {status}")

    # Step 2 summary
    hr = horserace
    print(f"\nStep 2 — Horse-race:")
    if hr["all_used"]:
        print("  No sub-index was individually significant; all four entered.")
    else:
        survivors = []
        for idx in hr["indices_used"]:
            m = hr["model"]
            if idx in m.params.index:
                p = m.pvalues[idx]
                if p < 0.10:
                    survivors.append(SUB_LABELS[idx])
        if survivors:
            print(f"  Survives competition: {', '.join(survivors)}")
        else:
            print("  No sub-index survives at p < 0.10 when combined.")

    for idx in hr["indices_used"]:
        m = hr["model"]
        if idx in m.params.index:
            print(f"  {SUB_LABELS[idx]:<16s}: "
                  f"β = {m.params[idx]:+.4f}, p = {m.pvalues[idx]:.3f}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building panel with KOF sub-indices...")
    panel = build_subindex_panel()
    print_diagnostics(panel)

    # Step 1
    sub_results = run_subindex_models(panel)

    # Step 2
    horserace = run_horserace_model(panel, sub_results)

    # Regression table
    print_regression_table(sub_results, horserace)

    # Step 3
    print_subindex_vifs(panel)
    print_correlation_matrix(panel)

    # Plot
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_subindex_coefficients(sub_results)

    # Summary
    print_findings_summary(sub_results, horserace)

    print("Done.")


if __name__ == "__main__":
    main()
