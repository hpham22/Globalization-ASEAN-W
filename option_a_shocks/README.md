# Option A: Country-Specific Globalization Shocks

Tests whether discrete, country-specific globalization events affect winning coalition size (W) in ASEAN-9 member states. This complements the main analysis where the smooth composite KOF index lost significance under year FE — here we test whether sharp exogenous shocks succeed where the continuous measure failed.

## Shock Events

| Country | Event | Year |
|---------|-------|------|
| Vietnam | Đổi Mới | 1986 |
| Vietnam | WTO Accession | 2007 |
| Myanmar | Sanctions Tightening | 1997 |
| Myanmar | Post-Junta Opening | 2011 |
| Cambodia | WTO Accession | 2004 |
| Laos | ASEAN Membership | 1997 |
| Laos | WTO Accession | 2013 |
| Indonesia | Asian Financial Crisis | 1997 |
| Thailand | Asian Financial Crisis | 1997 |

Singapore, Malaysia, and the Philippines serve as untreated controls.

## Analyses

1. **Pooled shock model** — all 9 shock dummies in a single regression with country FE and country-clustered SEs (no year FE).
2. **Pre/post comparisons** — mean W in symmetric ±10-year windows around each event, with Welch's t-test.
3. **Event-study plots** — DiD-style year-by-year coefficients for Vietnam WTO, Myanmar opening, and Indonesia AFC, showing leads and lags relative to the event year.

## Running

From the repository root:

```bash
python3 option_a_shocks/shock_analysis.py
```

## Outputs

- `fig_a1_event_study_vnm_wto.png`
- `fig_a2_event_study_mmr_opening.png`
- `fig_a3_event_study_idn_afc.png`
- Regression table and pre/post summary printed to stdout
