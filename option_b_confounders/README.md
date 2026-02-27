# Option B: Targeted Confounders Replacing Year FE

Identifies which temporal controls absorb the KOFGIdf effect on winning coalition size (W). The sub-index analysis showed that year FE eliminate all globalization effects — this analysis steps through progressively demanding temporal specifications to pinpoint the tipping point.

## Model Specifications

All models include country FE, GDP per capita, log(population), and country-clustered SEs.

| Model | Temporal Controls |
|-------|-------------------|
| 1 | None (baseline) |
| 2 | Cold War dummy (post-1991) |
| 3 | Post-AFC dummy (post-1997) |
| 4 | Both Cold War + post-AFC |
| 5 | Linear time trend (centered at 1993) |
| 6 | Quadratic time trend |
| 7 | Full year FE (47 dummies) |

## Key Question

If KOFGIdf survives Models 2-6 but dies in Model 7, the effect is robust to specific confounders and only disappears when all temporal variation is removed — suggesting year FE may overfit. If it dies at Model 5 (linear trend), a simple upward drift is the problem.

## Running

From the repository root:

```bash
python3 option_b_confounders/confounder_analysis.py
```

## Outputs

- `fig_b1_coefficient_progression.png` — KOFGIdf coefficient with 95% CI across all 7 specifications
- Regression table, VIF table, and findings summary printed to stdout
