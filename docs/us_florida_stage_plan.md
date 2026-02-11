# Stage 2 Plan: U.S. to Florida (Plain Language)

## Goal
Move from one national recession model to a practical economic monitoring system:
- National risk monitor (already built)
- State-level pilot (Florida first)
- Action-oriented dashboard for everyday people and local leaders

## Why start with Florida
- Strong population and housing dynamics (good stress test).
- Mix of tourism, construction, services, and migration effects.
- Public data availability is decent.

## Step-by-step roadmap
1. Build a Florida "Stress Index" (monthly)
- Inputs (examples): unemployment rate, payroll growth, housing permits, consumer credit stress, small-business optimism, initial claims.
- Output: one easy-to-read score from 0 to 100.

### Stage 1 implementation status (current repo)
- Implemented with public Florida FRED series:
  - `FLUR` (unemployment rate)
  - `FLNA` (total nonfarm payrolls)
  - `FLMFG` (manufacturing payrolls)
  - `FLBPPRIVSA` (private building permits)
  - `FLICLAIMS` (initial claims)
- Artifacts produced:
  - `artifacts/florida_stage1_index.csv`
  - `artifacts/florida_stage1_latest.json`

2. Build a Florida recession-like risk model
- Label options:
  - Rising unemployment episodes.
  - Real income contraction periods.
  - NBER national recession as weak label + state stress thresholds.
- Output: probability that Florida enters stress/recession-like conditions in next 6 months.

3. Add county breakdown (optional next)
- Start with Miami-Dade, Broward, Orange, Hillsborough.
- Show heatmap and monthly change.

4. Add policy-and-people view
- "What households feel": rent burden, job openings trend, wage growth vs inflation.
- "What firms feel": credit spread proxies, hiring slowdown, bankruptcies (if available).

## Models to use (beginner-to-advanced)
1. Logistic Regression
- Best for interpretability and teaching.
- Good baseline and easy to explain.

2. Gradient Boosted Trees
- Captures nonlinear relationships.
- Usually stronger raw performance.

3. Ensemble (average of models)
- Often more stable than single model.
- Good production default if calibrated.

4. Next algorithms (when ready)
- Quantile Regression Forest / NGBoost: probabilistic uncertainty.
- Bayesian Dynamic Models: better for regime shifts and uncertainty tracking.
- Temporal Fusion Transformer (advanced): for richer multivariate time-series, but more complex.

## Recommended practical stack for you now
- Keep: Logistic + Gradient Boosted + Ensemble.
- Add next: a simple Bayesian model for uncertainty bands.
- Add later: regime-switching model (Markov switching) for turning points.

## How this could help people
- Public-facing monthly risk bulletin with plain-language summary.
- Early warning for local workforce programs.
- Better timing for household financial education campaigns.
- Local government budget stress planning before downturns.

## A simple monthly checklist
1. Is probability rising 3 months in a row?
2. Are labor indicators and financial conditions deteriorating together?
3. Is model confidence narrowing or widening?
4. Is Florida diverging from the national trend?

## Important caution
- Models support decisions; they do not replace decisions.
- Use multiple evidence sources and communicate uncertainty clearly.
