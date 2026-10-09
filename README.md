# Conditional Return Law Project

This repository builds leakage-auditable conditional return distributions from
endogenous and cross-market predictors.

The code couples repeatable feature construction with econometric, tabular, and
deep probabilistic models. Signals are registered with metadata, grouped into
families, and computed on a chronological asset-date panel so every forecast
can be tied back to its information set, lookback windows, and raw columns.

## Data Contract

The core signal builders expect a panel with one row per asset and date:

| column | meaning |
| --- | --- |
| `date` | observation date or timestamp |
| `asset` | asset identifier |
| `close` | adjusted close or tradable end-of-period price |
| `volume` | traded shares or contracts, required for liquidity signals |
| optional fundamentals | `market_cap`, `book_equity`, `net_income_ttm`, `revenue_ttm` |
| optional joined signals | macro, sentiment, or alternative-data columns |

The convention is that features at time `t` may use information available at
time `t` or earlier. Targets such as `ret_fwd_1d` are shifted forward and must
never be used as inputs.

## Quick Start

```python
from finance_project.models import RollingGaussianReturnModel, RollingStudentTReturnModel
from finance_project.signals import SignalLibrary, create_future_return_target

library = SignalLibrary.default()

features = library.compute(
    panel,
    families=["momentum", "volatility", "drawdown", "liquidity"],
    on_missing="skip",
)
target = create_future_return_target(panel, horizon=1)

model = RollingGaussianReturnModel(window=63, min_periods=21, horizon=1)
distribution = model.predict(panel)

student_t_model = RollingStudentTReturnModel(window=63, min_periods=21, horizon=1)
fat_tailed_distribution = student_t_model.predict(panel)
```

Both models implement the `ReturnDistributionModel` interface, so research
pipelines can call the same prediction, probability, and quantile methods while
comparing Gaussian and fat-tailed forecasts.

## NGBoost Model

Install the optional machine-learning dependencies and fit NGBoost on a
chronological training sample:

```bash
python3 -m pip install -e ".[ml,dev]"
```

```python
from finance_project.models import NGBoostReturnModel

model = NGBoostReturnModel(
    feature_cols=("mom_21d", "vol_21d", "drawdown_252d"),
    n_estimators=500,
    random_state=7,
)
model.fit(training_panel, target_col="ret_fwd_1d")
conditional_distribution = model.predict(test_panel)

student_t_model = NGBoostReturnModel(
    feature_cols=("mom_21d", "vol_21d", "drawdown_252d"),
    distribution="student_t",
    n_estimators=500,
    random_state=7,
)
student_t_model.fit(training_panel, target_col="ret_fwd_1d")
fat_tailed_conditional_distribution = student_t_model.predict(test_panel)
```

Split training and test panels chronologically. Forward-return targets belong
only in `target_col`; the model rejects conventional forward-target names in
`feature_cols`. The default conditional Normal model outputs mean, standard
deviation, variance, and training-row count. The conditional Student-t option
uses NGBoost's stable fixed `df=3` law and outputs degrees of freedom, conditional
location and scale, variance, and training-row count for each scorable
asset-date row.

## Advanced Model Ladder

The research layer now includes:

- GJR-GARCH with Student-t innovations;
- walk-forward signal-conditioned Student-t location/scale regression;
- nonlinear histogram-boosted quantile forecasts;
- a TimeXer-inspired exogenous patch/cross-attention encoder with a
  mixture-of-Student-t density head.

The Student-t regression is an interpretable benchmark, not the intended paper
contribution. See [docs/research_program.md](docs/research_program.md) for the
full top-conference model matrix, leakage controls, ablations, and evaluation
protocol.

The proposed research direction is now
[Q2P-Shift](docs/q2p_shift_spec.md): an identity-nested, state-dependent
transport from an option-implied risk-neutral return distribution to a
physical return distribution. Its mathematical core and timestamp contracts
are implemented, but no real option-surface result is claimed yet.

Run the cross-market model ladder:

```bash
python3 -m pip install -e ".[data,dev,ml]"
python3 examples/run_spy_signal_models.py
```

On Apple Silicon, use a native ARM Python environment for the deep model:

```bash
python3 examples/run_spy_deep_model.py
```

Build and evaluate the pooled 12-ETF experiment:

```bash
python3 examples/build_etf_universe.py --start 2015-01-01 --end 2025-12-31
python3 examples/run_etf_deep_model.py --epochs 15 --patience 4 \
  --d-model 32 --seed 7 --feature-set full
python3 examples/summarize_etf_deep_experiments.py
```

The current five-seed and information-set results are in
[`reports/etf_universe_deep_summary.md`](reports/etf_universe_deep_summary.md).
The full model improves sample NLL in all five seeds, but HAC and block-
bootstrap inference disagree at the 5% threshold; point-error gains are not
statistically distinguishable from random walk. This is promising
distributional evidence, not a finished paper claim.

Run the matched pooled NGBoost comparator on the same ETF universe and frozen
holdout:

```bash
python3 examples/run_etf_ngboost.py --feature-set full --seed 7
```

This evaluates conditional Normal and fixed-`df=3` Student-t NGBoost laws,
rolling baselines, per-asset diagnostics, and paired forecast-origin inference.
See
[`reports/etf_universe_ngboost_full_seed7.md`](reports/etf_universe_ngboost_full_seed7.md)
for the current result.

See [docs/signal_library.md](docs/signal_library.md) for the signal taxonomy and
[examples/build_signal_library.py](examples/build_signal_library.py) for a
synthetic end-to-end example.

## Real-data Baseline

```bash
python3 -m pip install -e ".[data,dev]"
python3 examples/run_spy_baseline.py
```

This caches SPY history locally, evaluates random-walk, Gaussian, and rolling
Student-t forecasts over the final 252 sessions, and writes
`reports/spy_baseline.md`. Use `--refresh` to replace the cached snapshot.