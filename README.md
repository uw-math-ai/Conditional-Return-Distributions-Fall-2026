# Conditional Return Distributions

- `src/finance_project/signals/core.py`: signal functions and the forward-return target.
- `experiments/linear_regression.py`: cross-sectional linear regression on 100 large US stocks (21-day excess return). Results are in `experiments/linear_regression.md`.

## Setup

```
pip install -e .
python experiments/linear_regression.py
```

Price data is downloaded with yfinance on the first run and cached as a CSV (not tracked).
