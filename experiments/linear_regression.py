"""Cross-sectional linear regression: rank 100 large US stocks by 21-day excess return."""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from finance_project.signals import compute_signals, create_future_return_target
from finance_project.signals.core import SIGNALS

TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "IBM", "INTC", "QCOM", "TXN", "ADBE",
    "CRM", "ACN", "AMD", "MU", "AMAT", "ADI",
    # Communication
    "GOOGL", "CMCSA", "VZ", "T", "DIS", "NFLX",
    # Consumer discretionary
    "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "TJX", "F", "EBAY",
    # Consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "MO", "CL", "MDLZ", "KMB", "GIS",
    # Health care
    "JNJ", "PFE", "MRK", "UNH", "ABT", "LLY", "BMY", "AMGN", "GILD", "MDT", "CVS",
    "TMO", "DHR", "CI",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "TRV", "MET", "AIG", "SCHW",
    "BLK", "PNC", "COF", "ALL",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "EOG",
    # Industrials
    "GE", "HON", "MMM", "CAT", "BA", "UPS", "UNP", "LMT", "DE", "EMR", "FDX",
    # Utilities
    "NEE", "DUK", "SO", "D", "EXC",
    # Materials
    "APD", "NEM", "ECL",
    # Real estate
    "SPG", "AMT",
]
HORIZON = 21
SPLITS = {
    "train": ("2010-01-01", "2018-12-31"),
    "validation": ("2019-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-09-30"),
}
ALPHAS = [0, 1e3, 1e4, 1e5, 1e6]  # ridge penalties; 0 is plain OLS
CACHE = Path("data/stocks100_2009-01-01_2026-09-30.csv")
REPORT = Path("experiments/linear_regression.md")


def load_prices():
    """Adjusted daily closes from Yahoo Finance, cached to CSV after the first download."""
    if not CACHE.exists():
        close = yf.download(TICKERS, start="2009-01-01", end="2026-10-01", auto_adjust=True, progress=False)["Close"]
        close.rename_axis(index="date", columns="asset").stack().rename("close").reset_index().to_csv(
            CACHE, index=False
        )
    return pd.read_csv(CACHE, parse_dates=["date"])


def build_dataset(prices):
    """Signals ranked across stocks per date (missing = median); target = return minus universe mean."""
    names = list(SIGNALS)
    target = f"ret_fwd_{HORIZON}d"
    data = compute_signals(prices, names).merge(
        create_future_return_target(prices, HORIZON), on=["date", "asset"]
    )
    data = data.dropna(subset=[target])
    data[names] = (data.groupby("date")[names].rank(pct=True) - 0.5).fillna(0)
    data["y"] = data[target] - data.groupby("date")[target].transform("mean")
    return data, names


def split(data):
    """Date splits; drop the last HORIZON dates of train and validation so no target overlaps the next split."""
    parts = {}
    for name, (start, end) in SPLITS.items():
        part = data[data["date"].between(start, end)]
        if name != "test":
            part = part[part["date"] < np.sort(part["date"].unique())[-HORIZON]]
        parts[name] = part
    return parts


def fit(part, names, alpha):
    X, y = part[names].to_numpy(), part["y"].to_numpy()
    return np.linalg.solve(X.T @ X + alpha * np.eye(len(names)), X.T @ y)


def evaluate(part, names, beta):
    """Out-of-sample R², rank IC and top-minus-bottom decile returns on monthly (non-overlapping) dates."""
    part = part.assign(pred=part[names].to_numpy() @ beta)
    r2 = 1 - ((part["y"] - part["pred"]) ** 2).sum() / (part["y"] ** 2).sum()
    monthly = part[part["date"].isin(np.sort(part["date"].unique())[::HORIZON])]
    ic = monthly.groupby("date").apply(lambda g: g["pred"].corr(g["y"], method="spearman"))
    decile = monthly.groupby("date")["pred"].transform(lambda p: pd.qcut(p.rank(method="first"), 10, labels=False))
    spread = monthly[decile == 9].groupby("date")["y"].mean() - monthly[decile == 0].groupby("date")["y"].mean()
    periods = 252 / HORIZON
    return {
        "R² (%)": 100 * r2,
        "IC mean": ic.mean(),
        "IC t-stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "IC > 0 (%)": 100 * (ic > 0).mean(),
        "L/S return (%/yr)": 100 * spread.mean() * periods,
        "L/S Sharpe": spread.mean() / spread.std() * np.sqrt(periods),
        "months": len(ic),
    }


def table(frame):
    return f"```\n{frame.round(3).to_string()}\n```"


def main():
    prices = load_prices()
    data, names = build_dataset(prices)
    parts = split(data)

    validation = pd.DataFrame(
        {alpha: evaluate(parts["validation"], names, fit(parts["train"], names, alpha)) for alpha in ALPHAS}
    ).T.rename_axis("alpha")
    alpha = validation["IC mean"].idxmax()

    final = fit(pd.concat([parts["train"], parts["validation"]]), names, alpha)
    results = pd.DataFrame(
        {name: evaluate(part, names, final) for name, part in parts.items()}
    ).T
    coefficients = pd.Series(final, index=names, name="coefficient (bp per rank unit)") * 1e4

    ranges = "\n".join(
        f"- {name}: {part['date'].min():%Y-%m-%d} to {part['date'].max():%Y-%m-%d}, "
        f"{len(part):,} rows" for name, part in parts.items()
    )
    REPORT.write_text(
        f"# Linear regression on {prices['asset'].nunique()} stocks\n\n"
        f"Target: {HORIZON}-day forward return minus the equal-weight universe mean.\n\n{ranges}\n\n"
        f"## Validation (fit on train)\n\n{table(validation)}\n\n"
        f"Chosen alpha: {alpha:g}. Final model refit on train + validation.\n\n"
        f"## Final model\n\n{table(results)}\n\n"
        f"Train and validation rows are in-sample for the final model; only test is out-of-sample.\n\n"
        f"## Coefficients\n\n{table(coefficients.sort_values().to_frame())}\n"
    )
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
