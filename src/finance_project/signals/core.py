"""Price-based signals for data with date, asset, and close columns."""

import numpy as np


def returns(frame, periods=1):
    return frame.groupby("asset",sort=False)["close"].pct_change(periods)


def rolling(values, frame, window, statistic):
    return values.groupby(frame["asset"],sort=False).transform(
        lambda group: group.rolling(window, min_periods=max(5, window // 2)).agg(
            statistic
        )
    )


def volatility(frame, window):
    return rolling(returns(frame), frame, window, "std") * np.sqrt(252)


def downside_volatility(frame, window):
    squared = returns(frame).clip(upper=0).pow(2)
    return rolling(squared, frame, window, "mean").pow(0.5) * np.sqrt(252)


def downside_frequency(frame, window):
    daily = returns(frame)
    return rolling(daily.lt(0).astype(float).where(daily.notna()), frame, window, "mean")


def drawdown(frame, window):
    high = rolling(frame["close"], frame, window, "max")
    return frame["close"] / high - 1


SIGNALS = {
    "reversal_1d": lambda f: -returns(f),
    "mom_5d": lambda f: returns(f, 5),
    "mom_21d": lambda f: returns(f, 21),
    "mom_63d": lambda f: returns(f, 63),
    "mom_126d": lambda f: returns(f, 126),
    "mom_252d": lambda f: returns(f, 252),
    "vol_21d": lambda f: volatility(f, 21),
    "vol_63d": lambda f: volatility(f, 63),
    "vol_126d": lambda f: volatility(f, 126),
    "downside_vol_21d": lambda f: downside_volatility(f, 21),
    "downside_vol_63d": lambda f: downside_volatility(f, 63),
    "drawdown_21d": lambda f: drawdown(f, 21),
    "drawdown_63d": lambda f: drawdown(f, 63),
    "drawdown_252d": lambda f: drawdown(f, 252),
    "skew_63d": lambda f: rolling(returns(f), f, 63, "skew"),
    "skew_126d": lambda f: rolling(returns(f), f, 126, "skew"),
    "downside_freq_21d": lambda f: downside_frequency(f, 21),
    "downside_freq_63d": lambda f: downside_frequency(f, 63),
}


def prepare(data):
    """Drop duplicate asset-date rows (keeping the first) and sort by asset, date."""
    return data.drop_duplicates(["asset", "date"]).sort_values(["asset", "date"])


def compute_signals(data, names):
    """Warm-up rows stay NaN; infinities (e.g. from zero prices) become NaN."""
    frame = prepare(data)
    features = frame[["date", "asset"]].copy()
    for name in names:
        features[name] = SIGNALS[name](frame).replace([np.inf, -np.inf], np.nan)
    return features


def create_future_return_target(data, horizon=1, column="close"):
    """Forward simple return from t to t + horizon, aligned to features at t."""
    frame = prepare(data)
    target = frame[["date", "asset"]].copy()
    future = frame.groupby("asset")[column].shift(-horizon)
    target[f"ret_fwd_{horizon}d"] = future / frame[column] - 1
    return target

