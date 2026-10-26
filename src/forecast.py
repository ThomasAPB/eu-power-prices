"""Day-ahead price forecasting without look-ahead.

Forecasting problem
-------------------
At 12:00 on day D-1 (gate closure of the day-ahead auction) we forecast the 24
hourly prices of day D. Every feature of an hourly row of day D must therefore
be known at 12:00 on D-1:

* prices of day D-1 and earlier (D-1 prices were published on D-2),
* the TSO day-ahead load forecast of day D (published >= 2 h before gate closure),
* gas (TTF) and carbon (EUA) closes of D-2 and earlier (the D-1 close does not
  exist yet at noon on D-1),
* calendar variables.

The TSO wind and solar forecasts may be published up to 18:00 on D-1, i.e.
after gate closure (Reg. EU 543/2013, art. 14). They are kept in a separate
feature set (FEATURES_WS) so that the cost of this assumption is visible.

Evaluation is a walk-forward: train on everything before month M, forecast
month M, then slide. Hyper-parameters are chosen on the last days of the
*training* window only; the test month is never used to fit or to select.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Same constants as notebook 02 (see README, Method)
ETA = 0.50   # assumed efficiency of a gas plant
EF = 0.202   # tCO2 per MWh of gas (IPCC 2006)

FEATURES_BASE = [
    "lag_24", "lag_48", "lag_168",
    "prev_day_mean", "prev_day_min", "prev_day_max", "prev_week_mean",
    "load_fc", "ttf", "eua", "mc",
    "hour", "dow", "holiday",
]
FEATURES_WS = FEATURES_BASE + ["solar_fc", "wind_fc", "res_fc"]
CATEGORICAL = ["hour", "dow"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_fuels(raw_dir: Path) -> pd.DataFrame:
    """Daily TTF and EUA closes (naive dates), as downloaded for notebook 02."""
    raw_dir = Path(raw_dir)
    ttf = pd.read_csv(raw_dir / "ttf_daily.csv", index_col=0, parse_dates=True)["ttf"]
    eua = pd.read_csv(raw_dir / "Carbon Emissions Futures Historical Data.csv")
    eua["Date"] = pd.to_datetime(eua["Date"], format="%m/%d/%Y")   # US format: month first
    eua = eua.set_index("Date").sort_index()["Price"].rename("eua")
    return pd.concat([ttf, eua], axis=1).sort_index()


def local_date(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive local calendar date (midnight) of each timestamp."""
    return index.tz_localize(None).normalize()


def _check_hourly(index: pd.DatetimeIndex) -> None:
    if index.tz is None:
        raise ValueError("hourly index must be timezone-aware")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("hourly index must be sorted and without duplicates")
    steps = pd.Series(index).diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(hours=1):
        raise ValueError(f"hourly index must have no gap: steps found {steps}")


def known_fuels(fuels: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Gas and carbon closes known at 12:00 on D-1, for each day D in `days`.

    The value on D is the last close dated D-2 or earlier: forward-fill over a
    full calendar (weekends, holidays), then shift by 2 days.
    """
    cal = pd.date_range(fuels.index.min(), days.max())
    f = fuels.reindex(cal).ffill(limit=5).shift(2)
    return f.reindex(days)


def build_dataset(hourly: pd.DataFrame, fuels: pd.DataFrame) -> pd.DataFrame:
    """One row per hour of delivery, features known at 12:00 on D-1, target `price`."""
    h = hourly.sort_index()
    _check_hourly(h.index)
    price = h["price"]
    day = local_date(h.index)

    ds = pd.DataFrame(index=h.index)
    ds["price"] = price

    # Same hour 1, 2, 7 days earlier (absolute 24 h steps; one clock hour off on DST days)
    for k in (24, 48, 168):
        ds[f"lag_{k}"] = price.shift(k)

    # Daily statistics of the previous local day / week, attached to day D
    daily = price.groupby(day).agg(["mean", "min", "max"])
    daily.columns = ["prev_day_mean", "prev_day_min", "prev_day_max"]
    daily["prev_week_mean"] = daily["prev_day_mean"].rolling(7, min_periods=7).mean()
    daily.index = daily.index + pd.Timedelta(days=1)          # day D sees the stats of D-1
    for c in daily.columns:
        ds[c] = daily[c].reindex(day).to_numpy()

    # Day-ahead forecasts of day D (see module docstring for the timing caveat)
    ds["load_fc"] = h["load_fc"]
    ds["solar_fc"] = h["solar_fc"]
    ds["wind_fc"] = h["wind_fc"]
    ds["res_fc"] = h["load_fc"] - h["wind_fc"] - h["solar_fc"]

    # Gas and carbon: closes of D-2 and earlier
    kf = known_fuels(fuels, pd.DatetimeIndex(day.unique()))
    ds["ttf"] = kf["ttf"].reindex(day).to_numpy()
    ds["eua"] = kf["eua"].reindex(day).to_numpy()
    ds["mc"] = (ds["ttf"] + EF * ds["eua"]) / ETA

    # Calendar
    ds["hour"] = h.index.hour
    ds["dow"] = h.index.dayofweek
    hol = holidays.country_holidays("ES", years=sorted(set(day.year)))   # national holidays
    ds["holiday"] = np.isin(day.date, list(hol.keys())).astype(int)

    # First 7 days have no full history: drop them
    return ds.dropna(subset=["lag_168", "prev_week_mean"])


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def baselines(ds: pd.DataFrame) -> pd.DataFrame:
    """Naive benchmarks. `naive_weekly` is the usual one in the price-forecasting
    literature: yesterday for Tue-Fri, same day last week for Sat, Sun, Mon."""
    weekday_rule = ds["dow"].isin([1, 2, 3, 4])
    return pd.DataFrame(
        {
            "naive_d1": ds["lag_24"],
            "naive_d7": ds["lag_168"],
            "naive_weekly": np.where(weekday_rule, ds["lag_24"], ds["lag_168"]),
        },
        index=ds.index,
    )


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------
@dataclass
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp        # exclusive
    train: np.ndarray             # boolean mask over ds rows
    test: np.ndarray


def monthly_folds(ds: pd.DataFrame, first_test: str = "2024-04-01") -> list[Fold]:
    """Expanding window: train on all days before the month, test on the month."""
    d = local_date(ds.index)
    folds = []
    for s in pd.date_range(first_test, d.max(), freq="MS"):
        e = s + pd.offsets.MonthBegin(1)
        folds.append(Fold(s, e, np.asarray(d < s), np.asarray((d >= s) & (d < e))))
    return folds


def _estimator(kind: str, params: dict, features: list[str]):
    cats = [c for c in CATEGORICAL if c in features]
    nums = [c for c in features if c not in cats]
    if kind == "ridge":
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ])
        return make_pipeline(pre, Ridge(alpha=params["alpha"]))
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=cats,
            early_stopping=False,     # no automatic stopping: it would need a validation set
            random_state=0,
        )
    raise ValueError(kind)


GRIDS = {
    "ridge": [{"alpha": a} for a in (1, 10, 100, 1000)],
    "hgb": [{"max_iter": i, "max_leaf_nodes": n} for i in (150, 400) for n in (8, 31)],
}


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def walk_forward(ds: pd.DataFrame, features: list[str], kind: str,
                 folds: list[Fold], grid: list[dict] | None = None,
                 inner_days: int = 28) -> tuple[pd.Series, pd.DataFrame]:
    """Forecast each test month with a model fitted on the past only.

    Inside each training window, the last `inner_days` days are held out to pick
    the hyper-parameters; the model is then refitted on the whole window.
    Returns the forecasts (NaN outside test months) and the chosen parameters.
    """
    grid = grid or GRIDS[kind]
    d = local_date(ds.index)
    X, y = ds[features], ds["price"]
    pred = pd.Series(np.nan, index=ds.index, name=f"{kind}")
    chosen = []
    for f in folds:
        cut = f.test_start - pd.Timedelta(days=inner_days)
        inner_fit = f.train & np.asarray(d < cut)
        inner_val = f.train & np.asarray(d >= cut)
        scores = []
        for p in grid:
            m = _estimator(kind, p, features).fit(X[inner_fit], y[inner_fit])
            scores.append(_mae(y[inner_val], m.predict(X[inner_val])))
        best = grid[int(np.argmin(scores))]
        m = _estimator(kind, best, features).fit(X[f.train], y[f.train])
        pred[f.test] = m.predict(X[f.test])
        chosen.append({"test_month": f.test_start.strftime("%Y-%m"), **best,
                       "inner_val_mae": min(scores)})
    return pred, pd.DataFrame(chosen)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def scores(y: pd.Series, preds: pd.DataFrame, ref: str) -> pd.DataFrame:
    """MAE, RMSE (EUR/MWh) and MAE relative to the reference forecast."""
    rows = {}
    for c in preds.columns:
        e = y - preds[c]
        rows[c] = {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean())}
    out = pd.DataFrame(rows).T
    out["rMAE"] = out["MAE"] / out.loc[ref, "MAE"]
    return out


def diebold_mariano(loss_a, loss_b, lag: int = 6) -> tuple[float, float]:
    """Test of equal predictive accuracy on a loss differential series.

    H0: E[loss_a - loss_b] = 0. The variance of the mean uses a Bartlett (Newey-West)
    window of `lag` terms because daily differentials are autocorrelated.
    Returns (statistic, two-sided p-value, normal approximation).
    A negative statistic means that forecast A has the smaller loss.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k] / n)
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def daily_abs_error(y: pd.Series, p: pd.Series) -> pd.Series:
    """Mean absolute error per local day (24 hourly errors are not independent)."""
    return (y - p).abs().groupby(local_date(y.index)).mean()
