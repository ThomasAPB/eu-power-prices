"""Tests that the forecasting set-up does not use information from the future.

Run from the repository root:  pytest -q
Data needed: data/processed/ES_hourly_2024.parquet and data/raw/ (same as notebook 02).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import forecast as F  # noqa: E402

REALISED = ["price", "solar", "wind", "load", "residual_load", "gas_gen"]
FORECASTS = ["load_fc", "solar_fc", "wind_fc"]


@pytest.fixture(scope="module")
def hourly():
    return pd.read_parquet(ROOT / "data" / "processed" / "ES_hourly_2024.parquet")


@pytest.fixture(scope="module")
def fuels():
    return F.load_fuels(ROOT / "data" / "raw")


@pytest.fixture(scope="module")
def ds(hourly, fuels):
    return F.build_dataset(hourly, fuels)


def _day_rows(df, day):
    return df[F.local_date(df.index) == pd.Timestamp(day)]


def test_features_of_day_d_ignore_realised_data_from_d_onwards(hourly, fuels, ds):
    """Replace every realised value from 00:00 of day D with noise: the features of day D
    (and of all earlier days) must not move. Only the target may change."""
    D = pd.Timestamp("2024-06-15")
    rng = np.random.default_rng(0)
    h2 = hourly.copy()
    late = np.asarray(F.local_date(h2.index) >= D)
    for c in REALISED:
        h2.loc[late, c] = rng.normal(1e6, 1e5, late.sum())
    ds2 = F.build_dataset(h2, fuels)

    feats = F.FEATURES_WS
    upto_d = np.asarray(F.local_date(ds.index) <= D)
    pd.testing.assert_frame_equal(ds.loc[upto_d, feats], ds2.loc[upto_d, feats])
    assert not ds.loc[upto_d, "price"].equals(ds2.loc[upto_d, "price"])   # the test can fail


def test_features_of_day_d_ignore_forecasts_of_later_days(hourly, fuels, ds):
    D = pd.Timestamp("2024-06-15")
    h2 = hourly.copy()
    later = np.asarray(F.local_date(h2.index) > D)
    for c in FORECASTS:
        h2.loc[later, c] = 1e6
    ds2 = F.build_dataset(h2, fuels)
    upto_d = np.asarray(F.local_date(ds.index) <= D)
    pd.testing.assert_frame_equal(ds.loc[upto_d, F.FEATURES_WS], ds2.loc[upto_d, F.FEATURES_WS])


def test_fuel_closes_of_d_minus_1_are_not_used(hourly, fuels, ds):
    """At noon on D-1 the close of D-1 does not exist: changing it must not move day D;
    changing the close of D-2 must (otherwise the test proves nothing)."""
    D = pd.Timestamp("2024-06-18")                      # a Tuesday
    f2 = fuels.copy()
    f2.loc[D - pd.Timedelta(days=1):, :] = 999.0       # closes of D-1 and later
    ds2 = F.build_dataset(hourly, f2)
    pd.testing.assert_frame_equal(_day_rows(ds, D)[F.FEATURES_WS], _day_rows(ds2, D)[F.FEATURES_WS])

    f3 = fuels.copy()
    f3.loc[D - pd.Timedelta(days=2), :] = 999.0         # close of D-2
    ds3 = F.build_dataset(hourly, f3)
    assert (_day_rows(ds3, D)["ttf"] == 999.0).all()


def test_fuels_after_weekend_use_friday_close(fuels):
    """Monday D: last close dated D-2 or earlier is Friday's (Sat D-2 / Sun D-1 do not trade)."""
    monday = pd.Timestamp("2024-06-17")
    known = F.known_fuels(fuels, pd.DatetimeIndex([monday]))
    assert known.loc[monday, "ttf"] == fuels.loc[pd.Timestamp("2024-06-14"), "ttf"]


def test_lags_are_exact_shifts(ds, hourly):
    t = pd.Timestamp("2024-09-10 14:00", tz="Europe/Madrid")
    assert ds.loc[t, "lag_24"] == hourly.loc[t - pd.Timedelta(hours=24), "price"]
    assert ds.loc[t, "lag_168"] == hourly.loc[t - pd.Timedelta(hours=168), "price"]


def test_prev_day_stats_use_previous_local_day(ds, hourly):
    day = pd.Timestamp("2024-09-10")
    prev = hourly.loc[F.local_date(hourly.index) == day - pd.Timedelta(days=1), "price"]
    row = _day_rows(ds, day)
    assert np.allclose(row["prev_day_mean"], prev.mean())
    assert np.allclose(row["prev_day_max"], prev.max())


def test_walk_forward_splits_are_chronological(ds):
    folds = F.monthly_folds(ds)
    assert len(folds) == 9
    d = F.local_date(ds.index)
    covered = np.zeros(len(ds), bool)
    for f in folds:
        assert d[f.train].max() < f.test_start <= d[f.test].min()
        assert not (f.train & f.test).any()
        assert not (covered & f.test).any()             # test months do not overlap
        covered |= f.test
    assert d[covered].min() == folds[0].test_start


def test_naive_weekly_rule(ds):
    b = F.baselines(ds)
    tue_fri = ds["dow"].isin([1, 2, 3, 4])
    assert (b.loc[tue_fri, "naive_weekly"] == ds.loc[tue_fri, "lag_24"]).all()
    assert (b.loc[~tue_fri, "naive_weekly"] == ds.loc[~tue_fri, "lag_168"]).all()


@pytest.mark.parametrize("kind", ["ridge", "hgb"])
def test_fit_and_tuning_only_see_the_past(ds, kind, monkeypatch):
    """Spy on every fit and every tuning score: the rows used must all be dated before the
    test month (tuning rows before it, final fit rows before it). Catches tuning or early
    stopping on the test set, which a change of target values can miss when the same
    hyper-parameters happen to win."""
    seen = {"fit": [], "score": []}
    real_estimator, real_mae = F._estimator, F._mae

    class Spy:
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            seen["fit"].append(F.local_date(X.index).max())
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

    monkeypatch.setattr(F, "_estimator", lambda *a, **k: Spy(real_estimator(*a, **k)))
    monkeypatch.setattr(F, "_mae", lambda y, p: (seen["score"].append(F.local_date(y.index).max()),
                                                 real_mae(y, p))[1])
    small = {"ridge": [{"alpha": 10}, {"alpha": 100}],
             "hgb": [{"max_iter": 30, "max_leaf_nodes": 8}, {"max_iter": 60, "max_leaf_nodes": 8}]}[kind]
    folds = F.monthly_folds(ds)[-3:]
    F.walk_forward(ds, F.FEATURES_BASE, kind, folds, grid=small)

    per_fold = len(small) + 1                              # inner fits + final refit
    assert len(seen["fit"]) == per_fold * len(folds)
    assert len(seen["score"]) == len(small) * len(folds)
    for i, f in enumerate(folds):
        assert all(t < f.test_start for t in seen["fit"][i * per_fold:(i + 1) * per_fold])
        assert all(t < f.test_start for t in seen["score"][i * len(small):(i + 1) * len(small)])


@pytest.mark.parametrize("kind", ["ridge", "hgb"])
def test_forecasts_do_not_depend_on_test_month_targets(ds, kind):
    """The target of the last test month is never a training target of any fold. Corrupting it
    (not the lag features, built before) must leave every forecast unchanged."""
    folds = F.monthly_folds(ds)[-3:]
    small = {"ridge": [{"alpha": 10}, {"alpha": 100}],
             "hgb": [{"max_iter": 30, "max_leaf_nodes": 8}, {"max_iter": 60, "max_leaf_nodes": 8}]}[kind]
    p1, _ = F.walk_forward(ds, F.FEATURES_BASE, kind, folds, grid=small)
    ds2 = ds.copy()
    ds2.loc[folds[-1].test, "price"] = 1e6
    p2, _ = F.walk_forward(ds2, F.FEATURES_BASE, kind, folds, grid=small)
    pd.testing.assert_series_equal(p1, p2)
    assert p1[folds[-1].test].notna().all()


def test_dm_test_detects_a_difference_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 1, 300)
    stat, p = F.diebold_mariano(a, a + rng.normal(0, 1, 300))        # same mean loss
    assert p > 0.01
    stat, p = F.diebold_mariano(a, a + 3 + rng.normal(0, 1, 300))    # B clearly worse
    assert stat < 0 and p < 1e-6
