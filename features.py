"""
Feature engineering shared by train.py and app.py.

Two kinds of features:

1. Row-local parsing (no fitting): turn the raw strings ("5' 8\"", "137lbs",
   "34d") into numbers, clean impossible ages, add missing-value flags.
2. Features that need information from other rows. Each of these follows ONE rule:
   a rental may only use information that existed before that rental's date.
   - Item / user history rates: computed from rentals strictly earlier in time
     (expanding window), so the same code is valid for train, val, test and a
     live prediction.
   - Expected size: a small linear model (size ~ body measurements) fitted on
     the TRAIN split only, then applied everywhere.
"""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = "data/rtr_fit_sample.csv"

# Post-rental columns: known only after the customer has worn the item.
LEAKY_COLUMNS = ["rating"]

CUP_ORDER = {"aa": 0, "a": 1, "b": 2, "c": 3, "d": 4, "d+": 4.5, "dd": 5,
             "ddd/e": 6, "f": 7, "g": 8, "h": 9, "i": 10, "j": 11}

BODY_COLS = ["height_in", "weight_lb", "bust_band", "cup"]
CATEGORICAL = ["body_type", "rented_for", "category"]

# Smoothing strength for history rates: an item with n past rentals gets
# (misfits + M * prior) / (n + M), so few observations stay near the prior.
SMOOTHING_M = 20


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["review_date"])
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df.sort_values("review_date", kind="stable").reset_index(drop=True)


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """misfit = 1 if the customer reported 'small' or 'large'."""
    df = df.copy()
    df["is_small"] = (df["fit"] == "small").astype(int)
    df["is_large"] = (df["fit"] == "large").astype(int)
    df["misfit"] = df["is_small"] + df["is_large"]
    return df


# ---------------------------------------------------------------- parsing
def _height_to_inches(s):
    if not isinstance(s, str):
        return np.nan
    m = re.match(r"(\d+)'\s*(\d+)", s)
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def _weight_to_lb(s):
    if not isinstance(s, str):
        return np.nan
    m = re.match(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan


def _bust(s):
    if not isinstance(s, str):
        return np.nan, np.nan
    m = re.match(r"(\d+)(.*)", s.strip().lower())
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), CUP_ORDER.get(m.group(2), np.nan)


def parse_row_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns]).copy()
    df["height_in"] = df["height"].map(_height_to_inches)
    df["weight_lb"] = df["weight"].map(_weight_to_lb)
    bust = df["bust_size"].map(_bust)
    df["bust_band"] = [b for b, _ in bust]
    df["cup"] = [c for _, c in bust]

    # Ages like 0 or 116 are data-entry errors, not customers.
    df["age"] = df["age"].where(df["age"].between(15, 80))

    # Missingness is informative (customers who skip their weight may differ).
    for col in ["weight_lb", "bust_band", "age"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df["body_type"] = df["body_type"].fillna("unknown")

    df["bmi"] = 703 * df["weight_lb"] / df["height_in"] ** 2
    return df


# ---------------------------------------------------------------- history
def add_history_rates(df: pd.DataFrame, key: str, prefix: str, prior: dict) -> pd.DataFrame:
    """For each row, smoothed small/large/misfit rates of `key` over STRICTLY
    earlier dates. Same-day rentals don't see each other."""
    daily = df.groupby([key, "review_date"])[["is_small", "is_large"]].agg(["sum", "count"])
    daily.columns = ["small", "n", "large", "_n2"]
    daily = daily.drop(columns="_n2")
    before = daily.groupby(level=0).cumsum() - daily  # cumulative up to, excluding, this date
    before.columns = [f"{prefix}_past_{c}" for c in before.columns]
    out = df.merge(before, left_on=[key, "review_date"], right_index=True, how="left")

    n = out[f"{prefix}_past_n"]
    for kind in ["small", "large"]:
        out[f"{prefix}_{kind}_rate"] = (out[f"{prefix}_past_{kind}"] + SMOOTHING_M * prior[kind]) / (n + SMOOTHING_M)
    out[f"{prefix}_misfit_rate"] = out[f"{prefix}_small_rate"] + out[f"{prefix}_large_rate"]
    out[f"{prefix}_log_past_n"] = np.log1p(n)
    return out.drop(columns=[f"{prefix}_past_small", f"{prefix}_past_large", f"{prefix}_past_n"])


def history_snapshot(df: pd.DataFrame, key: str, prior: dict) -> pd.DataFrame:
    """Rates using ALL rows, i.e. what a live prediction made today would see."""
    g = df.groupby(key).agg(n=("misfit", "size"), small=("is_small", "sum"), large=("is_large", "sum"))
    for kind in ["small", "large"]:
        g[f"{kind}_rate"] = (g[kind] + SMOOTHING_M * prior[kind]) / (g["n"] + SMOOTHING_M)
    return g


# ---------------------------------------------------------------- expected size
class ExpectedSize:
    """Predicts the size a customer *should* rent from height/weight/bust,
    learned only from TRAIN rentals that fit. The gap between the size actually
    rented and this expectation is the core signal: renting far above or below
    your body's typical size is how misfits happen."""

    def fit(self, df: pd.DataFrame):
        rows = df[(df["misfit"] == 0)].dropna(subset=BODY_COLS)
        self.model = LinearRegression().fit(rows[BODY_COLS], rows["size"])
        self.fill = rows[BODY_COLS].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        complete = df[BODY_COLS].notna().all(axis=1)
        expected = self.model.predict(df[BODY_COLS].fillna(self.fill))
        dev = df["size"] - expected
        # Unknown body measurements -> we don't know the deviation; use 0 and flag it.
        df["size_dev"] = np.where(complete, dev, 0.0)
        df["size_dev_unknown"] = (~complete).astype(int)
        # Logistic regression is linear in its inputs, so it cannot learn
        # "too big OR too small is bad" from size_dev alone (a V shape).
        # Two hinges let each direction get its own slope.
        df["size_dev_up"] = df["size_dev"].clip(lower=0)
        df["size_dev_down"] = (-df["size_dev"]).clip(lower=0)
        # Does this item run small, and did the customer size down? (and vice versa)
        df["risk_small"] = df["item_small_rate"] * df["size_dev_down"]
        df["risk_large"] = df["item_large_rate"] * df["size_dev_up"]
        return df


# ---------------------------------------------------------------- feature sets
# Grouped so train.py can run an ablation and the report can say what each
# group contributed.
FEATURE_GROUPS = {
    "raw_numeric": ["size", "age", "age_missing"],
    "parsed_body": ["height_in", "weight_lb", "bust_band", "cup", "bmi",
                    "weight_lb_missing", "bust_band_missing"],
    "size_deviation": ["size_dev_up", "size_dev_down", "size_dev_unknown"],
    "item_history": ["item_small_rate", "item_large_rate", "item_log_past_n"],
    "user_history": ["user_misfit_rate", "user_log_past_n"],
    "interactions": ["risk_small", "risk_large"],
}
NUMERIC_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
