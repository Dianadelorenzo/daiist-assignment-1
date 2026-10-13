"""
One-off script that built data/rtr_fit_sample.csv from the raw Rent the Runway
dump. It is NOT part of `main.py train`: the committed CSV is the pipeline's input.
Kept here so the sampling is reproducible and explainable.

Raw source (UCSD McAuley Lab, Misra et al. 2018, "Decomposing fit semantics for
product size recommendation in metric spaces"):
    https://mcauleylab.ucsd.edu/public_datasets/data/renttherunway/renttherunway_final_data.json.gz

Usage:
    uv run python data/make_sample.py <path/to/renttherunway_final_data.json.gz>

Sampling choices:
- Keep 2015-01-01 .. 2017-12-31 only (2010-2014 are sparse/early-business years,
  2018 has just one week of data).
- Keep the most-rented items in that window until we reach ~20k rows. Item-level
  "does this item run small/large?" features need several rentals per item; a
  uniform random sample of 20k rows from 5,850 items would leave ~3 rows per item.
- Drop review_text / review_summary: they are written *after* the rental (leakage)
  and are most of the file size. `rating` is kept only so the report can show
  why it is excluded.
"""

import sys
from pathlib import Path

import pandas as pd

TARGET_ROWS = 20_000

raw = pd.read_json(sys.argv[1], lines=True)
raw["review_date"] = pd.to_datetime(raw["review_date"], format="%B %d, %Y")
raw = raw[(raw["review_date"] >= "2015-01-01") & (raw["review_date"] <= "2017-12-31")]

counts = raw["item_id"].value_counts()
n_items = int((counts.cumsum() <= TARGET_ROWS).sum())
keep = counts.index[:n_items]
sample = raw[raw["item_id"].isin(keep)].drop(columns=["review_text", "review_summary"])
sample = sample.sort_values("review_date").reset_index(drop=True)

out = Path(__file__).parent / "rtr_fit_sample.csv"
sample.to_csv(out, index=False)
print(f"{len(sample):,} rows, {n_items} items, "
      f"{sample['review_date'].min().date()} .. {sample['review_date'].max().date()} -> {out}")
