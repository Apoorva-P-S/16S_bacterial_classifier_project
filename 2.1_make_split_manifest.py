#!/usr/bin/env python3
"""
2.1_make_split_manifest.py
==========================
Computes the group-aware (accession-level) train/validation/test split
ONCE, from taxonomy_table.csv alone -- no k-mer matrix needed, since the
split only depends on `accession` (the leakage-prevention grouping key)
and `genus` (for stratification).

WHY THIS EXISTS: without a saved manifest, the R correlation-reduction
script and the Python classifier script would each need to independently
recompute the split. Two independent calls to GroupShuffleSplit with the
"same" seed are only guaranteed identical if the input row order is also
identical -- which is fragile once a merge/join reorders rows anywhere in
the pipeline. A single saved manifest removes that fragility, AND is the
prerequisite for fitting the correlation-based feature reduction on the
TRAIN split only (see the modified R script).

Usage:
    python 2.1_make_split_manifest.py

Output: split_manifest.csv with columns sequence_id, accession, genus, split
(split in {"train", "val", "test"}). Both the R reduction script and the
Python classifier should read this file rather than deriving their own split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# CONFIG -- must match the values used in the classifier script
# =============================================================================

TAXONOMY_TABLE_CSV = Path("./taxonomy_table.csv")
OUTPUT_MANIFEST_CSV = Path("./split_manifest.csv")

TEST_SIZE = 0.2   # fraction of the FULL dataset
VAL_SIZE = 0.2    # fraction of the FULL dataset
RANDOM_STATE = 42


def main():
    df = pd.read_csv(TAXONOMY_TABLE_CSV)

    if "accession" not in df.columns:
        raise ValueError(f"'accession' column not found in {TAXONOMY_TABLE_CSV}.")
    if "genus" not in df.columns:
        raise ValueError(f"'genus' column not found in {TAXONOMY_TABLE_CSV}.")

    y = df["genus"].values
    groups = df["accession"].values
    idx_all = np.arange(len(df))

    remaining_fraction = 1.0 - TEST_SIZE
    if VAL_SIZE <= 0 or VAL_SIZE >= remaining_fraction:
        raise ValueError(
            f"VAL_SIZE ({VAL_SIZE}) must be > 0 and less than (1 - TEST_SIZE) = {remaining_fraction:.3f}."
        )
    val_size_within_trainval = VAL_SIZE / remaining_fraction

    # Stage 1: test vs. (train+val)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss_test.split(idx_all, y, groups))

    # Stage 2: train vs. val, drawn from the train+val pool only
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_within_trainval, random_state=RANDOM_STATE)
    train_idx_rel, val_idx_rel = next(
        gss_val.split(trainval_idx, y[trainval_idx], groups[trainval_idx])
    )
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    split_labels = np.array(["train"] * len(df), dtype=object)
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    # Hard leakage check: no accession may appear in more than one split
    acc_train, acc_val, acc_test = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    overlap = (acc_train & acc_val) | (acc_train & acc_test) | (acc_val & acc_test)
    if overlap:
        preview = sorted(overlap)[:5]
        raise RuntimeError(f"Leakage check failed: accession(s) in multiple splits: {preview}")

    manifest = df[["sequence_id", "accession", "genus"]].copy()
    manifest["split"] = split_labels
    manifest.to_csv(OUTPUT_MANIFEST_CSV, index=False)

    print("Split sizes (sequences):")
    print(manifest["split"].value_counts())
    print("\nUnique accessions per split:")
    print(f"  train: {len(acc_train)} | val: {len(acc_val)} | test: {len(acc_test)}")
    print(f"\nSaved: {OUTPUT_MANIFEST_CSV}")
    print("Point both the R reduction script (TRAIN_MANIFEST_CSV) and the "
          "classifier script at this file so all stages agree on the same split.")


if __name__ == "__main__":
    main()
