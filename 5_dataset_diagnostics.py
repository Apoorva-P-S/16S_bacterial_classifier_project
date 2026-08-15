# Performs quality control and exploratory diagnostics on a
# 16S rRNA k-mer dataset before machine learning.
# ------
# ✓ Logging
# ✓ Dataset loading
# ✓ Dataset summary
# ✓ Missing value analysis
# ✓ Class imbalance analysis
# ✓ Duplicate sequence IDs
# ✓ Duplicate feature vectors

#Author: Apoorva


from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

# ============================================================
# CONFIGURATION
# ============================================================

KMER_MATRIX_CSV = Path("./kmer_matrix_reduced.csv")
TAXONOMY_TABLE_CSV = Path("./taxonomy_table.csv")

OUTPUT_DIR = Path("./diagnostics")

FIGURE_DPI = 300

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

def create_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diagnostics directory: {OUTPUT_DIR.resolve()}")

#Dataset loading
def load_dataset():

    logger.info("Loading taxonomy table...")

    taxonomy = pd.read_csv(TAXONOMY_TABLE_CSV)

    logger.info("Loading k-mer matrix...")

    kmer = pd.read_csv(KMER_MATRIX_CSV)

    logger.info("Merging datasets...")

    merged = kmer.merge(
        taxonomy[["sequence_id", "accession", "sequence_length"]],
        on="sequence_id",
        how="left"
    )

    logger.info(f"Loaded {len(merged)} sequences.")

    return merged

#Dataset summary
def dataset_summary(df):

    logger.info("Computing dataset summary...")

    summary = {

        "Total sequences":
            len(df),

        "Total genera":
            df["genus"].nunique(),

        "Total features":
            len([
                c
                for c in df.columns
                if c not in
                [
                    "sequence_id",
                    "genus",
                    "accession",
                    "header",
                    "sequence_length"
                ]
            ]),

        "Average sequence length":
            df["sequence_length"].mean(),

        "Minimum sequence length":
            df["sequence_length"].min(),

        "Maximum sequence length":
            df["sequence_length"].max()
    }

    summary_df = pd.DataFrame(
        summary.items(),
        columns=["Metric", "Value"]
    )

    summary_df.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False
    )

    logger.info("\n%s", summary_df)

    return summary

#Missing value analysis
def check_missing_values(df):

    logger.info("Checking missing values...")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        logger.info("No missing values detected.")

        return

    missing_df = missing.reset_index()

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    missing_df.to_csv(
        OUTPUT_DIR / "missing_values.csv",
        index=False
    )

    logger.warning(
        f"{len(missing_df)} columns contain missing values."
    )

    logger.info("\n%s", missing_df)

#Class imbalance analysis
def class_distribution(df):

    logger.info("Analysing class distribution...")

    counts = (
        df["genus"]
        .value_counts()
        .sort_values(ascending=False)
    )

    counts.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        header=["Count"]
    )

    ratio = counts.max() / counts.min()

    logger.info(f"Number of genera: {len(counts)}")
    logger.info(f"Largest class: {counts.max()}")
    logger.info(f"Smallest class: {counts.min()}")
    logger.info(f"Imbalance ratio: {ratio:.2f}")

    if ratio > 10:

        logger.warning(
            "Highly imbalanced dataset detected."
        )

    elif ratio > 5:

        logger.warning(
            "Moderately imbalanced dataset detected."
        )

    else:

        logger.info(
            "Dataset appears reasonably balanced."
        )

    plt.figure(figsize=(14,5))

    counts.plot(kind="bar")

    plt.ylabel("Sequences")

    plt.xlabel("Genus")

    plt.title("Class Distribution")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=FIGURE_DPI
    )

    plt.close()

    return counts

#Duplicate sequence IDs
def duplicate_sequence_ids(df):

    logger.info("Checking duplicate sequence IDs...")

    duplicates = df.duplicated(
        subset=["sequence_id"]
    )

    duplicate_df = df[duplicates]

    duplicate_df.to_csv(
        OUTPUT_DIR / "duplicate_sequence_ids.csv",
        index=False
    )

    logger.info(
        f"Duplicate sequence IDs: {len(duplicate_df)}"
    )

    return duplicate_df

#Duplicate feature vectors
def duplicate_feature_vectors(df):

    logger.info("Checking duplicate feature vectors...")

    feature_columns = [

        c

        for c in df.columns

        if c not in

        [

            "sequence_id",

            "genus",

            "accession",

            "header",

            "sequence_length"

        ]

    ]

    duplicates = df.duplicated(
        subset=feature_columns
    )

    duplicate_df = df[duplicates]

    duplicate_df.to_csv(

        OUTPUT_DIR /

        "duplicate_feature_vectors.csv",

        index=False

    )

    logger.info(

        f"Duplicate feature vectors: {len(duplicate_df)}"

    )

    return duplicate_df

def write_summary(summary, counts):

    logger.info("Writing diagnostic report...")

    with open(
        OUTPUT_DIR / "diagnostic_summary.txt",
        "w"
    ) as f:

        f.write(
            "DATASET DIAGNOSTIC REPORT\n"
        )

        f.write(
            "=" * 40 + "\n\n"
        )

        for k, v in summary.items():

            f.write(
                f"{k}: {v}\n"
            )

        f.write("\n")

        f.write(
            f"Imbalance ratio: "
            f"{counts.max()/counts.min():.2f}\n"
        )

        f.write(
            f"Number of genera: "
            f"{len(counts)}\n"
        )

#Identify feature columns
def get_feature_columns(df):
    """
    Return only k-mer feature columns.
    """

    non_feature_columns = {
        "sequence_id",
        "genus",
        "accession",
        "header",
        "sequence_length",
    }

    return [c for c in df.columns if c not in non_feature_columns]

def diagnose_kmers(df):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()

    # 1. Check for raw count vs sequence length inflation
    print(f"Row count range: {X.sum(axis=1).min()} to {X.sum(axis=1).max()}")

    # 2. Convert to Relative Frequencies (Row-wise normalization)
    X_norm = X.div(X.sum(axis=1), axis=0)

    # 3. Calculate variances of normalized frequencies
    variances = X_norm.var()

    print(f"Total 6-mers: {len(variances)}")
    print(f"Absolute zero variance 6-mers: {(variances == 0).sum()}")
    print(f"Top 5 most variable 6-mers:\n{variances.nlargest(5)}")

    return variances


#Constant Features
def constant_features(df):

    logger.info("Checking constant features...")

    feature_cols = get_feature_columns(df)

    selector = VarianceThreshold(threshold=0)

    selector.fit(df[feature_cols])

    keep = selector.get_support()

    constant = np.array(feature_cols)[~keep]

    constant_df = pd.DataFrame(
        {"constant_feature": constant}
    )

    constant_df.to_csv(
        OUTPUT_DIR / "constant_features.csv",
        index=False,
    )

    logger.info(
        f"Constant features: {len(constant)}"
    )

    return constant

#Near-Zero Variance Features
def near_zero_variance(df, threshold=1e-5):

    logger.info("Checking near-zero variance features...")

    feature_cols = get_feature_columns(df)

    variances = df[feature_cols].var()

    nzv = variances[variances < threshold]

    nzv_df = nzv.reset_index()

    nzv_df.columns = [
        "feature",
        "variance"
    ]

    nzv_df.to_csv(
        OUTPUT_DIR / "near_zero_variance.csv",
        index=False,
    )

    logger.info(
        f"Near-zero variance features: {len(nzv_df)}"
    )

    return nzv_df

#Sparse Features
def sparse_features(df, threshold=0.99):

    logger.info("Checking sparse features...")

    feature_cols = get_feature_columns(df)

    sparsity = (
        (df[feature_cols] == 0)
        .mean()
    )

    sparse = sparsity[
        sparsity >= threshold
    ]

    sparse_df = sparse.reset_index()

    sparse_df.columns = [
        "feature",
        "fraction_zero"
    ]

    sparse_df.to_csv(
        OUTPUT_DIR / "sparse_features.csv",
        index=False,
    )

    logger.info(
        f"Sparse features: {len(sparse_df)}"
    )

    return sparse_df

#Feature Statistics
def feature_statistics(df):

    logger.info("Computing feature statistics...")

    feature_cols = get_feature_columns(df)

    stats = pd.DataFrame({

        "mean":
            df[feature_cols].mean(),

        "std":
            df[feature_cols].std(),

        "variance":
            df[feature_cols].var(),

        "minimum":
            df[feature_cols].min(),

        "maximum":
            df[feature_cols].max(),

    })

    stats.to_csv(
        OUTPUT_DIR / "feature_statistics.csv"
    )

    return stats

#Highly Correlated Features
def correlated_features(df, threshold=0.98):

    logger.info("Checking correlated features...")

    feature_cols = get_feature_columns(df)

    corr = (
        df[feature_cols]
        .corr()
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1,
        ).astype(bool)
    )

    correlated = []

    for col in upper.columns:

        high = upper.index[
            upper[col] > threshold
        ].tolist()

        for h in high:

            correlated.append(
                (h, col, upper.loc[h, col])
            )

    corr_df = pd.DataFrame(
        correlated,
        columns=[
            "feature1",
            "feature2",
            "correlation",
        ],
    )

    corr_df.to_csv(
        OUTPUT_DIR /
        "highly_correlated_features.csv",
        index=False,
    )

    logger.info(
        f"Highly correlated feature pairs: "
        f"{len(corr_df)}"
    )

    return corr_df

#Correlation Heatmap
def correlation_heatmap(df, top_n=50):

    logger.info("Generating correlation heatmap...")

    feature_cols = get_feature_columns(df)

    variances = (
        df[feature_cols]
        .var()
        .sort_values(ascending=False)
    )

    selected = variances.head(top_n).index

    corr = df[selected].corr()

    plt.figure(figsize=(10, 8))

    plt.imshow(
        corr,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar()

    plt.title(
        f"Correlation Heatmap ({top_n} most variable features)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "correlation_heatmap.png",
        dpi=FIGURE_DPI,
    )

    plt.close()

#Recommendations
def feature_recommendations(
    constant,
    nzv,
    sparse,
    correlated,
):

    with open(
        OUTPUT_DIR /
        "feature_recommendations.txt",
        "w",
    ) as f:

        f.write(
            "FEATURE ENGINEERING RECOMMENDATIONS\n"
        )

        f.write("=" * 45 + "\n\n")

        if len(constant):

            f.write(
                f"- Remove {len(constant)} constant features.\n"
            )

        if len(nzv):

            f.write(
                f"- Remove {len(nzv)} near-zero variance features.\n"
            )

        if len(sparse):

            f.write(
                f"- Consider removing {len(sparse)} sparse features.\n"
            )

        if len(correlated):

            f.write(
                f"- Consider removing one feature from each of the "
                f"{len(correlated)} highly correlated pairs.\n"
            )

        if (
            len(constant) == 0
            and len(nzv) == 0
            and len(sparse) == 0
        ):

            f.write(
                "No major feature engineering recommendations.\n"
            )

def main():

    create_output_directory()

    df = load_dataset()

    summary = dataset_summary(df)

    check_missing_values(df)

    counts = class_distribution(df)

    duplicate_sequence_ids(df)

    duplicate_feature_vectors(df)

    write_summary(summary, counts)

    logger.info(
        "Dataset diagnostics (Part 1) complete."
    )

    variances = diagnose_kmers(df)

    constant = constant_features(df)

    nzv = near_zero_variance(df)

    sparse = sparse_features(df)

    feature_statistics(df)

    correlated = correlated_features(df)

    correlation_heatmap(df)

    feature_recommendations(
        constant,
        nzv,
        sparse,
        correlated,
    )

if __name__ == "__main__":

    main()



