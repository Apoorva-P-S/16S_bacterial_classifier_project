import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# =============================================================================
# CONFIG -- edit these values directly, then just hit Run
# =============================================================================

# --- Input files (outputs of scripts 1 and 2) ---
KMER_MATRIX_CSV = Path("./kmer_matrix_reduced.csv")
TAXONOMY_TABLE_CSV = Path("./taxonomy_table.csv")

# --- Output location ---
OUTPUT_DIR = Path("./kmer_reduced_model_outputs")

# --- Which model(s) to train: "random_forest", "xgboost", or "both" ---
MODEL_TYPE = "both"

# --- Train/validation/test split ---
# Both fractions are expressed relative to the FULL dataset, not to each
# other -- e.g. TEST_SIZE=0.2, VAL_SIZE=0.2 means 20% test / 20% validation /
# 60% train. The split is done in two group-aware stages (see
# split_data_by_group_three_way) so that TEST_SIZE + VAL_SIZE must be < 1.0.
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42  # fixed seed for reproducibility

# --- Cross-validation (run on the TRAIN split only) ---
RUN_CROSS_VALIDATION = True
CV_FOLDS = 5  # reduce this if any genus has very few unique accessions

# --- Random Forest hyperparameters ---
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",  # helps if genera are imbalanced
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# --- XGBoost hyperparameters ---
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="mlogloss",
    reg_alpha=0.3,
    reg_lambda=1.0,
    base_score=0.5,  # required for SHAP TreeExplainer compatibility with XGBoost 2.x
)

# --- XGBoost early stopping, using the validation split (RF has no
#     equivalent mechanism, so this only applies to XGBoost) ---
EARLY_STOPPING_ROUNDS = 10

# --- How many top features to report ---
TOP_N_FEATURES = 20

# --- SHAP interpretability (runs after each model is trained/evaluated) ---
RUN_SHAP = True
SHAP_MAX_DISPLAY = 15  # top-N k-mers shown in the global bar chart


# =============================================================================
# Functions
# =============================================================================


def load_data(kmer_csv: Path, taxonomy_csv: Path):
    """Load and merge the k-mer matrix with the taxonomy table on sequence_id.

    Guarantees row-count alignment across features, labels, and groups.
    """
    kmer_df = pd.read_csv(kmer_csv)
    taxonomy_df = pd.read_csv(taxonomy_csv)

    if "accession" not in taxonomy_df.columns:
        raise ValueError(
            f"'accession' column not found in {taxonomy_csv}. This is required "
            "as the strain-grouping key for leakage-safe splitting/CV."
        )

    merged = kmer_df.merge(
        taxonomy_df[["sequence_id", "accession", "sequence_length"]],
        on="sequence_id",
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(
            "The merged DataFrame is empty! Check if 'sequence_id' matches between your files."
        )
    if len(merged) < len(kmer_df):
        warnings.warn(
            f"{len(kmer_df) - len(merged)} of {len(kmer_df)} k-mer matrix rows had no "
            "matching sequence_id in the taxonomy table and were dropped."
        )

    non_feature_cols = ["sequence_id", "genus", "accession", "sequence_length", "split"]
    feature_cols = [c for c in merged.columns if c not in non_feature_cols]

    if merged[feature_cols].isna().any().any():
        raise ValueError(
            "NaNs found in k-mer feature columns after merge -- check the upstream matrix."
        )

    X = merged[feature_cols].values
    y_raw = merged["genus"].values
    sequence_ids = merged["sequence_id"].values
    accessions = merged["accession"].values

    # If the k-mer matrix carries a "split" column (written by the leakage-fixed
    # R reduction script, sourced from split_manifest.csv), use it directly so
    # the train/val/test assignment here is guaranteed identical to the one the
    # R script used when deciding which k-mers to keep. Falls back to None if
    # absent, in which case main() derives a fresh split itself.
    split_labels = merged["split"].values if "split" in merged.columns else None

    assert (
        len(X) == len(y_raw) == len(accessions)
    ), f"Length mismatch! X: {len(X)}, y: {len(y_raw)}, accessions: {len(accessions)}"

    return X, y_raw, feature_cols, sequence_ids, accessions, split_labels


def split_from_labels(X, y_encoded, accessions, split_labels):
    """Uses a pre-computed train/val/test assignment (from the "split" column
    written into kmer_matrix_reduced.csv by the leakage-fixed R reduction
    script) instead of deriving a fresh split here. This guarantees the split
    used for final model training/evaluation is IDENTICAL to the one the R
    script used when deciding which k-mers to keep -- see
    0_make_split_manifest.py and visualize_correlation_and_reduce_features_fixed.R.
    """
    X = np.asarray(X)
    y_encoded = np.asarray(y_encoded).ravel()
    accessions = np.asarray(accessions).ravel()
    split_labels = np.asarray(split_labels).ravel()

    valid = {"train", "val", "test"}
    unexpected = set(np.unique(split_labels)) - valid
    if unexpected:
        raise ValueError(f"Unexpected split label(s) found: {unexpected}. Expected only {valid}.")

    train_mask = split_labels == "train"
    val_mask = split_labels == "val"
    test_mask = split_labels == "test"

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train, y_val, y_test = y_encoded[train_mask], y_encoded[val_mask], y_encoded[test_mask]
    groups_train, groups_val, groups_test = accessions[train_mask], accessions[val_mask], accessions[test_mask]

    # Same hard leakage check as split_data_by_group_three_way -- re-verified
    # here rather than just trusted, in case the manifest and this k-mer
    # matrix somehow fell out of sync.
    train_acc, val_acc, test_acc = set(groups_train), set(groups_val), set(groups_test)
    overlap = (train_acc & val_acc) | (train_acc & test_acc) | (val_acc & test_acc)
    if overlap:
        preview = sorted(overlap)[:5]
        raise RuntimeError(
            f"Leakage check failed: {len(overlap)} accession(s) appear in more than one "
            f"split (e.g. {preview}). The 'split' column and the underlying data have "
            "fallen out of sync -- regenerate split_manifest.csv and rerun the R script."
        )

    all_classes = set(np.unique(y_encoded))
    for name, y_split in (("train", y_train), ("validation", y_val), ("test", y_test)):
        missing = all_classes - set(np.unique(y_split))
        if missing:
            warnings.warn(f"{len(missing)} class label(s) absent from the {name} split: {sorted(missing)}.")

    print(
        f"\nUsing pre-computed split -> Train: {len(X_train)} | Validation: {len(X_val)} | "
        f"Test: {len(X_test)} sequences"
    )
    print(
        f"Unique accessions/strains -> Train: {len(train_acc)} | "
        f"Validation: {len(val_acc)} | Test: {len(test_acc)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test


def split_data_by_group_three_way(X, y_encoded, groups, test_size, val_size, random_state):
    """Group-aware train/validation/test split so that sequences from the
    same accession/strain never cross-contaminate any of the three sets.

    Implemented as two sequential GroupShuffleSplit stages:
      Stage 1: carve the TEST set off the full dataset (test_size, relative
               to the full dataset).
      Stage 2: carve the VALIDATION set off the remaining train+val pool
               (val_size, also relative to the FULL dataset -- rescaled
               internally to a fraction of the remaining pool).

    test_size + val_size must be < 1.0. Raises if that's violated, and
    raises again (as a hard leakage check, not just a warning) if any
    accession somehow ends up in more than one of the three resulting sets.
    """
    X = np.asarray(X)
    y_encoded = np.asarray(y_encoded).ravel()
    groups = np.asarray(groups).ravel()

    remaining_fraction = 1.0 - test_size
    if val_size <= 0 or val_size >= remaining_fraction:
        raise ValueError(
            f"VAL_SIZE ({val_size}) must be > 0 and less than (1 - TEST_SIZE) = "
            f"{remaining_fraction:.3f}."
        )

    # --- Stage 1: test vs. (train+val) ---
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(X, y_encoded, groups))

    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y_encoded[trainval_idx], y_encoded[test_idx]
    groups_trainval, groups_test = groups[trainval_idx], groups[test_idx]

    # --- Stage 2: train vs. val, drawn from the train+val pool only ---
    val_size_within_trainval = val_size / remaining_fraction
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_within_trainval, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups_trainval))

    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
    groups_train, groups_val = groups_trainval[train_idx], groups_trainval[val_idx]

    # --- Hard leakage check: no accession may appear in more than one split ---
    train_acc, val_acc, test_acc = set(groups_train), set(groups_val), set(groups_test)
    overlap = (train_acc & val_acc) | (train_acc & test_acc) | (val_acc & test_acc)
    if overlap:
        preview = sorted(overlap)[:5]
        raise RuntimeError(
            f"Leakage check failed: {len(overlap)} accession(s) appear in more than one "
            f"split (e.g. {preview}). This should be impossible with GroupShuffleSplit -- "
            "check for duplicate accession values across your input data."
        )

    # --- Soft check: flag any genus missing entirely from a split (can't be scored there) ---
    all_classes = set(np.unique(y_encoded))
    for name, y_split in (("train", y_train), ("validation", y_val), ("test", y_test)):
        missing = all_classes - set(np.unique(y_split))
        if missing:
            warnings.warn(f"{len(missing)} class label(s) absent from the {name} split: {sorted(missing)}.")

    print(
        f"\nGroup split -> Train: {len(X_train)} | Validation: {len(X_val)} | "
        f"Test: {len(X_test)} sequences"
    )
    print(
        f"Unique accessions/strains -> Train: {len(train_acc)} | "
        f"Validation: {len(val_acc)} | Test: {len(test_acc)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test


def train_random_forest(X_train, y_train, X_val=None, y_val=None):
    """Random Forest has no native validation-based early-stopping mechanism,
    so X_val/y_val are accepted (for a signature matching train_xgboost) but
    unused here.
    """
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """Trains XGBoost, using the validation split for early stopping when
    provided. Early stopping is configured only on this final-fit model --
    NOT on the estimator used for cross-validation (see build_cv_estimator),
    because cross_val_score clones and refits the estimator per fold without
    access to a per-fold eval_set, and an XGBClassifier configured with
    early_stopping_rounds raises an error if fit() is called without one.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "xgboost is not installed. Run `pip install xgboost` to use MODEL_TYPE='xgboost' or 'both'."
        )

    params = dict(XGB_PARAMS)
    if X_val is not None and y_val is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
    return model


def build_cv_estimator(model_name):
    """Builds a *plain* estimator (no validation-based early stopping) purely
    for cross_val_score. See the docstring in train_xgboost for why the
    early-stopping-configured model can't be reused here directly.
    """
    if model_name == "random_forest":
        return RandomForestClassifier(**RF_PARAMS)
    if model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(**XGB_PARAMS)  # note: no early_stopping_rounds here
    raise ValueError(f"Unknown model_name: {model_name!r}")


def run_group_cross_validation(cv_estimator, X_train, y_train, groups_train, cv_folds):
    """Runs StratifiedGroupKFold on the TRAIN split (validation and test sets
    are never touched here) to prevent leakage across internal folds while
    keeping genus distributions balanced.
    """
    df_check = pd.DataFrame({"y": y_train, "grp": groups_train})
    min_groups_per_class = df_check.groupby("y")["grp"].nunique().min()

    effective_folds = min(cv_folds, min_groups_per_class)

    if effective_folds < cv_folds and min_groups_per_class >= 2:
        warnings.warn(
            f"Smallest genus only has {min_groups_per_class} unique accessions; "
            f"reducing CV folds from {cv_folds} to {effective_folds}."
        )

    if effective_folds < 2:
        print(
            f"Skipping cross-validation: smallest genus has {min_groups_per_class} unique accession(s). "
            "Stratified Group CV requires at least 2 unique groups per class."
        )
        return None

    cv_strategy = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(
        cv_estimator,
        X_train,
        y_train,
        groups=groups_train,
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1,
    )

    print(
        f"Leakage-free group cross-val macro-F1 ({effective_folds}-fold, TRAIN split only): "
        f"mean={scores.mean():.3f}  std={scores.std():.3f}  scores={np.round(scores, 3)}"
    )

    return scores


def evaluate_on_validation(model, X_val, y_val, label_encoder, model_name, output_dir):
    """Lightweight validation-set check, run on the final-fit model before
    the held-out test evaluation. Intended for model-selection / sanity
    checking (e.g. did XGBoost's early stopping converge sensibly, does RF
    vs. XGBoost differ noticeably before we touch the test set at all) --
    NOT as the reported headline metric. The test-set evaluation in
    evaluate_model() below remains the authoritative, reported result.
    """
    y_pred = model.predict(X_val)
    target_names = label_encoder.classes_
    n_classes = len(target_names)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"\n--- Validation-set check ({model_name}) ---")
    print(f"Validation Accuracy    : {acc:.4f}")
    print(f"Validation Macro-F1    : {f1_macro:.4f}")
    print(f"Validation Weighted-F1 : {f1_weighted:.4f}")

    report_dict = classification_report(
        y_val, y_pred, labels=range(n_classes), target_names=target_names,
        output_dict=True, zero_division=0,
    )
    report_path = output_dir / f"classification_report_val_{model_name}.csv"
    pd.DataFrame(report_dict).transpose().to_csv(report_path)
    print(f"Saved validation classification report: {report_path}")

    return {"val_accuracy": acc, "val_f1_macro": f1_macro, "val_f1_weighted": f1_weighted}


def compute_roc_auc(y_test, y_proba, n_classes):
    """Computes per-class and macro ROC-AUC using One-vs-Rest, skipping any
    class that has zero positive examples in y_test.
    """
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    per_class_auc = {}
    curves = {}

    for i in range(n_classes):
        n_pos = int(y_test_bin[:, i].sum())
        if n_pos == 0 or n_pos == len(y_test_bin):
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        per_class_auc[i] = auc(fpr, tpr)
        curves[i] = (fpr, tpr)

    macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan")
    return macro_auc, per_class_auc, curves, y_test_bin


def plot_roc_curve(y_test, y_proba, target_names, model_name, output_dir):
    n_classes = len(target_names)
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        macro_auc = roc_auc
        per_class_auc = {1: roc_auc}
    else:
        macro_auc, per_class_auc, curves, _ = compute_roc_auc(y_test, y_proba, n_classes)
        for i, (fpr, tpr) in curves.items():
            ax.plot(fpr, tpr, lw=1.5, label=f"{target_names[i]} (AUC = {per_class_auc[i]:.2f})")
        skipped = n_classes - len(per_class_auc)
        skip_note = f" ({skipped} class(es) skipped, absent from test)" if skipped else ""
        ax.plot([], [], " ", label=f"Macro-Average AUC = {macro_auc:.4f}{skip_note}")

    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves (One-vs-Rest) - {model_name.upper()}", fontsize=13)
    ax.legend(loc="lower right", fontsize=8 if n_classes > 5 else 10)
    fig.tight_layout()

    roc_path = output_dir / f"roc_curve_{model_name}.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve plot: {roc_path}")

    return macro_auc, {target_names[i]: v for i, v in per_class_auc.items()}


def evaluate_model(model, X_test, y_test, label_encoder, model_name, output_dir):
    """Evaluates the FINAL model on the held-out TEST set. This is the
    authoritative, reported evaluation -- the validation set (see
    evaluate_on_validation) is a separate, earlier check and should not be
    conflated with this result.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    target_names = label_encoder.classes_
    n_classes = len(target_names)

    report_dict = classification_report(
        y_test, y_pred, labels=range(n_classes),
        target_names=target_names, output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = output_dir / f"classification_report_{model_name}.csv"
    report_df.to_csv(report_path)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    r2_discrete = r2_score(y_test, y_pred)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    if n_classes == 2 and y_proba.shape[1] == 2:
        r2_proba = r2_score(y_test, y_proba[:, 1])
    else:
        r2_proba = r2_score(y_test_bin, y_proba, multioutput="uniform_average")

    macro_auc, per_class_auc = plot_roc_curve(y_test, y_proba, target_names, model_name, output_dir)

    print("\n" + "=" * 60)
    print(f" PERFORMANCE SUMMARY: {model_name.upper()} (HOLDOUT TEST)")
    print("=" * 60)
    print(f"Global Accuracy       : {acc:.4f}")
    print(f"Macro ROC-AUC (OvR)   : {macro_auc:.4f}")
    print(f"R2 (discrete labels)  : {r2_discrete:.4f}  [not a meaningful metric for nominal classes]")
    print(f"R2 (probability space): {r2_proba:.4f}  [supplementary; prefer F1/AUC for model comparisons]")
    print(f"Macro Average         : Precision={prec_macro:.4f} | Recall={rec_macro:.4f} | F1={f1_macro:.4f}")
    print(f"Weighted Average      : Precision={prec_weighted:.4f} | Recall={rec_weighted:.4f} | F1={f1_weighted:.4f}")
    print("-" * 60)
    print(f"\n--- Detailed Per-Class Report ({model_name}) ---")
    print(classification_report(
        y_test, y_pred, labels=range(n_classes), target_names=target_names, zero_division=0,
    ))
    print(f"Saved complete metrics CSV: {report_path}")

    labels_present = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    display_labels = [target_names[i] for i in labels_present]

    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(display_labels)), max(7, 0.4 * len(display_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
    ax.set_title(f"Confusion Matrix - {model_name.upper()}", fontsize=12, pad=15)
    fig.tight_layout()

    cm_path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix chart: {cm_path}")

    metrics_summary = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": macro_auc,
        "roc_auc_per_class": per_class_auc,
        "r2_discrete_labels_NOT_MEANINGFUL": r2_discrete,
        "r2_probability_space": r2_proba,
    }
    metrics_path = output_dir / f"metrics_summary_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2, default=float)
    print(f"Saved metrics summary JSON: {metrics_path}")

    return report_df, metrics_summary


def save_feature_importance(model, feature_cols, model_name, output_dir, top_n):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"kmer": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    importance_path = output_dir / f"feature_importance_{model_name}.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\nTop {top_n} most important k-mers ({model_name}):")
    print(importance_df.head(top_n).to_string(index=False))
    print(f"Saved: {importance_path}")


def run_shap_analysis(model, X_test, feature_cols, label_encoder, model_name, output_dir, max_display):
    """Generates a TreeExplainer-based SHAP analysis, compatible with both
    RandomForestClassifier and XGBClassifier, for the multi-class k-mer
    feature matrix. Runs on the TEST split (not validation), so the
    interpretability output describes the same held-out data the reported
    performance metrics in evaluate_model() were computed on.

    Called wrapped in try/except from run_pipeline_for_model: a SHAP
    failure (version incompatibilities, memory, etc.) should not cost the
    already-trained model, its saved metrics, or its feature-importance
    CSV -- those are all written before this function is ever called.
    """
    print("\n" + "=" * 60)
    print(f" LAUNCHING SHAP ANALYSIS FOR MODEL: {model_name.upper()}")
    print("=" * 60)

    explainer = shap.TreeExplainer(model)
    shap_results = explainer(X_test)

    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    raw_values = shap_results.values if hasattr(shap_results, "values") else shap_results

    # --- Global multi-class feature-importance bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(raw_values, X_test_df, plot_type="bar", show=False, max_display=max_display)
    plt.title(f"Global K-mer Feature Importance Across All Genera ({model_name.upper()})")
    fig.tight_layout()
    global_path = output_dir / f"shap_global_importance_{model_name}.png"
    fig.savefig(global_path, dpi=150)
    plt.close(fig)
    print(f"Saved global SHAP chart: {global_path}")

    # --- Per-genus directional scatter plots ---
    print("Generating directional scatter profiles for each genus class...")
    for class_idx, genus_name in enumerate(label_encoder.classes_):
        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(raw_values, list):
            class_shap_matrix = raw_values[class_idx]  # Random-Forest-style: list of 2D arrays
        elif len(raw_values.shape) == 3:
            class_shap_matrix = raw_values[:, :, class_idx]  # XGBoost-style: single 3D array
        else:
            class_shap_matrix = raw_values  # fallback for older SHAP versions

        shap.summary_plot(class_shap_matrix, X_test_df, show=False, max_display=max_display)
        plt.title(f"Directional K-mer Impact Profile -> Genus: {genus_name} ({model_name.upper()})")
        fig.tight_layout()

        clean_name = str(genus_name).replace(" ", "_").replace("/", "_")
        class_path = output_dir / f"shap_profile_{model_name}_{clean_name}.png"
        fig.savefig(class_path, dpi=150)
        plt.close(fig)

    print(f"Completed per-genus SHAP profiles for {model_name.upper()}.")


def save_model(model, label_encoder, model_name, output_dir):
    model_path = output_dir / f"model_{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")

    encoder_path = output_dir / "label_encoder.joblib"
    if not encoder_path.exists():
        joblib.dump(label_encoder, encoder_path)
        print(f"Saved label encoder: {encoder_path}")


def run_pipeline_for_model(
    model_name, train_fn, X_train, y_train, X_val, y_val, X_test, y_test,
    groups_train, feature_cols, label_encoder, output_dir,
):
    print("\n" + "=" * 50)
    print(f" TRAINING MODEL: {model_name.upper()}")
    print("=" * 50)

    if RUN_CROSS_VALIDATION:
        cv_estimator = build_cv_estimator(model_name)
        run_group_cross_validation(cv_estimator, X_train, y_train, groups_train, CV_FOLDS)

    # Final model: trained on the TRAIN split, using the VALIDATION split for
    # early stopping where the model architecture supports it (XGBoost only).
    model = train_fn(X_train, y_train, X_val, y_val)

    evaluate_on_validation(model, X_val, y_val, label_encoder, model_name, output_dir)
    evaluate_model(model, X_test, y_test, label_encoder, model_name, output_dir)
    save_feature_importance(model, feature_cols, model_name, output_dir, TOP_N_FEATURES)
    save_model(model, label_encoder, model_name, output_dir)

    # SHAP runs last and is wrapped defensively: a SHAP failure (version
    # incompatibilities, memory, etc.) must not cost the model, metrics, or
    # feature-importance CSV already written above.
    if RUN_SHAP:
        if not SHAP_AVAILABLE:
            print(f"[SKIPPED] shap is not installed -- run `pip install shap` to enable "
                  f"interpretability analysis for {model_name}.")
        else:
            try:
                run_shap_analysis(model, X_test, feature_cols, label_encoder, model_name, output_dir, SHAP_MAX_DISPLAY)
            except Exception as e:
                print(f"[WARNING] SHAP analysis failed for {model_name}, skipping it: {e}")

    return model


def save_run_metadata(output_dir, feature_cols, label_encoder, n_train, n_val, n_test):
    metadata = {
        "model_type": MODEL_TYPE,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "n_features": len(feature_cols),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "genera": list(label_encoder.classes_),
        "rf_params": RF_PARAMS if MODEL_TYPE in ("random_forest", "both") else None,
        "xgb_params": XGB_PARAMS if MODEL_TYPE in ("xgboost", "both") else None,
        "early_stopping_rounds_xgboost": EARLY_STOPPING_ROUNDS if MODEL_TYPE in ("xgboost", "both") else None,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_TYPE not in ("random_forest", "xgboost", "both"):
        raise ValueError(f"MODEL_TYPE must be 'random_forest', 'xgboost', or 'both', got {MODEL_TYPE!r}")

    X, y_raw, feature_cols, sequence_ids, accessions, split_labels = load_data(KMER_MATRIX_CSV, TAXONOMY_TABLE_CSV)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    if split_labels is not None:
        print("Found a 'split' column in the k-mer matrix -- using the pre-computed split "
              "(from split_manifest.csv, via the R feature-reduction step) instead of "
              "deriving a new one.")
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            groups_train, groups_val, groups_test,
        ) = split_from_labels(X, y_encoded, accessions, split_labels)
    else:
        warnings.warn(
            "No 'split' column found in the k-mer matrix -- deriving a fresh train/val/test "
            "split independently of any upstream feature-selection step. If the k-mer matrix "
            "was reduced using a train-only-fitted correlation step, run "
            "0_make_split_manifest.py first and make sure the R script carries the 'split' "
            "column through into kmer_matrix_reduced.csv, so both stages agree on the same split."
        )
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            groups_train, groups_val, groups_test,
        ) = split_data_by_group_three_way(X, y_encoded, accessions, TEST_SIZE, VAL_SIZE, RANDOM_STATE)

    save_run_metadata(OUTPUT_DIR, feature_cols, label_encoder, len(X_train), len(X_val), len(X_test))

    if MODEL_TYPE in ("random_forest", "both"):
        run_pipeline_for_model(
            "random_forest", train_random_forest, X_train, y_train, X_val, y_val, X_test, y_test,
            groups_train, feature_cols, label_encoder, OUTPUT_DIR,
        )

    if MODEL_TYPE in ("xgboost", "both"):
        run_pipeline_for_model(
            "xgboost", train_xgboost, X_train, y_train, X_val, y_val, X_test, y_test,
            groups_train, feature_cols, label_encoder, OUTPUT_DIR,
        )
