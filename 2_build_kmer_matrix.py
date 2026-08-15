#!/usr/bin/env python3
"""
2_build_kmer_matrix.py
========================
Builds a k-mer frequency feature matrix from a folder of per-genus
FASTA files -- the standard feature representation for classical ML
models (Random Forest, XGBoost, Naive Bayes) on 16S sequences, and
conceptually what the RDP classifier's Naive Bayes model is built on.

In addition to writing the raw feature matrix, this script now runs a
diagnostic unsupervised pipeline on top of it:

    1. StandardScaler is fit on the *entire* k-mer feature matrix
       (all sequences, all genera together -- a single global scaler,
       not one per genus/class) so every k-mer column is centered to
       mean 0 / unit variance. This matters because raw k-mer
       frequencies are compositional (they sum to ~1 per row) and have
       very different variances across columns -- PCA/UMAP/t-SNE on
       unscaled compositional data is dominated by the highest-variance
       k-mers (usually the most common/low-information ones like poly-A
       tracts) rather than the discriminative signal.
    2. PCA, UMAP, and t-SNE are each fit on the scaled matrix to produce
       2D embeddings, purely for *diagnostic visualization* of how well
       genera separate in k-mer space -- this is exploratory QC, not a
       replacement for a supervised classifier.
    3. Each embedding is plotted as a scatter plot colored by genus and
       saved as a PNG, and the embedding coordinates themselves are
       saved as CSVs for downstream inspection.

Usage:
    python 2_build_kmer_matrix.py \\
        --input_dir  ./cleaned_fasta \\
        --output_csv ./2_kmer_matrix.csv \\
        --k 6 \\
        --plot_dir ./kmer_diagnostics

Output:
    <output_csv>                       one row per sequence, one column per
                                        possible k-mer (4^k columns for k=6
                                        -> 4096 columns), values = normalized
                                        frequency (count / total k-mers in
                                        that sequence), plus sequence_id and
                                        genus columns for joining/labeling.
    <plot_dir>/kmer_matrix_scaled.csv  the same matrix after the global
                                        StandardScaler transform (features
                                        only, plus sequence_id/genus).
    <plot_dir>/pca_embedding.csv        2D PCA coordinates + genus
    <plot_dir>/umap_embedding.csv       2D UMAP coordinates + genus (if umap-learn installed)
    <plot_dir>/tsne_embedding.csv       2D t-SNE coordinates + genus
    <plot_dir>/pca_scatter.png
    <plot_dir>/umap_scatter.png
    <plot_dir>/tsne_scatter.png

Any k-mer containing a non-ACGU base (shouldn't remain after your
cleaning step, but checked defensively) is skipped when counting.
"""
import argparse
import itertools
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from fasta_utils import load_genus_fasta_folder


BASES = "ACGU"


def all_possible_kmers(k, bases):
    return ["".join(p) for p in itertools.product(bases, repeat=k)]


def count_kmers(sequence, k, bases):
    """Return a Counter of k-mer counts for one sequence, skipping any
    k-mer window that contains a base outside the given alphabet."""
    counts = Counter()
    valid = set(bases)
    seq = sequence.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if all(b in valid for b in kmer):
            counts[kmer] += 1
    return counts


def build_kmer_matrix(input_dir, k, bases=BASES):
    records = load_genus_fasta_folder(input_dir)
    kmer_columns = all_possible_kmers(k, bases)
    kmer_index = {kmer: idx for idx, kmer in enumerate(kmer_columns)}

    matrix = np.zeros((len(records), len(kmer_columns)), dtype=np.float32)
    meta_rows = []
    keep_rows = []  # track which matrix rows actually got a meta row (non-empty seqs)

    for row_idx, record in enumerate(records):
        counts = count_kmers(record.sequence, k, bases)
        total = sum(counts.values())
        if total == 0:
            continue  # sequence shorter than k (or no valid k-mers); dropped entirely
        for kmer, count in counts.items():
            matrix[row_idx, kmer_index[kmer]] = count / total  # normalized frequency
        meta_rows.append({"sequence_id": record.sequence_id, "genus": record.genus})
        keep_rows.append(row_idx)

    if len(keep_rows) < len(records):
        warnings.warn(
            f"Dropped {len(records) - len(keep_rows)} of {len(records)} sequences "
            f"with zero valid k-mers (shorter than k={k} or all non-{bases} bases)."
        )

    matrix = matrix[keep_rows]  # keep matrix/meta rows aligned 1:1
    meta_df = pd.DataFrame(meta_rows)
    kmer_df = pd.DataFrame(matrix, columns=kmer_columns)
    full_df = pd.concat([meta_df.reset_index(drop=True), kmer_df.reset_index(drop=True)], axis=1)
    return full_df


def scale_feature_matrix(df, feature_cols):
    """Fit one global StandardScaler across all sequences/genera and
    transform the k-mer feature block. Returns the scaled ndarray and
    the fitted scaler (so it can be reused/serialized downstream, e.g.
    to transform held-out sequences the same way before classification)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)
    return scaled, scaler


def run_pca(X, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    return embedding, explained


def run_tsne(X, n_components=2, perplexity=30, random_state=42):
    n_samples = X.shape[0]
    # t-SNE requires perplexity < n_samples; clamp down for small datasets
    # rather than letting sklearn raise.
    eff_perplexity = min(perplexity, max(5, (n_samples - 1) // 3))
    if eff_perplexity != perplexity:
        warnings.warn(
            f"t-SNE perplexity {perplexity} too high for {n_samples} samples; "
            f"using {eff_perplexity} instead."
        )
    tsne = TSNE(
        n_components=n_components,
        perplexity=eff_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(X)


def run_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """Returns None if umap-learn isn't installed, so the rest of the
    pipeline (PCA/t-SNE/CSV output) can still run."""
    try:
        import umap
    except ImportError:
        warnings.warn(
            "umap-learn is not installed; skipping UMAP embedding. "
            "Install with `pip install umap-learn` to enable it."
        )
        return None
    n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def plot_embedding(embedding, genus_labels, title, output_path, max_legend_genera=25):
    """Scatter plot of a 2D embedding colored by genus. Uses a headless
    matplotlib backend so this works fine on servers/CI with no display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genera = pd.Series(genus_labels)
    uniq = genera.unique()
    cmap = plt.get_cmap("tab20" if len(uniq) <= 20 else "nipy_spectral")

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, g in enumerate(uniq):
        mask = (genera == g).values
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.7,
            color=cmap(i / max(1, len(uniq) - 1)),
            label=g,
        )
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    if len(uniq) <= max_legend_genera:
        ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.text(
            0.99, 0.01, f"{len(uniq)} genera (legend omitted)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_diagnostics(full_df, feature_cols, plot_dir, args):
    plot_dir.mkdir(parents=True, exist_ok=True)

    scaled, _scaler = scale_feature_matrix(full_df, feature_cols)

    scaled_df = pd.concat(
        [
            full_df[["sequence_id", "genus"]].reset_index(drop=True),
            pd.DataFrame(scaled, columns=feature_cols),
        ],
        axis=1,
    )
    scaled_df.to_csv(plot_dir / "kmer_matrix_scaled.csv", index=False)

    genus_labels = full_df["genus"].values

    # PCA
    pca_emb, explained = run_pca(scaled, n_components=2, random_state=args.random_state)
    pd.DataFrame({
        "sequence_id": full_df["sequence_id"],
        "genus": full_df["genus"],
        "PC1": pca_emb[:, 0],
        "PC2": pca_emb[:, 1],
    }).to_csv(plot_dir / "pca_embedding.csv", index=False)
    plot_embedding(
        pca_emb, genus_labels,
        f"PCA (PC1 {explained[0]*100:.1f}% / PC2 {explained[1]*100:.1f}% var)",
        plot_dir / "pca_scatter.png",
    )
    print(f"PCA: PC1={explained[0]*100:.1f}% PC2={explained[1]*100:.1f}% variance explained")

    # UMAP (optional dependency)
    umap_emb = run_umap(
        scaled, n_components=2,
        n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
        random_state=args.random_state,
    )
    if umap_emb is not None:
        pd.DataFrame({
            "sequence_id": full_df["sequence_id"],
            "genus": full_df["genus"],
            "UMAP1": umap_emb[:, 0],
            "UMAP2": umap_emb[:, 1],
        }).to_csv(plot_dir / "umap_embedding.csv", index=False)
        plot_embedding(umap_emb, genus_labels, "UMAP", plot_dir / "umap_scatter.png")

    # t-SNE
    tsne_emb = run_tsne(
        scaled, n_components=2,
        perplexity=args.tsne_perplexity, random_state=args.random_state,
    )
    pd.DataFrame({
        "sequence_id": full_df["sequence_id"],
        "genus": full_df["genus"],
        "TSNE1": tsne_emb[:, 0],
        "TSNE2": tsne_emb[:, 1],
    }).to_csv(plot_dir / "tsne_embedding.csv", index=False)
    plot_embedding(tsne_emb, genus_labels, "t-SNE", plot_dir / "tsne_scatter.png")

    print(f"Diagnostic outputs written to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Build a k-mer frequency matrix from per-genus FASTA files.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--k", type=int, default=6, help="k-mer length (default 6 -> 4096 features)")
    parser.add_argument("--alphabet", default=BASES, help="Valid base alphabet (default ACGT; use ACGU for RNA FASTAs)")

    parser.add_argument("--plot_dir", type=Path, default=None,
                         help="If set, run StandardScaler + PCA/UMAP/t-SNE diagnostics and write "
                              "embeddings/plots here. Omit to skip diagnostics entirely.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)

    args = parser.parse_args()

    df = build_kmer_matrix(args.input_dir, args.k, bases=args.alphabet)
    df.to_csv(args.output_csv, index=False)
    n_feature_cols = df.shape[1] - 2  # minus sequence_id, genus
    print(f"Wrote {len(df)} rows x {n_feature_cols} k-mer features to {args.output_csv}")
    print(f"(k={args.k} -> {len(args.alphabet)**args.k} possible k-mer columns)")

    if args.plot_dir is not None:
        feature_cols = [c for c in df.columns if c not in ("sequence_id", "genus")]
        if df["genus"].nunique() < 2:
            print("Only one genus present; skipping cluster-visualization diagnostics.", file=sys.stderr)
        else:
            run_diagnostics(df, feature_cols, args.plot_dir, args)


if __name__ == "__main__":
    main()
