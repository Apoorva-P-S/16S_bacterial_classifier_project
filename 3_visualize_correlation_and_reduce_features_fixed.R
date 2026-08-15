# =============================================================================
# 3_visualize_correlation_and_reduce_features.R  (leakage-fixed version)
# =============================================================================
# Visualizes the correlation structure of the k-mer feature matrix,
# performs a PCA class-separability plot, and drops redundant k-mers.
#
# LEAKAGE FIX (read this before running):
#   The correlation matrix and the greedy feature-drop decision are now fit
#   on the TRAIN split ONLY, using split_manifest.csv (see
#   0_make_split_manifest.py), rather than on the full pooled dataset. The
#   resulting retained-feature list is then applied to ALL rows (train,
#   val, AND test) when writing REDUCED_KMER_CSV -- so every split ends up
#   with the same columns, but which columns were kept was decided without
#   ever looking at val/test rows.
#
#   If TRAIN_MANIFEST_CSV is not found, the script falls back to the old
#   behavior (fit on the full dataset) but prints an explicit warning, since
#   that constitutes a real (if mild, label-blind) leakage risk into
#   whatever train/val/test split is applied downstream.
#
# VISUALIZATIONS PRODUCED (all still descriptive of the FULL cleaned
# dataset by default -- see PLOTS_USE_TRAIN_ONLY below to restrict these
# too, for full rigor if these plots will be presented as performance
# evidence rather than purely exploratory description):
#   1. Correlation heatmap, hierarchically clustered         (base R only)
#   2. "Ordered correlogram" with clustered blocks              (corrplot, optional upgrade)
#   3. Standalone feature dendrogram (1 - |correlation| distance) (base R only)
#   4. Histogram of all pairwise correlation values             (base R only)
#   5. Network graph of highly-correlated feature pairs          (igraph, optional)
#   6. Dimensionality Reduction & Class Separability Plot (PCA) (base R / ggplot2)
#
# DESIGNED FOR RSTUDIO: edit the CONFIG block below, then click "Source".
# =============================================================================

# =============================================================================
# CONFIG -- edit these values directly
# =============================================================================

KMER_CSV               <- "./2_kmer_matrix.csv"
OUTPUT_DIR             <- "./correlation_outputs"
REDUCED_KMER_CSV       <- "./kmer_matrix_reduced.csv"

# --- Leakage fix: point this at the manifest from 0_make_split_manifest.py ---
# Set to "" (empty string) to disable and fall back to the old (leaky)
# full-dataset fitting behavior -- not recommended, see warning above.
TRAIN_MANIFEST_CSV     <- "./split_manifest.csv"

# If TRUE, plots 1/2/3/5/6 also use ONLY the train split (fully rigorous,
# but the plots then describe a subset rather than the whole cleaned
# dataset). If FALSE (default), the correlation-matrix DECISION is still
# train-only, but the descriptive plots use the full dataset for a more
# complete exploratory picture. Keep FALSE for exploratory use; set TRUE if
# these plots will be presented as part of a performance/rigor claim.
PLOTS_USE_TRAIN_ONLY   <- FALSE

CORR_METHOD            <- "pearson"   # "pearson" or "spearman"
CORR_CUTOFF            <- 0.95        # features correlated above this get reduced
MAX_FEATURES_TO_PLOT   <- 60          # top-variance k-mers shown in heatmap/dendrogram/network
NETWORK_CORR_THRESHOLD <- 0.85        # edge-drawing threshold for the optional network plot

# =============================================================================
# Setup
# =============================================================================

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
META_COLS <- c("sequence_id", "genus")

# =============================================================================
# Load data + apply train/val/test split labels (if a manifest is provided)
# =============================================================================

load_kmer_features <- function(kmer_csv, manifest_csv) {
  df <- read.csv(kmer_csv, stringsAsFactors = FALSE)

  # --- Attach split labels from the manifest, if available ---
  use_manifest <- nzchar(manifest_csv) && file.exists(manifest_csv)
  if (use_manifest) {
    manifest <- read.csv(manifest_csv, stringsAsFactors = FALSE)[, c("sequence_id", "split")]
    n_before <- nrow(df)
    df <- merge(df, manifest, by = "sequence_id", all.x = TRUE, sort = FALSE)
    if (nrow(df) != n_before) {
      stop("Row count changed after merging split_manifest.csv -- check for duplicate sequence_ids.")
    }
    n_missing <- sum(is.na(df$split))
    if (n_missing > 0) {
      stop(sprintf(
        "%d sequence(s) in %s have no matching entry in %s -- regenerate the manifest.",
        n_missing, kmer_csv, manifest_csv
      ))
    }
    cat(sprintf("Loaded split manifest: %s\n", manifest_csv))
    print(table(df$split))
  } else {
    df$split <- NA_character_
    cat("\n*** WARNING: no split manifest found (TRAIN_MANIFEST_CSV = '", manifest_csv, "'). ***\n", sep = "")
    cat("*** Correlation matrix and feature reduction will be fit on the FULL pooled  ***\n")
    cat("*** dataset (train+val+test combined). This is a mild, label-blind leakage   ***\n")
    cat("*** risk into any downstream train/val/test split -- run                     ***\n")
    cat("*** 0_make_split_manifest.py first and point TRAIN_MANIFEST_CSV at it.        ***\n\n")
  }

  feature_cols <- setdiff(colnames(df), c(META_COLS, "split"))
  X <- as.matrix(df[, feature_cols])

  # Drop zero-variance k-mers first. IMPORTANT: variance is computed on the
  # same row subset (train-only, if a manifest is present) that will be used
  # to fit the correlation matrix below, for the same leakage-avoidance reason.
  fit_rows <- if (use_manifest) which(df$split == "train") else seq_len(nrow(df))
  variances <- apply(X[fit_rows, , drop = FALSE], 2, var, na.rm = TRUE)
  zero_var_cols <- names(variances)[is.na(variances) | variances == 0]
  if (length(zero_var_cols) > 0) {
    cat(sprintf("Dropping %d zero-variance k-mer columns (variance computed on %s rows).\n",
                length(zero_var_cols), if (use_manifest) "train-only" else "all"))
    X <- X[, !(colnames(X) %in% zero_var_cols), drop = FALSE]
  }

  list(df = df, X = X, variances = variances[colnames(X)], fit_rows = fit_rows, use_manifest = use_manifest)
}

compute_correlation_matrix <- function(X, method = CORR_METHOD) {
  cor(X, method = method, use = "pairwise.complete.obs")
}

top_variance_subset <- function(X, variances, max_features) {
  if (ncol(X) <= max_features) return(colnames(X))
  top_names <- names(sort(variances, decreasing = TRUE))[1:max_features]
  top_names
}

# =============================================================================
# Plot 1: Heatmap
# =============================================================================

plot_heatmap_base <- function(cor_sub, output_path) {
  png(output_path, width = 1200, height = 1200, res = 150)
  stats::heatmap(
    cor_sub, symm = TRUE,
    col = colorRampPalette(c("blue", "white", "red"))(100),
    main = "K-mer correlation heatmap (hierarchically clustered)",
    margins = c(6, 6)
  )
  dev.off()
  cat(sprintf("Saved: %s\n", output_path))
}

# =============================================================================
# Plot 2: Correlogram
# =============================================================================

plot_corrplot_ordered <- function(cor_sub, output_path, cutoff) {
  if (!requireNamespace("corrplot", quietly = TRUE)) {
    cat("[SKIPPED] 'corrplot' package not installed.\n")
    return(invisible(NULL))
  }
  n_clusters <- max(2, min(8, floor(ncol(cor_sub) / 8)))

  png(output_path, width = 1400, height = 1400, res = 150)
  corrplot::corrplot(
    cor_sub, method = "color", order = "hclust", addrect = n_clusters,
    tl.col = "black", tl.cex = 0.6, tl.srt = 90,
    title = sprintf("Ordered correlogram (hclust, %d blocks)", n_clusters),
    mar = c(0, 0, 2, 0)
  )
  dev.off()
  cat(sprintf("Saved: %s\n", output_path))
}

# =============================================================================
# Plot 3: Dendrogram
# =============================================================================

plot_dendrogram <- function(cor_sub, output_path) {
  dist_matrix <- as.dist(1 - abs(cor_sub))
  hc <- hclust(dist_matrix, method = "average")

  png(output_path, width = 1400, height = 800, res = 150)
  plot(hc, main = "K-mer feature clustering (distance = 1 - |correlation|)",
       xlab = "", sub = "", cex = 0.6)
  dev.off()
  cat(sprintf("Saved: %s\n", output_path))
  hc
}

# =============================================================================
# Plot 4: Correlation Distribution
# =============================================================================

plot_correlation_distribution <- function(cor_full, output_path, cutoff) {
  upper_vals <- cor_full[upper.tri(cor_full)]

  png(output_path, width = 1000, height = 700, res = 150)
  hist(upper_vals, breaks = 60, col = "steelblue", border = "white",
       main = "Distribution of pairwise k-mer correlations (all features)",
       xlab = "Pearson correlation")
  abline(v = cutoff, col = "red", lwd = 2, lty = 2)
  abline(v = -cutoff, col = "red", lwd = 2, lty = 2)
  legend("topright", legend = sprintf("cutoff = +/-%.2f", cutoff),
         col = "red", lty = 2, lwd = 2, bty = "n")
  dev.off()
  cat(sprintf("Saved: %s\n", output_path))
}

# =============================================================================
# Plot 5: Network Graph
# =============================================================================

plot_correlation_network <- function(cor_sub, output_path, threshold) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    cat("[SKIPPED] 'igraph' package not installed.\n")
    return(invisible(NULL))
  }

  adj <- abs(cor_sub) > threshold
  diag(adj) <- FALSE
  g <- igraph::graph_from_adjacency_matrix(adj, mode = "undirected", diag = FALSE)
  g <- igraph::delete_vertices(g, igraph::degree(g) == 0)

  if (igraph::vcount(g) == 0) {
    cat(sprintf("No feature pairs exceed threshold (%.2f) -- skipping network plot.\n", threshold))
    return(invisible(NULL))
  }

  png(output_path, width = 1200, height = 1200, res = 150)
  plot(g, vertex.size = 6, vertex.label.cex = 0.6, vertex.color = "lightblue",
       edge.color = "gray60",
       main = sprintf("K-mer correlation network (|r| > %.2f)", threshold))
  dev.off()
  cat(sprintf("Saved: %s\n", output_path))
}

# =============================================================================
# Plot 6: Dimensionality Reduction & Class Separability Plot (PCA)
# NOTE: purely descriptive -- never fed into the classifier. Fit on whatever
# row subset is passed in (see PLOTS_USE_TRAIN_ONLY in main()). Do not treat
# this plot as evidence of held-out generalization if it was fit on pooled
# data; it describes feature-space structure, not test-time performance.
# =============================================================================

plot_pca_separability <- function(X, genera, output_path) {
  pca_res <- prcomp(X, center = TRUE, scale. = TRUE)

  var_explained <- (pca_res$sdev)^2 / sum((pca_res$sdev)^2)
  pc1_var <- round(var_explained[1] * 100, 2)
  pc2_var <- round(var_explained[2] * 100, 2)

  pca_df <- data.frame(
    PC1 = pca_res$x[, 1],
    PC2 = pca_res$x[, 2],
    Genus = as.factor(genera)
  )

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    library(ggplot2)
    p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Genus, fill = Genus)) +
      geom_point(alpha = 0.7, size = 2) +
      stat_ellipse(geom = "polygon", alpha = 0.15, level = 0.95) +
      theme_minimal(base_size = 14) +
      labs(
        title = "PCA of 6-mer Profiles: Class Separability",
        subtitle = "Visualizing genus separation across bacterial 16S sequences",
        x = sprintf("PC1 (%.2f%% variance)", pc1_var),
        y = sprintf("PC2 (%.2f%% variance)", pc2_var)
      ) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "right"
      )

    ggsave(output_path, plot = p, width = 9, height = 7, dpi = 150)
  } else {
    png(output_path, width = 1200, height = 1000, res = 150)
    palette <- rainbow(length(unique(pca_df$Genus)))
    plot(
      pca_df$PC1, pca_df$PC2,
      col = palette[pca_df$Genus],
      pch = 19, cex = 0.8,
      main = "PCA of 6-mer Profiles (Class Separability)",
      xlab = sprintf("PC1 (%.2f%% variance)", pc1_var),
      ylab = sprintf("PC2 (%.2f%% variance)", pc2_var)
    )
    legend("topright", legend = levels(pca_df$Genus), col = palette, pch = 19, bty = "n")
    dev.off()
  }

  cat(sprintf("Saved: %s (PC1: %.2f%%, PC2: %.2f%% variance explained)\n",
              output_path, pc1_var, pc2_var))
}

# =============================================================================
# Feature reduction
# =============================================================================

find_correlated_features_to_drop <- function(cor_matrix, cutoff) {
  abs_cor <- abs(cor_matrix)
  diag(abs_cor) <- 0
  mean_cor <- rowMeans(abs_cor)
  feature_names <- colnames(abs_cor)

  idx <- which(upper.tri(abs_cor) & abs_cor > cutoff, arr.ind = TRUE)
  if (nrow(idx) == 0) {
    return(list(to_drop = character(0), pairs = data.frame()))
  }

  pair_cor <- abs_cor[idx]
  ord <- order(-pair_cor)
  idx <- idx[ord, , drop = FALSE]
  pair_cor <- pair_cor[ord]

  to_drop <- character(0)
  drop_log <- list()

  for (r in seq_len(nrow(idx))) {
    f1 <- feature_names[idx[r, 1]]
    f2 <- feature_names[idx[r, 2]]
    if (f1 %in% to_drop || f2 %in% to_drop) next

    dropped <- if (mean_cor[f1] >= mean_cor[f2]) f1 else f2
    kept <- if (dropped == f1) f2 else f1

    to_drop <- c(to_drop, dropped)
    drop_log[[length(drop_log) + 1]] <- data.frame(
      dropped_feature = dropped, kept_feature = kept,
      correlation = pair_cor[r], stringsAsFactors = FALSE
    )
  }

  list(to_drop = unique(to_drop), pairs = do.call(rbind, drop_log))
}

# =============================================================================
# Main
# =============================================================================

main <- function() {
  loaded <- load_kmer_features(KMER_CSV, TRAIN_MANIFEST_CSV)
  df <- loaded$df
  X <- loaded$X
  variances <- loaded$variances
  fit_rows <- loaded$fit_rows        # train-only row indices (or all rows, if no manifest)
  use_manifest <- loaded$use_manifest

  # --- THE LEAKAGE FIX: fit the correlation matrix on fit_rows (train-only) ---
  cat(sprintf("\nComputing %s correlation matrix on %d k-mer features, using %d %s rows...\n",
              CORR_METHOD, ncol(X), length(fit_rows), if (use_manifest) "TRAIN-only" else "ALL (pooled)"))
  cor_fit <- compute_correlation_matrix(X[fit_rows, , drop = FALSE], CORR_METHOD)

  # --- Visualization row subset: full dataset by default, or train-only if requested ---
  plot_rows <- if (PLOTS_USE_TRAIN_ONLY) fit_rows else seq_len(nrow(X))
  cor_for_plots <- if (PLOTS_USE_TRAIN_ONLY) cor_fit else compute_correlation_matrix(X[plot_rows, , drop = FALSE], CORR_METHOD)

  plot_features <- top_variance_subset(X[plot_rows, , drop = FALSE], variances, MAX_FEATURES_TO_PLOT)
  cor_sub <- cor_for_plots[plot_features, plot_features]
  cat(sprintf("Using top %d highest-variance k-mers for plots 1/2/3/5 (%s rows).\n",
              length(plot_features), if (PLOTS_USE_TRAIN_ONLY) "train-only" else "all"))

  # --- Plot 1: Heatmap ---
  plot_heatmap_base(cor_sub, file.path(OUTPUT_DIR, "1_correlation_heatmap.png"))

  # --- Plot 2: Ordered correlogram ---
  plot_corrplot_ordered(cor_sub, file.path(OUTPUT_DIR, "2_correlogram_ordered.png"), CORR_CUTOFF)

  # --- Plot 3: Dendrogram ---
  plot_dendrogram(cor_sub, file.path(OUTPUT_DIR, "3_feature_dendrogram.png"))

  # --- Plot 4: Distribution (uses the same fit-vs-plot row choice as above) ---
  plot_correlation_distribution(cor_for_plots, file.path(OUTPUT_DIR, "4_correlation_distribution.png"), CORR_CUTOFF)

  # --- Plot 5: Network graph ---
  plot_correlation_network(cor_sub, file.path(OUTPUT_DIR, "5_correlation_network.png"), NETWORK_CORR_THRESHOLD)

  # --- Plot 6: PCA Class Separability Plot (descriptive only; see note above plot_pca_separability) ---
  if ("genus" %in% colnames(df)) {
    plot_pca_separability(X[plot_rows, , drop = FALSE], df$genus[plot_rows],
                           file.path(OUTPUT_DIR, "6_pca_class_separability.png"))
  } else {
    cat("[SKIPPED] Column 'genus' not found in input CSV. Skipping PCA plot.\n")
  }

  # --- Feature reduction: decision made on cor_fit (train-only, if manifest present) ---
  cat(sprintf("\nFinding correlated features to drop (cutoff = %.2f, fit on %s rows)...\n",
              CORR_CUTOFF, if (use_manifest) "train-only" else "pooled"))
  reduction <- find_correlated_features_to_drop(cor_fit, CORR_CUTOFF)

  if (nrow(reduction$pairs) > 0) {
    dropped_log_path <- file.path(OUTPUT_DIR, "dropped_correlated_features.csv")
    write.csv(reduction$pairs, dropped_log_path, row.names = FALSE)
    cat(sprintf("Dropped %d of %d k-mer features (redundant, |r| > %.2f).\n",
                length(reduction$to_drop), ncol(X), CORR_CUTOFF))
    cat(sprintf("Details: %s\n", dropped_log_path))
  } else {
    cat("No feature pairs exceeded the correlation cutoff -- nothing dropped.\n")
  }

  # --- Apply the train-derived column list to ALL rows (train+val+test) ---
  kept_features <- setdiff(colnames(X), reduction$to_drop)
  output_cols <- c(META_COLS, if (use_manifest) "split" else NULL)
  reduced_df <- cbind(df[, output_cols, drop = FALSE], as.data.frame(X[, kept_features, drop = FALSE]))
  write.csv(reduced_df, REDUCED_KMER_CSV, row.names = FALSE)

  cat(sprintf("\nReduced k-mer matrix: %d -> %d features, %d rows (all splits).\n",
              ncol(X), length(kept_features), nrow(reduced_df)))
  cat(sprintf("Saved: %s\n", REDUCED_KMER_CSV))
  if (use_manifest) {
    cat("Column selection was fit on the TRAIN split only -- safe to use with the ")
    cat("matching split_manifest.csv in the downstream classifier.\n")
  }
}

main()
