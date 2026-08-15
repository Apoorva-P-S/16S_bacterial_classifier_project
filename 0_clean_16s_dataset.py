#!/usr/bin/env python3
"""
clean_16s_dataset.py
=====================

Quality-filtering and cleaning pipeline for 16S rRNA reference FASTA files
(e.g. sequences pulled from SILVA), one FASTA file per genus.

Steps performed per genus FASTA file, in order:
  1. Ambiguous-taxonomy-label filtering  (header-text based)
  2. Ambiguous-base / length filtering    (pure Python)
  3. Exact-duplicate removal              (seqkit rmdup, falls back to Python)
  4. Nested/contained-sequence removal    (pure Python, containment check)
  5. Chimera removal                      (vsearch --uchime_denovo)

NOTE ON DADA2 vs VSEARCH FOR CHIMERA REMOVAL
---------------------------------------------
DADA2's `removeBimeraDenovo()` (R) is designed to work on an ASV table built
from real amplicon sequencing reads, where each sequence has an abundance
(read count) attached -- that's what its bimera model needs. Reference
database FASTA sequences (like the ones you pull from SILVA) have no
abundance information, so DADA2 isn't the natural tool for this specific
step. The standard approach used to clean 16S reference databases
(e.g. by SILVA/RDP curators) is `vsearch --uchime_denovo`, which needs
only the sequences themselves. This script uses vsearch for that reason.

If you specifically need DADA2's bimera removal (e.g. because you're
cleaning an ASV table with abundances, not a reference FASTA), see the
`run_chimera_removal_dada2_rscript()` stub near the bottom -- it shells
out to an R script and expects a 2-column CSV (sequence, abundance).

Requirements
------------
  - Python 3.8+
  - biopython   (pip install biopython)
  - seqkit      (https://github.com/shenwei356/seqkit)   -- optional but recommended
  - vsearch     (https://github.com/torognes/vsearch)     -- required for chimera step

Usage
-----
  python 0_clean_16s_dataset.py \\
      --input_dir  ./data_fasta_small\\
      --output_dir ./cleaned_fasta \\
      --min_len 800 --max_len 1600 \\
      --max_n_frac 0.01

Input layout expected: one FASTA file per genus inside --input_dir,
e.g.:
    raw_fastas/
        Bacillus.fasta
        Lactobacillus.fasta
        Pseudomonas.fasta
        Streptococcus.fasta
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

# --------------------------------------------------------------------------
# Config: terms that mark a taxonomy label as too ambiguous to train on
# --------------------------------------------------------------------------
AMBIGUOUS_LABEL_TERMS = [
    "uncultured",
    "unidentified",
    "unclassified",
    "environmental sample",
    "metagenome",
    "sp.",
    "candidatus",
    "incertae sedis",
    "unknown",
]

VALID_BASES = set("ACGU")


# --------------------------------------------------------------------------
# Dependency checks
# --------------------------------------------------------------------------
def check_dependency(tool_name, required=True):
    path = shutil.which(tool_name)
    if path is None:
        msg = f"[WARN] '{tool_name}' not found on PATH."
        if required:
            msg = f"[ERROR] '{tool_name}' is required but not found on PATH. Install it and retry."
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
    return path


# --------------------------------------------------------------------------
# Step 1: ambiguous taxonomy label filtering
# --------------------------------------------------------------------------
def is_ambiguous_label(header, ambiguous_terms=AMBIGUOUS_LABEL_TERMS):
    """Return True if the FASTA header/description looks like an
    ambiguous / low-resolution taxonomy label that should be dropped."""
    header_lower = header.lower()
    return any(term in header_lower for term in ambiguous_terms)


# --------------------------------------------------------------------------
# Step 2: ambiguous base / length filtering
# --------------------------------------------------------------------------
def sequence_passes_quality(seq_str, min_len=1000, max_len=1200, max_n_frac=0.01):
    """Check length bounds and fraction of non-ACGT ('ambiguous') bases."""
    seq_str = seq_str.upper()
    length = len(seq_str)
    if length < min_len or length > max_len:
        return False, "length_out_of_range"

    non_acgu = sum(1 for b in seq_str if b not in VALID_BASES)
    if length == 0:
        return False, "empty_sequence"
    if (non_acgu / length) > max_n_frac:
        return False, "too_many_ambiguous_bases"

    return True, "pass"


def label_and_quality_filter(input_fasta, output_fasta, min_len, max_len, max_n_frac):
    """Step 1 + Step 2 combined: write only records that pass both the
    label check and the base/length check. Returns a stats dict."""
    stats = {
        "input_total": 0,
        "dropped_ambiguous_label": 0,
        "dropped_length_out_of_range": 0,
        "dropped_too_many_ambiguous_bases": 0,
        "dropped_empty_sequence": 0,
        "kept_after_label_and_quality": 0,
    }

    kept_records = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        stats["input_total"] += 1

        if is_ambiguous_label(record.description):
            stats["dropped_ambiguous_label"] += 1
            continue

        ok, reason = sequence_passes_quality(str(record.seq), min_len, max_len, max_n_frac)
        if not ok:
            stats[f"dropped_{reason}"] = stats.get(f"dropped_{reason}", 0) + 1
            continue

        kept_records.append(record)

    stats["kept_after_label_and_quality"] = len(kept_records)
    SeqIO.write(kept_records, output_fasta, "fasta")
    return stats


# --------------------------------------------------------------------------
# Step 3: exact-duplicate removal (seqkit, with pure-Python fallback)
# --------------------------------------------------------------------------
def dedup_with_seqkit(input_fasta, output_fasta):
    """Use `seqkit rmdup -s` to remove exact-sequence duplicates
    (identical sequence content, regardless of header)."""
    cmd = ["seqkit", "rmdup", "-s", "-o", str(output_fasta), str(input_fasta)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"seqkit rmdup failed:\n{result.stderr}")
    return result.stderr  # seqkit prints a short summary to stderr


def dedup_with_python(input_fasta, output_fasta):
    """Fallback exact-duplicate removal if seqkit isn't installed."""
    seen_seqs = set()
    kept = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        seq_str = str(record.seq).upper()
        if seq_str in seen_seqs:
            continue
        seen_seqs.add(seq_str)
        kept.append(record)
    SeqIO.write(kept, output_fasta, "fasta")
    return len(kept)


def deduplicate(input_fasta, output_fasta, use_seqkit=True):
    n_before = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    if use_seqkit and shutil.which("seqkit"):
        dedup_with_seqkit(input_fasta, output_fasta)
    else:
        dedup_with_python(input_fasta, output_fasta)
    n_after = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
    return {"input_before_dedup": n_before, "kept_after_dedup": n_after,
             "dropped_exact_duplicates": n_before - n_after}


# --------------------------------------------------------------------------
# Step 4: nested / contained sequence removal
# --------------------------------------------------------------------------
def remove_nested_sequences(input_fasta, output_fasta):
    """
    Remove sequences that are fully contained (100% exact substring, either
    strand) within a longer, already-kept sequence. This is the classic
    'nested sequence' problem in reference databases like SILVA where a
    short partial-16S entry is a literal substring of a longer full-length
    entry from the same lineage.

    Approach: sort by length descending, keep the first (longest) copy of
    each, and check each subsequent (shorter) sequence for containment
    against sequences already kept. This is O(n^2) in the worst case --
    fine for a few thousand sequences per genus; for very large reference
    sets (>20k per genus) consider `cd-hit-est -c 1.0 -aS 1.0` instead,
    which does the same job much faster using k-mer indexing.
    """
    records = list(SeqIO.parse(input_fasta, "fasta"))
    records.sort(key=lambda r: len(r.seq), reverse=True)

    kept = []
    kept_seqs_fwd = []
    kept_seqs_rev = []
    n_dropped_nested = 0

    for record in records:
        seq_str = str(record.seq).upper()
        rc_str = str(Seq(seq_str).reverse_complement())

        is_nested = any(seq_str in kept_seq for kept_seq in kept_seqs_fwd) or \
                    any(rc_str in kept_seq for kept_seq in kept_seqs_fwd)

        if is_nested:
            n_dropped_nested += 1
            continue

        kept.append(record)
        kept_seqs_fwd.append(seq_str)

    SeqIO.write(kept, output_fasta, "fasta")
    return {"kept_after_nested_removal": len(kept), "dropped_nested_sequences": n_dropped_nested}


# --------------------------------------------------------------------------
# Step 5: chimera removal (vsearch --uchime_denovo)
#Not necessary here as SILVA already has its own curation
# --------------------------------------------------------------------------
def run_chimera_removal_vsearch(input_fasta, output_fasta, chimeras_fasta=None):
    """De novo chimera detection/removal on a reference FASTA using vsearch.
    No abundance information is required for uchime_denovo."""
    if chimeras_fasta is None:
        chimeras_fasta = str(output_fasta) + ".chimeras.fasta"

    cmd = [
        "vsearch",
        "--uchime_denovo", str(input_fasta),
        "--nonchimeras", str(output_fasta),
        "--chimeras", str(chimeras_fasta),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vsearch --uchime_denovo failed:\n{result.stderr}")

    n_nonchimeric = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
    n_chimeric = sum(1 for _ in SeqIO.parse(chimeras_fasta, "fasta"))
    return {"kept_after_chimera_removal": n_nonchimeric, "dropped_chimeras": n_chimeric}
#
#
# def run_chimera_removal_dada2_rscript(asv_csv_path, output_csv_path, r_script_path="remove_bimera.R"):
#     """
#     OPTIONAL stub: use this instead of vsearch if you are cleaning an ASV
#     table (sequence + abundance) rather than a reference FASTA.
#     Requires R + the DADA2 package installed separately.
#     Writes and runs a small R script via Rscript.
#     """
#     r_code = f"""
#     library(dada2)
#     df <- read.csv("{asv_csv_path}")               # columns: sequence, abundance
#     seqtab <- matrix(df$abundance, nrow = 1, dimnames = list("sample1", df$sequence))
#     seqtab_nochim <- removeBimeraDenovo(seqtab, method = "consensus", verbose = TRUE)
#     out <- data.frame(sequence = colnames(seqtab_nochim),
#                        abundance = as.numeric(seqtab_nochim[1, ]))
#     write.csv(out, "{output_csv_path}", row.names = FALSE)
#     """
#     Path(r_script_path).write_text(r_code)
#     result = subprocess.run(["Rscript", r_script_path], capture_output=True, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"DADA2 Rscript failed:\n{result.stderr}")
#     return output_csv_path


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def process_genus_file(fasta_path, genus_name, work_dir, args):
    print(f"\n=== Processing genus: {genus_name} ===")
    stats = {"genus": genus_name}

    f1_label_qual = work_dir / f"{genus_name}.1_label_quality.fasta"
    f2_dedup = work_dir / f"{genus_name}.2_dedup.fasta"
    f3_nonnested = work_dir / f"{genus_name}.3_nonnested.fasta"
    f4_final = work_dir / f"{genus_name}.4_final_clean.fasta"

    # Step 1 + 2
    s = label_and_quality_filter(fasta_path, f1_label_qual, args.min_len, args.max_len, args.max_n_frac)
    stats.update(s)
    print(f"  label+quality filter: kept {s['kept_after_label_and_quality']} / {s['input_total']}")

    # Step 3
    s = deduplicate(f1_label_qual, f2_dedup, use_seqkit=not args.no_seqkit)
    stats.update(s)
    print(f"  dedup: kept {s['kept_after_dedup']} (dropped {s['dropped_exact_duplicates']} exact duplicates)")

    # Step 4
    s = remove_nested_sequences(f2_dedup, f3_nonnested)
    stats.update(s)
    print(f"  nested removal: kept {s['kept_after_nested_removal']} "
          f"(dropped {s['dropped_nested_sequences']} nested)")

    # Step 5
    s = run_chimera_removal_vsearch(f3_nonnested, f4_final)
    stats.update(s)
    print(f"  chimera removal: kept {s['kept_after_chimera_removal']} "
          f"(dropped {s['dropped_chimeras']} chimeras)")

    final_output = args.output_dir / f"{genus_name}.clean.fa"
    shutil.copy(f4_final, final_output)
    stats["final_output_path"] = str(final_output)
    stats["final_sequence_count"] = stats["kept_after_chimera_removal"]

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean 16S rRNA reference FASTA files, one per genus.")
    parser.add_argument("--input_dir", required = True, type=Path,
                         help="Folder containing one FASTA file per genus (e.g. Bacillus.fasta)")
    parser.add_argument("--output_dir", required = True, type=Path,
                         help="Folder to write cleaned FASTA files + summary report")
    parser.add_argument("--min_len", type=int, default=1000, help="Minimum sequence length to keep")
    parser.add_argument("--max_len", type=int, default=1200, help="Maximum sequence length to keep")
    parser.add_argument("--max_n_frac", type=float, default=0.01,
                         help="Max fraction of non-ACGT bases allowed (default 1%%)")
    parser.add_argument("--no_seqkit", action="store_true",
                         help="Skip seqkit and use the pure-Python dedup fallback instead")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "_intermediate"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Dependency checks
    check_dependency("vsearch", required=True)
    check_dependency("seqkit", required=False)

    fasta_files = sorted(list(args.input_dir.glob("*.fasta")) + list(args.input_dir.glob("*.fa")))
    if not fasta_files:
        print(f"[ERROR] No .fasta/.fa files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    all_stats = []
    for fasta_path in fasta_files:
        genus_name = fasta_path.stem
        stats = process_genus_file(fasta_path, genus_name, work_dir, args)
        all_stats.append(stats)

    # Write summary CSV
    summary_path = args.output_dir / "cleaning_summary.csv"
    fieldnames = sorted({k for row in all_stats for k in row.keys()})
    # Put a sensible column order up front
    preferred_order = ["genus", "input_total", "dropped_ambiguous_label",
                        "dropped_length_out_of_range", "dropped_too_many_ambiguous_bases",
                        "kept_after_label_and_quality", "dropped_exact_duplicates",
                        "kept_after_dedup", "dropped_nested_sequences",
                        "kept_after_nested_removal", "dropped_chimeras",
                        "final_sequence_count", "final_output_path"]
    ordered_fields = [f for f in preferred_order if f in fieldnames] + \
                      [f for f in fieldnames if f not in preferred_order]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(all_stats)

    print(f"\n=== Done. Cleaned FASTA files + summary written to: {args.output_dir} ===")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()