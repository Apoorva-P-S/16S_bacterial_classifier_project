# #!/usr/bin/env python3
# """
# 1_build_taxonomy_table.py
# ==========================
# Builds a taxonomy table from a folder of per-genus FASTA files.
# This table is the label/metadata source that the k-mer matrix and
# embeddings outputs (from scripts 2 and 3) join against on
# `sequence_id`.
#
# Usage:
#     python 1_build_taxonomy_table.py \\
#         --input_dir  ./cleaned_fasta \\
#         --output_csv ./taxonomy_table.csv
#
# Output columns:
#     sequence_id   - unique join key, e.g. "Bacillus_0"
#     genus         - ground-truth genus label (your ML target)
#     accession     - first token of the FASTA header (usually the accession)
#     header        - full original FASTA header text
#     sequence_length
# """


#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
from fasta_utils import load_genus_fasta_folder

def build_taxonomy_table(input_dir):
    records = load_genus_fasta_folder(input_dir)
    df = pd.DataFrame({
        "sequence_id": r.sequence_id,
        "genus": r.genus,
        "accession": r.accession,
        "header": r.header,
        "sequence_length": len(r.sequence),
    } for r in records)
    return df

def main():
    parser = argparse.ArgumentParser(description="Build a taxonomy table from per-genus FASTA files.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    df = build_taxonomy_table(args.input_dir)

    if df.empty:
        print(f"[ERROR] No sequence records loaded from {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if df["sequence_id"].duplicated().any():
        print("[WARN] Duplicate sequence_ids detected in table!", file=sys.stderr)

    # Ensure output parent directory exists
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print("\nSequences per genus:")
    print(df["genus"].value_counts())

if __name__ == "__main__":
    main()