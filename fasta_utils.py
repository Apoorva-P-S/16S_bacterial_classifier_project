"""
fasta_utils.py
==============
Shared FASTA-parsing utility used by all three task scripts
(taxonomy table, k-mer matrix, embeddings). Kept dependency-free
(no biopython required) so all three scripts run with just the
Python standard library + numpy/pandas.

Expected input layout: one FASTA file per genus inside a folder, e.g.

    cleaned_fastas/
        Bacillus.fasta
        Escherichia.fasta
        Lactobacillus.fasta
        Pseudomonas.fasta
        Streptococcus.fasta

Each record gets a unique sequence_id: "<genus>_<running_index>",
which is the join key across the taxonomy table, k-mer matrix, and
embeddings outputs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class FastaRecord:
    sequence_id: str   # unique join key, e.g. "Bacillus_0"
    genus: str         # taken from the filename (ground-truth label)
    header: str        # full original FASTA header (>... line, without ">")
    accession: str      # first whitespace-delimited token of the header
    sequence: str       # uppercase nucleotide sequence


def parse_fasta_file(path: Path):
    """Yield (header, sequence) tuples from a single FASTA file."""
    header, seq_chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header, seq_chunks = line[1:], []
            else:
                seq_chunks.append(line)
    if header is not None:
        yield header, "".join(seq_chunks)


def load_genus_fasta_folder(input_dir) -> List[FastaRecord]:
    """
    Load every .fasta/.fa file in input_dir. The genus label for each
    sequence is taken from the filename (e.g. Bacillus.fasta -> genus
    'Bacillus'), NOT parsed from the header -- this matches the layout
    used in the cleaning script from earlier in this pipeline, where
    one cleaned file per genus is produced.
    """
    input_dir = Path(input_dir)
    fasta_paths = sorted(list(input_dir.glob("*.fasta")) + list(input_dir.glob("*.fa")))
    if not fasta_paths:
        raise FileNotFoundError(f"No .fasta/.fa files found in {input_dir}")

    records = []
    for fasta_path in fasta_paths:
        genus = fasta_path.stem
        for i, (header, seq) in enumerate(parse_fasta_file(fasta_path)):
            accession = header.split()[0] if header else f"seq{i}"
            records.append(FastaRecord(
                sequence_id=f"{genus}_{i}",
                genus=genus,
                header=header,
                accession=accession,
                sequence=seq.upper(),
            ))
    return records
