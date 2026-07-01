#!/usr/bin/env python3
"""
Merge per-chromosome burden TSVs into a single Parquet feature table.

Input:  one or more TSV files from the BURDEN process (gene/chrom/n_rare/n_lof)
Output: gene_burden_features.parquet

Schema:
    gene     string   gene symbol (join key for the ML feature matrix)
    n_rare   int64    total rare variant count across all input chromosomes
    n_lof    int64    total rare LoF variant count
    chroms   string   comma-separated list of chromosomes contributing data
"""

import argparse
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output Parquet path")
    ap.add_argument("inputs", nargs="+", help="burden TSV files")
    args = ap.parse_args()

    records = defaultdict(lambda: {"n_rare": 0, "n_lof": 0, "chroms": set()})

    for path in args.inputs:
        with open(path) as fh:
            fh.readline()  # header
            for line in fh:
                gene, chrom, n_rare, n_lof = line.rstrip("\n").split("\t")
                records[gene]["n_rare"]  += int(n_rare)
                records[gene]["n_lof"]   += int(n_lof)
                records[gene]["chroms"].add(chrom)

    genes = sorted(records)
    table = pa.table(
        {
            "gene":   pa.array(genes, type=pa.string()),
            "n_rare": pa.array([records[g]["n_rare"] for g in genes], type=pa.int64()),
            "n_lof":  pa.array([records[g]["n_lof"]  for g in genes], type=pa.int64()),
            "chroms": pa.array(
                [",".join(sorted(records[g]["chroms"])) for g in genes],
                type=pa.string(),
            ),
        }
    )

    pq.write_table(table, args.out, compression="snappy")
    print(f"collect: {len(genes)} genes -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
