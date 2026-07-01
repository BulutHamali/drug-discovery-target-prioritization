#!/usr/bin/env python3
"""
Parse a bcftools csq-annotated VCF and produce per-gene variant burden counts.

Output columns:
    gene        gene symbol (from BCSQ INFO tag)
    chrom       chromosome
    n_rare      variants with AF < af_max that overlap this gene
    n_lof       rare variants with a loss-of-function consequence

These are raw counts for the input sample set. The ML layer combines them
with gnomAD constraint scores (pLI, LOEUF) which provide population-level
o/e metrics the pipeline cannot compute without population-level AN data.

bcftools csq consequence terms used for LoF classification:
    stop_gained, frameshift, splice_donor, splice_acceptor, start_lost
"""

import argparse
import gzip
import sys
from collections import defaultdict

LOF = {"stop_gained", "frameshift", "splice_donor", "splice_acceptor", "start_lost"}


def parse_info(raw):
    out = {}
    for item in raw.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
        else:
            out[item] = True
    return out


def iter_bcsq(raw):
    """
    Yield (consequence, gene_symbol) from a BCSQ INFO value.
    Entries starting with '@' are back-references to an upstream variant's
    consequence; they are skipped because the originating position already
    counted them.
    """
    for entry in raw.split(","):
        if entry.startswith("@"):
            continue
        parts = entry.split("|")
        if len(parts) >= 2:
            yield parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf",    required=True, help="bcftools csq-annotated VCF (gz or plain)")
    ap.add_argument("--af-max", type=float, default=0.01, help="rare-variant AF ceiling (default 0.01)")
    ap.add_argument("--chrom",  required=True, help="chromosome label for output column")
    ap.add_argument("--out",    required=True, help="output TSV path")
    args = ap.parse_args()

    rare = defaultdict(int)
    lof  = defaultdict(int)

    opener = gzip.open if args.vcf.endswith(".gz") else open
    n_variants = 0
    n_rare_variants = 0

    with opener(args.vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t", 8)
            if len(cols) < 8:
                continue
            n_variants += 1

            info = parse_info(cols[7])

            af_raw = info.get("AF", "")
            if not af_raw:
                continue
            try:
                af = float(af_raw.split(",")[0])
            except ValueError:
                continue

            if af >= args.af_max:
                continue
            n_rare_variants += 1

            bcsq_raw = info.get("BCSQ", "")
            if not bcsq_raw:
                continue

            genes_rare = set()
            genes_lof  = set()
            for csq, gene in iter_bcsq(bcsq_raw):
                if gene:
                    genes_rare.add(gene)
                    if csq in LOF:
                        genes_lof.add(gene)

            for g in genes_rare:
                rare[g] += 1
            for g in genes_lof:
                lof[g] += 1

    all_genes = sorted(rare.keys() | lof.keys())

    with open(args.out, "w") as fh:
        fh.write("gene\tchrom\tn_rare\tn_lof\n")
        for gene in all_genes:
            fh.write(f"{gene}\t{args.chrom}\t{rare[gene]}\t{lof[gene]}\n")

    print(
        f"burden: {n_variants} variants, {n_rare_variants} rare, "
        f"{len(all_genes)} genes -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
