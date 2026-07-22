#!/usr/bin/env python3
"""
Pre-flight health check for every remote source this project fetches from.

Pings one known-good, real identifier per source (never a made-up ID) so a
404 means the endpoint itself is broken, not that the test ID happens to be
missing. Uses HEAD requests where the source supports them, to avoid
downloading multi-hundred-MB files just to check reachability.

This exists because a dead endpoint used to be discovered ~90 minutes into a
~20,000-request disorder_fraction fetch (see ml/fetch_alphafold.py's history
and DESIGN.md section 13). Running this first finds the same problem in
seconds.

Run:  python3 ml/check_endpoints.py
Exit: 0 if every source is reachable, 1 if any source failed.
"""

import sys
import urllib.error
import urllib.request

TIMEOUT = 15

# Known-good real identifiers, not placeholders: TP53 (P04637) for
# UniProt/AlphaFold, a real PubMed ID for esummary, and small, stable
# directory/file paths for the bulk sources.
CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _request(url, method="GET", headers=None, max_bytes=0):
    """
    Return (status, body_bytes). Reads up to max_bytes of the body INSIDE
    the urlopen `with` block, since the underlying connection closes the
    moment that block exits; returning the live response object after the
    `with` block ends (an earlier version of this script did that) yields a
    closed stream that silently reads back empty, not an error.
    """
    req_headers = {"User-Agent": "drug-discovery-pipeline/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, method=method, headers=req_headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        status = r.status
        body = r.read(max_bytes) if max_bytes else b""
    return status, body


@register("UniProt Swiss-Prot (fetch_alphafold.py)")
def check_uniprot():
    status, body = _request("https://rest.uniprot.org/uniprotkb/P04637.fasta", max_bytes=200)
    if not body.startswith(b">"):
        raise RuntimeError(f"status {status} but body is not FASTA: {body[:60]!r}")
    return f"HTTP {status}, test ID P04637 (TP53)"


@register("AlphaFold DB prediction API (fetch_alphafold.py --disorder)")
def check_alphafold():
    status, body = _request("https://alphafold.ebi.ac.uk/api/prediction/P04637", max_bytes=65536)
    import json
    payload = json.loads(body)
    if not (isinstance(payload, list) and payload and "fractionPlddtVeryLow" in payload[0]):
        raise RuntimeError(f"status {status} but response missing fractionPlddtVeryLow")
    return f"HTTP {status}, test ID P04637 (TP53)"


@register("gnomAD v2.1.1 constraint (fetch_gnomad.py)")
def check_gnomad():
    status, _ = _request(
        "https://gnomad-public-us-east-1.s3.amazonaws.com/release/2.1.1/"
        "constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz",
        method="HEAD",
    )
    return f"HTTP {status} (HEAD)"


@register("STRING v12.0 protein.info (fetch_string.py)")
def check_string_info():
    status, _ = _request(
        "https://stringdb-downloads.org/download/"
        "protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        method="HEAD",
    )
    return f"HTTP {status} (HEAD)"


@register("STRING v12.0 protein.links (fetch_string.py)")
def check_string_links():
    status, _ = _request(
        "https://stringdb-downloads.org/download/"
        "protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        method="HEAD",
    )
    return f"HTTP {status} (HEAD)"


@register("GTEx v8 median gene TPM (fetch_expression.py)")
def check_gtex():
    status, _ = _request(
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
        "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
        method="HEAD",
    )
    return f"HTTP {status} (HEAD)"


@register("DepMap 24Q4 CRISPR gene effect (fetch_expression.py)")
def check_depmap():
    # figshare's ndownloader redirects to a signed S3 URL that does not
    # reliably support HEAD; a 1-byte ranged GET confirms reachability
    # through the full redirect chain without downloading the ~429 MB file.
    status, _ = _request(
        "https://ndownloader.figshare.com/files/51064667",
        headers={"Range": "bytes=0-0"},
    )
    if status not in (200, 206):
        raise RuntimeError(f"unexpected status {status}")
    return f"HTTP {status} (ranged GET)"


@register("HGNC complete set (gene_families.py, fetch_publications.py)")
def check_hgnc():
    status, _ = _request(
        "https://storage.googleapis.com/public-download-files/hgnc/"
        "tsv/tsv/hgnc_complete_set.txt",
        method="HEAD",
    )
    return f"HTTP {status} (HEAD)"


@register("NCBI gene2pubmed (fetch_publications.py)")
def check_gene2pubmed():
    status, _ = _request("https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz", method="HEAD")
    return f"HTTP {status} (HEAD)"


@register("NCBI esummary (fetch_publications.py)")
def check_esummary():
    # PMID 31452929: a real, stable PubMed ID.
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        "?db=pubmed&id=31452929&retmode=json"
    )
    status, body = _request(url, max_bytes=16384)
    import json
    data = json.loads(body)
    if "31452929" not in data.get("result", {}).get("uids", []):
        raise RuntimeError(f"status {status} but response missing the test PMID")
    return f"HTTP {status}, test PMID 31452929"


@register("Open Targets platform, targets + knownDrugsAggregated (data/fetch_open_targets.py, data/fetch_chembl_known_drugs.py)")
def check_open_targets():
    status, _ = _request(
        "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.09/"
        "output/etl/parquet/targets/"
    )
    return f"HTTP {status}, release 24.09"


@register("1000 Genomes phase 3, chr22 (data/fetch_1000genomes.sh)")
def check_1000genomes():
    status, _ = _request(
        "https://1000genomes.s3.amazonaws.com/release/20130502/"
        "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz",
        method="HEAD",
    )
    return f"HTTP {status} (HEAD), chr22"


@register("Ensembl GRCh37 r87 FASTA, chr22 (data/fetch_ref.sh)")
def check_ensembl():
    status, _ = _request(
        "http://ftp.ensembl.org/pub/grch37/release-87/fasta/homo_sapiens/dna/"
        "Homo_sapiens.GRCh37.dna.chromosome.22.fa.gz",
        method="HEAD",
    )
    return f"HTTP {status} (HEAD), chr22"


def main():
    results = []
    for name, fn in CHECKS:
        try:
            detail = fn()
            results.append((name, True, detail))
        except urllib.error.HTTPError as exc:
            results.append((name, False, f"HTTP {exc.code}: {exc.reason}"))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    name_width = max(len(n) for n, _, _ in results)
    print(f"{'SOURCE':<{name_width}}  STATUS   DETAIL")
    print("-" * (name_width + 60))
    n_failed = 0
    for name, ok, detail in results:
        status_str = "OK" if ok else "FAILED"
        if not ok:
            n_failed += 1
        print(f"{name:<{name_width}}  {status_str:<7}  {detail}")

    print()
    if n_failed:
        print(f"{n_failed} / {len(results)} sources FAILED. Fix these before running a fetch "
              f"that depends on them.")
        sys.exit(1)
    else:
        print(f"All {len(results)} sources OK.")
        sys.exit(0)


if __name__ == "__main__":
    main()
