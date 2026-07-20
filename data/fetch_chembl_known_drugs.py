#!/usr/bin/env python3
"""Download and cache the Open Targets known-drug evidence (Parquet).

This is the label source for the ML model: does a target have a drug at
clinical phase >= 1 (or approved)?

We pull from Open Targets rather than ChEMBL directly because:
  1. The known-drug evidence table is already gene-ID-keyed (Ensembl IDs),
     so no ChEMBL-to-Ensembl ID mapping step is needed.
  2. It uses the same schema version as the association evidence pulled by
     fetch_open_targets.py, eliminating cross-source ID drift.
  3. It includes max clinical phase per target, supporting a continuous
     regression label as well as the binary >= phase 1 label.

Fields we use (current OT schema, 24.x):
  targetId          Ensembl gene ID (ENSG...)
  diseaseId         EFO ID
  clinicalPhase     max clinical phase for this target-disease pair
  drugId            ChEMBL compound ID
  mechanismOfAction drug mechanism

Idempotent: each Parquet partition is skipped if already present.

Usage:
    python3 data/fetch_chembl_known_drugs.py [--release 24.09] [--dest data/cache]
"""

import argparse
import html.parser
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_RELEASE = "24.09"
FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
DATASET_PATH = "output/etl/parquet/knownDrugsAggregated"


class LinkParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, val in attrs:
                if name == "href" and val and not val.startswith("?") and val != "../":
                    self.links.append(val.rstrip("/"))


def ftp_list(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "drug-discovery-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        parser = LinkParser()
        parser.feed(resp.read().decode())
        return parser.links


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest}")
        return
    print(f"  downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "drug-discovery-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default=os.environ.get("OT_RELEASE", DEFAULT_RELEASE))
    parser.add_argument("--dest", default=os.environ.get("DATA_CACHE_DIR", Path(__file__).parent / "cache"))
    args = parser.parse_args()

    release_url = f"{FTP_BASE}/{args.release}"
    dataset_url = f"{release_url}/{DATASET_PATH}"
    dest_dir = Path(args.dest) / "open_targets" / args.release / "knownDrugsAggregated"

    print(f"Open Targets release: {args.release}")
    print(f"Dataset:              knownDrugsAggregated")
    print(f"Destination:          {dest_dir}")

    try:
        files = ftp_list(dataset_url + "/")
    except Exception as e:
        print(f"ERROR listing {dataset_url}: {e}", file=sys.stderr)
        sys.exit(1)

    parquet_files = [f for f in files if f.endswith(".parquet") or f.startswith("part-")]
    if not parquet_files:
        print(f"ERROR: no Parquet files found at {dataset_url}", file=sys.stderr)
        print(f"Files found: {files}", file=sys.stderr)
        sys.exit(1)

    for fname in parquet_files:
        download_file(f"{dataset_url}/{fname}", dest_dir / fname)

    print(f"\nknownDrugsAggregated ready in: {dest_dir}")
    print("Label column to use: clinicalPhase (binary: >= 1; or continuous regression).")


if __name__ == "__main__":
    main()
