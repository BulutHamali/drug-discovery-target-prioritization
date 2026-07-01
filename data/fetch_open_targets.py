#!/usr/bin/env python3
"""Download and cache the Open Targets targets table (Parquet).

Open Targets is used for the label only (via fetch_chembl_known_drugs.py,
which pulls knownDrugsAggregated). This script pulls only the targets table,
which provides gene metadata (Ensembl ID, symbol, biotype) needed to key
the feature matrix and the label table against each other.

associationByOverallDirect is deliberately not pulled: it is a bundled
overall score with no datatype breakdown, so filtering out genetic evidence
is not possible. Using it as a feature would reintroduce the circularity
described in section 4 of DESIGN.md.

Idempotent: each Parquet partition is skipped if already present locally.
Re-run freely.

Usage:
    python3 data/fetch_open_targets.py [--release 24.12] [--dest data/cache]

Set DATA_CACHE_DIR env var as an alternative to --dest.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Latest stable release as of project start. Pin this; schema shifts between
# releases and the ML layer is built against a specific field set.
DEFAULT_RELEASE = "24.12"

FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"

# Parquet datasets to pull (relative to the release root).
DATASETS = [
    "output/etl/parquet/targets",
]


def ftp_list(url: str) -> list[str]:
    """Return filenames listed at an FTP-over-HTTP directory URL."""
    import html.parser

    class LinkParser(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.links: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for name, val in attrs:
                    if name == "href" and val and not val.startswith("?") and val != "../":
                        self.links.append(val.rstrip("/"))

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


def fetch_dataset(release_url: str, dataset_path: str, dest_root: Path) -> None:
    dataset_url = f"{release_url}/{dataset_path}"
    dest_dir = dest_root / dataset_path.replace("output/etl/parquet/", "")

    print(f"\n=== {dataset_path} ===")
    try:
        files = ftp_list(dataset_url + "/")
    except Exception as e:
        print(f"  ERROR listing {dataset_url}: {e}", file=sys.stderr)
        sys.exit(1)

    parquet_files = [f for f in files if f.endswith(".parquet") or f.startswith("part-")]
    if not parquet_files:
        # Partitioned directory — list one level deeper
        for part in files:
            if part.startswith("part") or "=" in part:
                sub_url = f"{dataset_url}/{part}"
                sub_dest = dest_dir / part
                try:
                    sub_files = ftp_list(sub_url + "/")
                    for sf in sub_files:
                        if sf.endswith(".parquet") or sf.startswith("part-"):
                            download_file(f"{sub_url}/{sf}", sub_dest / sf)
                except Exception:
                    download_file(f"{dataset_url}/{part}", dest_dir / part)
    else:
        for fname in parquet_files:
            download_file(f"{dataset_url}/{fname}", dest_dir / fname)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default=os.environ.get("OT_RELEASE", DEFAULT_RELEASE))
    parser.add_argument("--dest", default=os.environ.get("DATA_CACHE_DIR", Path(__file__).parent / "cache"))
    args = parser.parse_args()

    release_url = f"{FTP_BASE}/{args.release}"
    dest_root = Path(args.dest) / "open_targets" / args.release

    print(f"Open Targets release: {args.release}")
    print(f"Destination:          {dest_root}")

    for dataset in DATASETS:
        fetch_dataset(release_url, dataset, dest_root)

    print("\nOpen Targets targets table ready.")


if __name__ == "__main__":
    main()
