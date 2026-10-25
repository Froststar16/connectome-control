"""Download FlyWire v783 data and verify checksums.

Sources (both CC-BY 4.0):
  * Connectivity: FlyWire Consortium, Zenodo record 10676866 (doi:10.5281/zenodo.10676866).
  * Neuron annotations: flyconnectome/flywire_annotations on GitHub, release v3.2.0
    (Schlegel et al. 2024, Dorkenwald et al. 2024, Berg et al. 2026, Tastekin et al. 2026,
    Matsliah et al. 2024). Pinned to a release tag so the file never changes under us.

Usage:
    python scripts/download_data.py              # downloads everything into data/raw/
    python scripts/download_data.py --data-dir /some/other/path
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO = "https://zenodo.org/records/10676866/files"
ANNOTATIONS_TAG = "v3.2.0"
GITHUB_RAW = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    f"{ANNOTATIONS_TAG}/supplemental_files"
)

# local filename -> (download URL, expected MD5).
# Zenodo MD5s come from the record page; the annotation MD5 was computed from the v3.2.0 tag.
FILES = {
    "proofread_root_ids_783.npy": (
        f"{ZENODO}/proofread_root_ids_783.npy?download=1",
        "e0e6c19732fd8c7a4e39a2d170105421",
    ),
    "proofread_connections_783.feather": (
        f"{ZENODO}/proofread_connections_783.feather?download=1",
        "f48f972d262323a102aed49af1396b8a",
    ),
    f"neuron_annotations_{ANNOTATIONS_TAG}.tsv": (
        f"{GITHUB_RAW}/Supplemental_file1_neuron_annotations.tsv",
        "fd4fcb205256ca2f80a2eca3a0cc5d87",
    ),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20  # 1 MiB


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream a file to disk via a temporary .part file, so an interrupted
    download never leaves a half-written file under the final name."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for block in r.iter_content(chunk_size=CHUNK):
                f.write(block)
                bar.update(len(block))
    tmp.replace(dest)


def fetch(name: str, url: str, expected_md5: str, data_dir: Path) -> None:
    dest = data_dir / name
    if dest.exists():
        if md5sum(dest) == expected_md5:
            print(f"[skip] {name} already present, checksum OK")
            return
        print(f"[redo] {name} exists but checksum mismatch, re-downloading")

    download(url, dest)

    actual = md5sum(dest)
    if actual != expected_md5:
        dest.unlink()
        sys.exit(f"[fail] {name}: expected MD5 {expected_md5}, got {actual}. File removed.")
    print(f"[ok]   {name} downloaded, checksum verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data" / "raw",
        help="where to save files (default: data/raw/)",
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for name, (url, md5) in FILES.items():
        fetch(name, url, md5, args.data_dir)
    print(f"\nAll files ready in {args.data_dir}")


if __name__ == "__main__":
    main()
