"""Download FlyWire v783 connectivity data from Zenodo and verify checksums.

Source: FlyWire Consortium, "FlyWire Whole-brain Connectome Connectivity Data",
Zenodo, doi:10.5281/zenodo.10676866 (CC-BY 4.0).

Usage:
    python scripts/download_data.py              # downloads into data/raw/
    python scripts/download_data.py --data-dir /some/other/path
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "10676866"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# filename -> expected MD5 (from the Zenodo record page)
FILES = {
    "proofread_root_ids_783.npy": "e0e6c19732fd8c7a4e39a2d170105421",
    "proofread_connections_783.feather": "f48f972d262323a102aed49af1396b8a",
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


def fetch(name: str, expected_md5: str, data_dir: Path) -> None:
    dest = data_dir / name
    if dest.exists():
        if md5sum(dest) == expected_md5:
            print(f"[skip] {name} already present, checksum OK")
            return
        print(f"[redo] {name} exists but checksum mismatch, re-downloading")

    download(f"{BASE_URL}/{name}?download=1", dest)

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

    for name, md5 in FILES.items():
        fetch(name, md5, args.data_dir)
    print(f"\nAll files ready in {args.data_dir}")


if __name__ == "__main__":
    main()
