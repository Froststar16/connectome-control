"""Download FlyWire v783 data and verify checksums.

Sources (both CC-BY 4.0):
  * Connectivity: FlyWire Consortium, Zenodo record 10676866 (doi:10.5281/zenodo.10676866).
  * Neuron annotations: flyconnectome/flywire_annotations on GitHub, release v3.2.0
    (Schlegel et al. 2024, Dorkenwald et al. 2024, Berg et al. 2026, Tastekin et al. 2026,
    Matsliah et al. 2024). Pinned to a release tag so the file never changes under us.

  * Optional (--shiu, Phase 3): the connectivity table of Shiu et al. 2024 (Nature), from their
    repository github.com/philshiu/Drosophila_brain_model, pinned to a commit (~100 MB).
  * Eye map (Phase 5, small): FlyWire right-optic-lobe column assignments (Matsliah et al. 2024,
    Nature) and ommatidium viewing directions (Zhao et al. 2025, Nature), as packaged in
    github.com/YijieYin/connectome_interpreter (MIT licence), pinned to a commit.

Usage:
    python scripts/download_data.py              # Zenodo + annotations + eye map into data/raw/
    python scripts/download_data.py --shiu       # ...plus the Shiu et al. table for Phase 3
    python scripts/download_data.py --minimal    # only what Phases 4-5 need (~130 MB, no Zenodo)
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

# Shiu et al. (2024) model inputs, pinned to commit 91bdd1e of philshiu/Drosophila_brain_model.
SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SHIU_FILES = {
    "shiu2024_connectivity_783.parquet": (
        (
            "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/"
            f"{SHIU_COMMIT}/Connectivity_783.parquet"
        ),
        "3d802fd542b5d18570ba1ba0bb0abed9",
    ),
}

# Phase 5 eye map, pinned to commit aaa7e73 of YijieYin/connectome_interpreter.
EYEMAP_COMMIT = "aaa7e738c161bbb6c475e092ca18c4b11e8e6324"
EYEMAP_RAW = ("https://raw.githubusercontent.com/YijieYin/connectome_interpreter/"
              f"{EYEMAP_COMMIT}/connectome_interpreter/data")
EYEMAP_FILES = {
    "matsliah2024_fafb_right_columns.csv": (
        f"{EYEMAP_RAW}/Matsliah2024/fafb_right_vis_cols.csv",
        "342a12b2da2f8e41f147e65c90e65a36",
    ),
    "zhao2025_eyemap_right.csv": (
        f"{EYEMAP_RAW}/Zhao2024/ucl_hex_right_20240701_tomale.csv",
        "22765893f44bcfec66fbf4246acc1aca",
    ),
}

# What the closed-loop phases need: annotations, the Shiu table (graph variant shiu_min5) and
# the eye map. The Zenodo connectivity (~1 GB) is only needed for Phases 1-3 and the "ours"
# graph variants.
ANNOTATIONS = f"neuron_annotations_{ANNOTATIONS_TAG}.tsv"

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
    parser.add_argument(
        "--shiu", action="store_true",
        help="also download the Shiu et al. 2024 connectivity table (Phase 3 comparison)",
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="only the files Phases 4-5 need: annotations, Shiu table, eye map (no Zenodo)",
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.minimal:
        files = {ANNOTATIONS: FILES[ANNOTATIONS]} | SHIU_FILES | EYEMAP_FILES
    else:
        files = FILES | EYEMAP_FILES | (SHIU_FILES if args.shiu else {})
    for name, (url, md5) in files.items():
        fetch(name, url, md5, args.data_dir)
    print(f"\nAll files ready in {args.data_dir}")


if __name__ == "__main__":
    main()
