#!/usr/bin/env python3
"""
Duplicate / near-duplicate image detector.

Usage:
  python dup_images.py /path/to/folder
  python dup_images.py /path/to/folder --threshold 6
  python dup_images.py /path/to/folder --recursive

Output:
  - Prints groups of duplicate/similar images
  - Saves groups to duplicate_groups.json
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

from PIL import Image

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".heic",
}


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file()]
    else:
        files = [p for p in folder.iterdir() if p.is_file()]
    return [p for p in files if p.suffix.lower() in IMAGE_EXTS]


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int:
    """
    Difference hash.
    Produces a hash robust to resize/crop/light changes.
    """
    with Image.open(path) as img:
        img = img.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        pixels = list(img.getdata())

    rows = []
    for y in range(hash_size):
        row_start = y * (hash_size + 1)
        row = pixels[row_start : row_start + hash_size + 1]
        bits = 0
        for x in range(hash_size):
            if row[x] > row[x + 1]:
                bits |= 1 << x
        rows.append(bits)

    # Pack rows into a single integer
    h = 0
    for i, row_bits in enumerate(rows):
        h |= row_bits << (i * hash_size)
    return h


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_groups(paths: List[Path], threshold: int) -> List[List[str]]:
    """
    1) Exact duplicates by MD5
    2) Near duplicates by dHash Hamming distance <= threshold
    """
    n = len(paths)
    uf = UnionFind(n)

    # Exact duplicates
    md5_map = {}
    for i, p in enumerate(paths):
        try:
            md5 = file_md5(p)
        except Exception:
            continue
        if md5 in md5_map:
            uf.union(i, md5_map[md5])
        else:
            md5_map[md5] = i

    # Near duplicates
    hashes = []
    valid_indices = []
    for i, p in enumerate(paths):
        try:
            hashes.append(dhash(p))
            valid_indices.append(i)
        except Exception:
            pass

    for a in range(len(valid_indices)):
        i = valid_indices[a]
        for b in range(a + 1, len(valid_indices)):
            j = valid_indices[b]
            if hamming_distance(hashes[a], hashes[b]) <= threshold:
                uf.union(i, j)

    groups = {}
    for i, p in enumerate(paths):
        root = uf.find(i)
        groups.setdefault(root, []).append(str(p))

    # Keep only groups with 2+ images
    result = [sorted(g) for g in groups.values() if len(g) > 1]
    result.sort(key=lambda g: (-len(g), g[0]))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate / near-duplicate images in a folder."
    )
    parser.add_argument("folder", type=str, help="Folder containing images")
    parser.add_argument(
        "--threshold",
        type=int,
        default=6,
        help="Hamming distance threshold for near-duplicates (default: 6)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Scan subfolders recursively"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="duplicate_groups.json",
        help="Output JSON file (default: duplicate_groups.json)",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Error: '{folder}' is not a valid directory.")

    images = list_images(folder, recursive=args.recursive)
    if not images:
        print("No images found.")
        return

    groups = build_groups(images, threshold=args.threshold)

    # Print groups
    print(f"Found {len(groups)} duplicate/similar groups.\n")
    for idx, group in enumerate(groups, 1):
        print(f"Group {idx} ({len(group)} images):")
        for item in group:
            print(f"  - {item}")
        print()

    # Save JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    print(f"Saved groups to: {args.output}")


if __name__ == "__main__":
    main()
