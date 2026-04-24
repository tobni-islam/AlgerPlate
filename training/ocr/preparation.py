from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from augment import augment_plate, resize_and_pad_gray, save_augmentation_preview
from torch.utils.data import DataLoader, Dataset

CHARS = list("0123456789")
BLANK_TOKEN = "<BLANK>"
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
CHAR_TO_IDX = {c: i for i, c in IDX_TO_CHAR.items()}
BLANK_IDX = len(CHARS)
NUM_CLASSES = len(CHARS) + 1


@dataclass
class PrepConfig:
    csv_path: Path
    images_dir: Path
    out_dir: Path
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    img_w: int = 94
    img_h: int = 24
    max_label_len: int = 10


def _split_group(
    indices: np.ndarray, ratios: Tuple[float, float, float], rng: np.random.Generator
):
    """Split a single group of indices into train/val/test."""
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(indices)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    shuffled = indices.copy()
    rng.shuffle(shuffled)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Ensure all samples are assigned.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train : n_train + n_val]
    test_idx = shuffled[n_train + n_val : n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def split_dataframe(
    df: pd.DataFrame, cfg: PrepConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simplified random split based on unique file_names."""

    # 1. Shuffle the entire dataframe once to ensure randomness
    df_shuffled = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    # 2. Calculate split indices
    n = len(df_shuffled)
    train_end = int(cfg.train_ratio * n)
    val_end = train_end + int(cfg.val_ratio * n)

    # 3. Slice the dataframe
    train_df = df_shuffled.iloc[:train_end]
    val_df = df_shuffled.iloc[train_end:val_end]
    test_df = df_shuffled.iloc[val_end:]

    # 4. Reset indices for a clean slate
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class PlateDataset(Dataset):
    """Dataset that returns (image_tensor, label_indices, label_length)."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        img_w: int = 94,
        img_h: int = 24,
        training: bool = False,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.img_w = img_w
        self.img_h = img_h
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_name = str(row["file_name"])
        text = str(row["text"])

        img_path = self.images_dir / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        rng = np.random.default_rng(self.seed + idx)
        if self.training:
            img = augment_plate(img, rng)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        # Final deterministic resize/pad for LPRNet.
        img = resize_and_pad_gray(
            img, target_w=self.img_w, target_h=self.img_h, pad_value=255
        )

        # Standard normalization used by many LPRNet implementations.
        # Maps [0,255] -> roughly [-1, 1]
        img = img.astype(np.float32)
        img = (img - 127.5) * 0.0078125
        img = np.expand_dims(img, axis=0)  # [1, H, W]

        label_indices = [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]
        if len(label_indices) == 0:
            raise ValueError(
                f"Empty label after cleaning for row {idx}: {row.to_dict()}"
            )

        return (
            torch.from_numpy(img),
            torch.tensor(label_indices, dtype=torch.long),
            torch.tensor(len(label_indices), dtype=torch.long),
            file_name,
            text,
        )


def collate_fn(batch):
    """Custom collate for variable-length CTC labels."""
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.cat([item[1] for item in batch], dim=0)
    label_lengths = torch.stack([item[2] for item in batch], dim=0)
    file_names = [item[3] for item in batch]
    texts = [item[4] for item in batch]
    return images, labels, label_lengths, file_names, texts


def build_dataloader(
    df: pd.DataFrame,
    images_dir: Path,
    batch_size: int,
    training: bool,
    shuffle: bool,
    num_workers: int = 2,
    seed: int = 42,
    img_w: int = 94,
    img_h: int = 24,
):
    dataset = PlateDataset(
        df=df,
        images_dir=images_dir,
        img_w=img_w,
        img_h=img_h,
        training=training,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=training,
        collate_fn=collate_fn,
    )


def save_split_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df[["file_name", "text"]].copy()
    df.to_csv(out_path, index=False)


def parse_args() -> PrepConfig:
    parser = argparse.ArgumentParser(
        description="Prepare Algerian plate dataset for LPRNet"
    )
    parser.add_argument(
        "--csv", dest="csv_path", type=str, required=True, help="Path to lp_dataset.csv"
    )
    parser.add_argument(
        "--images-dir", type=str, required=True, help="Folder containing plate crops"
    )
    parser.add_argument(
        "--out-dir", type=str, required=True, help="Output folder for split CSVs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--img-w", type=int, default=94)
    parser.add_argument("--img-h", type=int, default=24)
    parser.add_argument("--max-label-len", type=int, default=10)
    args = parser.parse_args()

    return PrepConfig(
        csv_path=Path(args.csv_path),
        images_dir=Path(args.images_dir),
        out_dir=Path(args.out_dir),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        img_w=args.img_w,
        img_h=args.img_h,
        max_label_len=args.max_label_len,
    )


def main():
    cfg = PrepConfig(
        csv_path=Path(
            "/kaggle/input/datasets/tobniislam/lp-dataset/LP Data/lp_dataset.csv"
        ),
        images_dir=Path("/kaggle/input/datasets/tobniislam/lp-dataset/LP Data/Images"),
        out_dir=Path("/kaggle/working/"),
        seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")
    if not cfg.images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {cfg.images_dir}")

    if abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    df = pd.read_csv(cfg.csv_path)
    if "file_name" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain columns: file_name, text")

    # Keep only rows whose image exists.
    def exists_fn(x: str) -> bool:
        return (cfg.images_dir / str(x)).exists()

    df = df[df["file_name"].astype(str).apply(exists_fn)].copy()
    df = df.reset_index(drop=True)

    # Drop duplicates if any.
    df = df.drop_duplicates(subset=["file_name", "text"]).reset_index(drop=True)

    print(f"Loaded cleaned dataset: {len(df)} rows")
    print("Label length distribution:")
    print(df["text"].astype(str).str.len().value_counts().sort_index())

    train_df, val_df, test_df = split_dataframe(df, cfg)

    print("Split sizes:")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")

    overlap_train_val = set(train_df["file_name"]).intersection(
        set(val_df["file_name"])
    )
    overlap_train_test = set(train_df["file_name"]).intersection(
        set(test_df["file_name"])
    )
    overlap_val_test = set(val_df["file_name"]).intersection(set(test_df["file_name"]))
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            "Split leakage detected: some file_name values appear in multiple splits"
        )

    # Save split CSVs.
    save_split_csv(train_df, cfg.out_dir / "train.csv")
    save_split_csv(val_df, cfg.out_dir / "val.csv")
    save_split_csv(test_df, cfg.out_dir / "test.csv")

    # Optional preview of augmentations to visually inspect quality.
    preview_dir = Path("/kaggle/working/sanity")
    save_augmentation_preview(
        train_df,
        cfg.images_dir,
        preview_dir,
        seed=cfg.seed,
        n=12,
        img_w=cfg.img_w,
        img_h=cfg.img_h,
    )

    print(f"Saved splits to: {cfg.out_dir}")
    print(f"Saved augmentation previews to: {preview_dir}")


if __name__ == "__main__":
    main()
