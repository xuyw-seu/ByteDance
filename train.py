#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ByteDance model from processed NPZ dataset.
"""

import argparse
import os
import random
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


# BD_train.py depends on `utils.get_time_dif`; inject fallback when missing.
if "utils" not in sys.modules:
    utils_module = types.ModuleType("utils")

    def get_time_dif(start_time):
        return time.time() - start_time

    utils_module.get_time_dif = get_time_dif
    sys.modules["utils"] = utils_module


from BD_train import train  # noqa: E402
from model.ByteDance import ByteDance, Config  # noqa: E402


class ByteDanceDataset(Dataset):
    def __init__(self, len_seq: torch.Tensor, byte_seq: torch.Tensor, labels: torch.Tensor):
        self.len_seq = len_seq
        self.byte_seq = byte_seq
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (self.len_seq[idx], self.byte_seq[idx]), self.labels[idx]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_collate_fn(device: torch.device):
    def _collate(batch):
        x_len = torch.stack([item[0][0] for item in batch], dim=0).to(device)
        x_byte = torch.stack([item[0][1] for item in batch], dim=0).to(device)
        y = torch.stack([item[1] for item in batch], dim=0).to(device)
        return (x_len, x_byte), y

    return _collate


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ByteDance with NPZ dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed npz file.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Output directory.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda / cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--require-improvement", type=int, default=2000, help="Early stop patience in steps.")

    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Validation split ratio.")

    parser.add_argument("--embedding-size", type=int, default=128, help="Embedding size.")
    parser.add_argument("--trf-heads", type=int, default=8, help="Transformer heads.")
    parser.add_argument("--trf-layers", type=int, default=2, help="Length branch transformer layers.")
    parser.add_argument("--trf-layers-res", type=int, default=2, help="Raw branch transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    parser.add_argument("--loss-modulation-starts", type=int, default=0, help="PDGC start epoch.")
    parser.add_argument("--loss-modulation-ends", type=int, default=30, help="PDGC end epoch.")
    parser.add_argument("--pmr-momentum-coef", type=float, default=0.9, help="PDGC prototype momentum.")

    parser.add_argument("--num-classes", type=int, default=0, help="0 means infer from labels.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0, 1).")
    if not (0 < args.dev_ratio < 1):
        raise ValueError("--dev-ratio must be in (0, 1).")
    if args.train_ratio + args.dev_ratio >= 1:
        raise ValueError("train_ratio + dev_ratio must be < 1.")

    seed_everything(args.seed)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.data, allow_pickle=True)
    len_seq = torch.tensor(data["len_seq"], dtype=torch.long)
    byte_seq = torch.tensor(data["byte_seq"], dtype=torch.long)
    labels = torch.tensor(data["labels"], dtype=torch.long)

    if len(len_seq.shape) != 2 or len(byte_seq.shape) != 3:
        raise ValueError("Invalid input shape: expected len_seq [N, P], byte_seq [N, P, D].")
    if len_seq.shape[0] != byte_seq.shape[0] or len_seq.shape[0] != labels.shape[0]:
        raise ValueError("Sample size mismatch among len_seq, byte_seq, and labels.")

    n_samples = len(labels)
    n_train = int(n_samples * args.train_ratio)
    n_dev = int(n_samples * args.dev_ratio)
    n_test = n_samples - n_train - n_dev
    if n_train <= 0 or n_dev <= 0 or n_test <= 0:
        raise ValueError("Split size too small. Adjust ratios or dataset size.")

    dataset = ByteDanceDataset(len_seq, byte_seq, labels)
    train_set, dev_set, test_set = random_split(
        dataset,
        [n_train, n_dev, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    collate_fn = build_collate_fn(device)
    train_iter = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_iter = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_iter = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    num_classes = args.num_classes if args.num_classes > 0 else int(labels.max().item()) + 1
    model_cfg = Config(
        embedding_size=args.embedding_size,
        max_packet_num=len_seq.shape[1],
        num_classes=num_classes,
        trf_heads=args.trf_heads,
        trf_layers=args.trf_layers,
        trf_layers_res=args.trf_layers_res,
        dropout=args.dropout,
        device=device,
    )
    model = ByteDance(model_cfg).to(device)

    train_cfg = SimpleNamespace(
        num_classes=num_classes,
        device=device,
        class_list=[str(i) for i in range(num_classes)],
        mode="train",
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        require_improvement=args.require_improvement,
        loss_modulation_starts=args.loss_modulation_starts,
        loss_modulation_ends=args.loss_modulation_ends,
        pmr_momentum_coef=args.pmr_momentum_coef,
        load=False,
        save_path=str(args.output_dir / "bytedance.pt"),
        print_path=str(args.output_dir / "train_log.txt"),
        log_path=str(args.output_dir / "tensorboard"),
        loss_path=str(args.output_dir / "loss.csv"),
    )

    result = train(train_cfg, model, train_iter, dev_iter, test_iter)
    print("Training finished.")
    print("Test result:", result)


if __name__ == "__main__":
    main()
