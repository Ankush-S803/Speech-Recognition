"""
train_model.py — Standalone training script for the soft-computing model.

Usage:
    python train_model.py --data_dir data/raw --model_type cnn --epochs 30

This script:
  1. Scans `data_dir` for folders named after the 6 supported commands
  2. Builds train/val/test splits (80/10/10)
  3. Calls soft_computing_model.train_model()
  4. Saves the best checkpoint to experiments/best_model.pth
"""

import os
import argparse
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Import our soft computing module
from soft_computing_model import COMMANDS, train_model, CHECKPOINT_PATH


def scan_dataset(data_dir: str):
    """
    Scan `data_dir` for audio files belonging to each command class.

    Expects directory structure:
        data_dir/
          left/
            file1.wav
            file2.wav
            ...
          right/ ...
          up/   ...
          ...

    Returns:
        List of (file_path, label) tuples.
    """
    samples = []
    data_path = Path(data_dir)

    for cmd in COMMANDS:
        cmd_dir = data_path / cmd
        if not cmd_dir.exists():
            logger.warning(f"Command directory not found: {cmd_dir}")
            continue
        wavs = list(cmd_dir.glob("*.wav"))
        logger.info(f"  {cmd}: {len(wavs)} audio files")
        for wav in wavs:
            samples.append((str(wav), cmd))

    return samples


def split_samples(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split samples into train / val / test sets."""
    random.seed(seed)
    random.shuffle(samples)
    n     = len(samples)
    n_tr  = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = samples[:n_tr]
    val   = samples[n_tr:n_tr + n_val]
    test  = samples[n_tr + n_val:]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Train the keyword-spotting model")
    parser.add_argument("--data_dir",   default="data/raw",         help="Root directory containing command sub-folders")
    parser.add_argument("--model_type", default="cnn",              choices=["cnn", "mlp"])
    parser.add_argument("--epochs",     default=30, type=int)
    parser.add_argument("--save_path",  default=CHECKPOINT_PATH)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Speech Keyword Spotting — Training")
    logger.info("=" * 60)
    logger.info(f"Data dir   : {args.data_dir}")
    logger.info(f"Model type : {args.model_type}")
    logger.info(f"Max epochs : {args.epochs}")

    # ── Scan dataset ─────────────────────────────────────────────────────────
    logger.info("\nScanning dataset...")
    all_samples = scan_dataset(args.data_dir)
    if not all_samples:
        logger.error("No audio samples found. Check --data_dir.")
        return

    logger.info(f"Total samples: {len(all_samples)}")

    # ── Split ─────────────────────────────────────────────────────────────────
    train_s, val_s, test_s = split_samples(all_samples)
    logger.info(f"Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("\nStarting training...")
    history = train_model(
        train_samples=train_s,
        val_samples=val_s,
        model_type=args.model_type,
        save_path=args.save_path,
    )

    logger.info("\n✓ Training complete!")
    logger.info(f"  Best val accuracy : {max(history['val_acc']):.4f}")
    logger.info(f"  Checkpoint saved  : {args.save_path}")
    logger.info("\nRestart app.py to load the trained model.")


if __name__ == "__main__":
    main()
