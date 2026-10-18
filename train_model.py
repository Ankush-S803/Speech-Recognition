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


import wave
import struct
import numpy as np


def generate_synthetic_dataset(data_dir: str, num_per_class: int = 50):
    """
    Generate realistic acoustic synthetic samples for the 4 directional commands
    ('left', 'right', 'up', 'down') using formant synthesis, pitch contours,
    and attack/decay envelopes.
    """
    logger.info(f"Generating synthetic acoustic dataset in '{data_dir}' ({num_per_class} per class)...")
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Formant frequency profiles (F1, F2 in Hz) for the vowel core
    profiles = {
        "left":  {"f1": 550, "f2": 1800, "pitch": 140, "noise_tail": True},   # /ɛ/ + /ft/
        "right": {"f1": 750, "f2": 1300, "pitch": 120, "glide_f2": 2100},    # /aɪ/ diphthong
        "up":    {"f1": 650, "f2": 1150, "pitch": 160, "burst": True},        # /ʌ/ + /p/
        "down":  {"f1": 750, "f2": 1100, "pitch": 110, "glide_f2": 850},     # /aʊ/ + /n/
    }

    data_path = Path(data_dir)
    for cmd in COMMANDS:
        cmd_dir = data_path / cmd
        cmd_dir.mkdir(parents=True, exist_ok=True)
        prof = profiles.get(cmd, {"f1": 500, "f2": 1500, "pitch": 130})

        for i in range(num_per_class):
            # Pitch with slight natural variation
            f0 = prof["pitch"] + np.random.uniform(-8, 8)
            f1 = prof["f1"] + np.random.uniform(-25, 25)
            f2_start = prof["f2"] + np.random.uniform(-40, 40)
            f2_end = prof.get("glide_f2", f2_start) + np.random.uniform(-30, 30)

            # Pitch glottal pulse train
            glottal = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(4 * np.pi * f0 * t)

            # Formant resonances
            f2_contour = np.linspace(f2_start, f2_end, len(t))
            formant1 = np.sin(2 * np.pi * f1 * t)
            formant2 = np.sin(2 * np.pi * f2_contour * t)

            # Envelope: attack, hold, decay
            env = np.ones_like(t)
            attack_len = int(sr * 0.1)
            decay_start = int(sr * 0.6)
            env[:attack_len] = np.linspace(0, 1, attack_len)
            env[decay_start:] = np.linspace(1, 0, len(t) - decay_start)

            # Composite vocal audio
            signal = (0.5 * glottal + 0.3 * formant1 + 0.2 * formant2) * env

            # Add subtle friction / burst
            if prof.get("noise_tail"):
                noise = np.random.normal(0, 0.08, len(t))
                tail_mask = (t > 0.45) & (t < 0.75)
                signal[tail_mask] += noise[tail_mask]
            elif prof.get("burst"):
                burst_idx = int(sr * 0.55)
                signal[burst_idx:burst_idx + 200] += np.random.normal(0, 0.2, min(200, len(signal) - burst_idx))

            # Normalize & add minor room ambient
            signal += np.random.normal(0, 0.01, len(t))
            signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.85

            # Save as 16-bit PCM WAV
            wav_path = cmd_dir / f"sample_{i:03d}.wav"
            int16_data = (signal * 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(int16_data.tobytes())

    logger.info("[OK] Synthetic dataset generation complete.")


def scan_dataset(data_dir: str):
    """Scan data_dir for audio files belonging to each command class."""
    samples = []
    data_path = Path(data_dir)

    for cmd in COMMANDS:
        cmd_dir = data_path / cmd
        if not cmd_dir.exists():
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
    parser.add_argument("--epochs",     default=15, type=int)
    parser.add_argument("--save_path",  default=CHECKPOINT_PATH)
    parser.add_argument("--synthetic",  action="store_true",        help="Force generation of synthetic dataset")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Speech Keyword Spotting — Training")
    logger.info("=" * 60)
    logger.info(f"Commands   : {COMMANDS}")
    logger.info(f"Data dir   : {args.data_dir}")
    logger.info(f"Model type : {args.model_type}")
    logger.info(f"Max epochs : {args.epochs}")

    # ── Scan or Generate dataset ─────────────────────────────────────────────
    all_samples = scan_dataset(args.data_dir) if not args.synthetic else []
    if not all_samples:
        logger.info("No existing audio samples found. Auto-generating synthetic acoustic data...")
        generate_synthetic_dataset(args.data_dir, num_per_class=60)
        all_samples = scan_dataset(args.data_dir)

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

    logger.info("\n[OK] Training complete!")
    logger.info(f"  Best val accuracy : {max(history['val_acc']):.4f}")
    logger.info(f"  Checkpoint saved  : {args.save_path}")
    logger.info("\nRestart app.py to load the trained model.")


if __name__ == "__main__":
    main()
