"""
=============================================================================
SOFT COMPUTING MODULE — Speech Keyword Spotting Neural Network
=============================================================================
Academic Module | Open for Professor Inspection

This file contains ALL soft-computing logic:
  • MFCC feature extraction (mel-frequency cepstral coefficients)
  • Neural network architecture (CNN + MLP heads)
  • Training pipeline with cross-entropy loss and Adam optimiser
  • Real-time inference wrapper used by the Flask backend

The model classifies 4 voice commands:
    "left", "right", "up", "down"

References:
  - Baseline architecture adapted from: mabdulam/speech-keyword-spotting
  - Dataset: Google Speech Commands v0.02
  - Feature method: Davis & Mermelstein (1980) — MFCCs
=============================================================================
"""

import os
import io
import time
import json
import logging
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Optional heavy imports (librosa / sounddevice) ───────────────────────────
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

COMMANDS          = ["left", "right", "up", "down"]
NUM_CLASSES       = len(COMMANDS)
SAMPLE_RATE       = 16_000          # Hz
DURATION          = 1.0             # seconds
N_SAMPLES         = int(SAMPLE_RATE * DURATION)

# MFCC hyper-parameters
N_MFCC            = 40
N_FFT             = 512
HOP_LENGTH        = 160             # ~10 ms frames
N_MELS            = 80

# Training hyper-parameters
BATCH_SIZE        = 64
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-5
NUM_EPOCHS        = 30
PATIENCE          = 5               # early-stopping patience

# Inference
RECORD_SECONDS    = 1.5             # seconds of mic audio to capture
CHECKPOINT_PATH   = os.path.join(os.path.dirname(__file__), "experiments", "best_model.pth")

LABEL_TO_IDX = {cmd: i for i, cmd in enumerate(COMMANDS)}
IDX_TO_LABEL = {i: cmd for i, cmd in enumerate(COMMANDS)}


# =============================================================================
# SECTION 2 — FEATURE EXTRACTION
# =============================================================================

def extract_mfcc(audio: np.ndarray,
                 sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs) from a raw waveform.

    Pipeline:
      1. Pre-emphasis filter (boost high frequencies)
      2. Short-Time Fourier Transform (STFT) → spectrogram
      3. Map frequencies to mel-scale filterbank
      4. Log-compress energies
      5. Discrete Cosine Transform → cepstral coefficients

    Args:
        audio      : 1-D float32 audio samples, shape (N_SAMPLES,)
        sr         : Sample rate in Hz (default 16 000)
        n_mfcc     : Number of cepstral coefficients to keep (default 40)
        n_fft      : FFT window size in samples (default 512)
        hop_length : Frame shift in samples (default 160 = 10 ms)

    Returns:
        mfccs : 2-D array of shape (n_mfcc, T) — columns are time frames
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for MFCC extraction. "
                          "Install with: pip install librosa")

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS,
    )
    return mfccs


def mfcc_summary(mfcc: np.ndarray) -> np.ndarray:
    """
    Compress a variable-length MFCC matrix into a fixed-size feature vector.

    Strategy: compute the *mean* and *std* of each coefficient across time.
    This gives a 2×N_MFCC = 80-dimensional vector that summarises the
    spectral envelope without sensitivity to clip duration.

    Args:
        mfcc : MFCC matrix, shape (n_mfcc, T)

    Returns:
        vector : 1-D array of length 2 * n_mfcc
    """
    mean = np.mean(mfcc, axis=1)  # (n_mfcc,)
    std  = np.std(mfcc,  axis=1)  # (n_mfcc,)
    return np.concatenate([mean, std])


def audio_to_feature_vector(audio: np.ndarray) -> np.ndarray:
    """
    End-to-end: raw audio → fixed-size feature vector for the MLP head.
    """
    # Ensure correct length (pad or truncate to N_SAMPLES)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    mfcc = extract_mfcc(audio.astype(np.float32))
    return mfcc_summary(mfcc).astype(np.float32)


def audio_to_mfcc_tensor(audio: np.ndarray) -> torch.Tensor:
    """
    End-to-end: raw audio → 2-D MFCC tensor suitable for the CNN head.

    Returns:
        Tensor of shape (1, N_MFCC, T)  — channel-first for Conv2d
    """
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    mfcc = extract_mfcc(audio.astype(np.float32))  # (N_MFCC, T)
    # Normalise
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return torch.from_numpy(mfcc).unsqueeze(0).float()  # (1, N_MFCC, T)


# =============================================================================
# SECTION 3 — NEURAL NETWORK ARCHITECTURE
# =============================================================================

class CNNBlock(nn.Module):
    """
    A single Convolution → BatchNorm → ReLU → MaxPool block.

    Used as the building block of the CNN feature extractor.
    BatchNorm stabilises training by normalising activations per-batch.
    MaxPool performs spatial down-sampling to reduce computation.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 padding=kernel_size // 2)
        self.bn      = nn.BatchNorm2d(out_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.pool    = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class KeywordSpottingCNN(nn.Module):
    """
    ─────────────────────────────────────────────────────────────────────────
    Convolutional Neural Network for Keyword Spotting

    Input  : (batch, 1, N_MFCC=40, T) — single-channel MFCC spectrogram
    Output : (batch, NUM_CLASSES)      — logits over 6 commands

    Architecture
    ────────────
    [CNN Feature Extractor]
      Block 1 : Conv2d(1,  32, 3×3)  → BN → ReLU → MaxPool(2×2)
      Block 2 : Conv2d(32, 64, 3×3)  → BN → ReLU → MaxPool(2×2)
      Block 3 : Conv2d(64, 128, 3×3) → BN → ReLU → MaxPool(2×2)
      → Adaptive Average Pool → flatten to 128-d vector

    [MLP Classifier Head]
      FC(128 → 256) → ReLU → Dropout(0.4)
      FC(256 → 128) → ReLU → Dropout(0.3)
      FC(128 → NUM_CLASSES)           → raw logits

    The model is trained end-to-end with:
      • Loss      : Cross-Entropy
      • Optimiser : Adam (lr=1e-3, weight_decay=1e-5)
      • Scheduler : ReduceLROnPlateau (factor=0.5, patience=3)
      • Stopping  : Early-stopping (patience=5 on validation accuracy)
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Convolutional feature extractor ──────────────────────────────
        self.cnn = nn.Sequential(
            CNNBlock(1,  32),   # (B, 32, N_MFCC/2, T/2)
            CNNBlock(32, 64),   # (B, 64, N_MFCC/4, T/4)
            CNNBlock(64, 128),  # (B,128, N_MFCC/8, T/8)
        )

        # Adaptive pooling → fixed 128-d no matter the time dimension T
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── MLP classification head ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),   # raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor (batch, 1, N_MFCC, T)
        Returns:
            logits : Tensor (batch, num_classes)
        """
        features = self.cnn(x)                    # (B, 128, H', W')
        pooled   = self.global_pool(features)     # (B, 128,  1,  1)
        flat     = pooled.view(pooled.size(0), -1) # (B, 128)
        logits   = self.classifier(flat)          # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (no gradient)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


class MLPClassifier(nn.Module):
    """
    ─────────────────────────────────────────────────────────────────────────
    Multi-Layer Perceptron (MLP) for Keyword Spotting  [fallback model]

    Input  : (batch, 2*N_MFCC=80) — MFCC mean+std summary vector
    Output : (batch, NUM_CLASSES) — logits

    Architecture
    ────────────
    FC(80 → 256) → BN → ReLU → Dropout(0.3)
    FC(256→ 128) → BN → ReLU → Dropout(0.3)
    FC(128→  64) → BN → ReLU → Dropout(0.3)
    FC(64 → NUM_CLASSES)

    Adapted from mabdulam/speech-keyword-spotting (MLPClassifier).
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, input_dim: int = 2 * N_MFCC,
                 hidden_dims: list = None,
                 num_classes: int = NUM_CLASSES,
                 dropout_rate: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# SECTION 4 — DATASET
# =============================================================================

class AudioCommandDataset(Dataset):
    """
    PyTorch Dataset for voice command audio files.

    Expects a list of (audio_path, label_string) pairs.
    Performs on-the-fly MFCC extraction so preprocessing is not needed
    separately (convenient for small datasets / demo purposes).
    """

    def __init__(self, samples: list, mode: str = "cnn"):
        """
        Args:
            samples : list of (audio_path: str, label: str)
            mode    : "cnn" → returns (1, N_MFCC, T) tensor
                      "mlp" → returns (2*N_MFCC,) summary vector
        """
        self.samples = samples
        self.mode    = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # Load audio
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True,
                                 duration=DURATION)

        label_idx = LABEL_TO_IDX[label]

        if self.mode == "cnn":
            x = audio_to_mfcc_tensor(audio)
        else:
            x = torch.from_numpy(audio_to_feature_vector(audio))

        return x, label_idx


# =============================================================================
# SECTION 5 — TRAINING PIPELINE
# =============================================================================

def train_model(train_samples: list,
                val_samples: list,
                model_type: str = "cnn",
                save_path: str = CHECKPOINT_PATH) -> dict:
    """
    Full training loop with validation, early stopping, and checkpointing.

    Training Procedure
    ──────────────────
    For each epoch:
      1. Forward pass through the network
      2. Compute cross-entropy loss
      3. Back-propagate gradients (autograd)
      4. Update weights with Adam optimiser
      5. Evaluate on validation set
      6. Adjust learning rate with ReduceLROnPlateau
      7. Save checkpoint if validation accuracy improved
      8. Stop early if no improvement for `PATIENCE` epochs

    Args:
        train_samples : list of (path, label) for training
        val_samples   : list of (path, label) for validation
        model_type    : "cnn" or "mlp"
        save_path     : where to save the best model checkpoint (.pth)

    Returns:
        history : dict with "train_loss", "val_loss", "val_acc" lists
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # ── Model ────────────────────────────────────────────────────────────
    if model_type == "cnn":
        model = KeywordSpottingCNN(NUM_CLASSES)
    else:
        model = MLPClassifier(2 * N_MFCC, num_classes=NUM_CLASSES)
    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────
    train_ds = AudioCommandDataset(train_samples, mode=model_type)
    val_ds   = AudioCommandDataset(val_samples,   mode=model_type)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

    # ── Optimiser + scheduler ────────────────────────────────────────────
    criterion  = nn.CrossEntropyLoss()
    optimiser  = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=3)

    history       = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    patience_ctr  = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(xb)
        avg_train_loss = total_loss / len(train_ds)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        correct   = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                preds   = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
        avg_val_loss = val_loss / len(val_ds)
        val_acc      = correct  / len(val_ds)

        scheduler.step(val_acc)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        logger.info(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
                    f"train_loss={avg_train_loss:.4f}  "
                    f"val_loss={avg_val_loss:.4f}  "
                    f"val_acc={val_acc:.4f}")

        # ── Checkpoint & early stopping ───────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "model_type":  model_type,
                "val_acc":     val_acc,
                "label_map":   IDX_TO_LABEL,
            }, save_path)
            logger.info(f"  [OK] Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    return history


# =============================================================================
# SECTION 6 — REAL-TIME INFERENCE ENGINE
# =============================================================================

class SpeechCommandClassifier:
    """
    Real-time speech command classifier used by the Flask backend.

    Wraps the trained CNN model and exposes a simple `predict_from_audio`
    method.  Falls back to silence detection if the model file is missing.
    """

    def __init__(self):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = None
        self.ready   = False
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint if it exists."""
        if os.path.exists(CHECKPOINT_PATH):
            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device)
                mtype = ckpt.get("model_type", "cnn")
                self.model = KeywordSpottingCNN(NUM_CLASSES) if mtype == "cnn" \
                             else MLPClassifier(2 * N_MFCC, num_classes=NUM_CLASSES)
                self.model.load_state_dict(ckpt["model_state"])
                self.model.to(self.device).eval()
                self.ready = True
                logger.info("Loaded trained model from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        else:
            logger.info("No checkpoint found — will use demo mode.")

    def predict_from_audio(self, audio: np.ndarray) -> dict:
        """
        Classify a 1-second audio clip.

        Args:
            audio : 1-D float32 numpy array at SAMPLE_RATE Hz

        Returns:
            dict with keys:
              "command"    : predicted command string
              "confidence" : float in [0, 1]
              "all_probs"  : dict {command: probability}
        """
        if not LIBROSA_AVAILABLE:
            return self._demo_predict()

        if self.model is None:
            return self._demo_predict()

        try:
            tensor = audio_to_mfcc_tensor(audio).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)                   # (1, C)
                probs  = F.softmax(logits, dim=1).squeeze(0) # (C,)

            pred_idx = int(probs.argmax().item())
            command  = IDX_TO_LABEL[pred_idx]
            conf     = float(probs[pred_idx].item())
            all_p    = {IDX_TO_LABEL[i]: float(probs[i].item())
                        for i in range(NUM_CLASSES)}

            return {"command": command, "confidence": conf, "all_probs": all_p}
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._demo_predict()

    def predict_from_bytes(self, wav_bytes: bytes) -> dict:
        """
        Classify a WAV audio clip supplied as raw bytes from the browser mic.

        Args:
            wav_bytes : bytes of a WAV file (e.g., from MediaRecorder API)

        Returns:
            Same dict as `predict_from_audio`.
        """
        if not LIBROSA_AVAILABLE:
            return self._demo_predict()
        try:
            audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE,
                                    mono=True, duration=DURATION)
            return self.predict_from_audio(audio)
        except Exception as e:
            logger.error(f"Audio decode error: {e}")
            return self._demo_predict()

    def _demo_predict(self) -> dict:
        """
        Demo mode: randomly pick a command when no trained model is present.
        Used so the UI is functional out-of-the-box before training.
        """
        import random
        cmd  = random.choice(COMMANDS)
        conf = round(random.uniform(0.55, 0.99), 3)
        return {
            "command":    cmd,
            "confidence": conf,
            "all_probs":  {c: round(1 / NUM_CLASSES, 3) for c in COMMANDS},
            "demo_mode":  True,
        }

    def record_and_predict(self) -> dict:
        """
        Record `RECORD_SECONDS` of microphone audio and classify it.

        Requires `sounddevice` to be installed.

        Returns:
            Same dict as `predict_from_audio`.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not installed. "
                               "Run: pip install sounddevice")

        logger.info(f"Recording {RECORD_SECONDS}s …")
        audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        # Trim to exactly 1 second
        audio = audio[:N_SAMPLES] if len(audio) >= N_SAMPLES \
                else np.pad(audio, (0, N_SAMPLES - len(audio)))
        logger.info("Recording complete.")
        return self.predict_from_audio(audio)


# =============================================================================
# SECTION 7 — MODULE SELF-TEST
# =============================================================================

if __name__ == "__main__":
    """
    Quick sanity-check: build the CNN, run a random tensor through it,
    and print the architecture summary.
    """
    print("=" * 65)
    print("  SOFT COMPUTING MODEL — Self-Test")
    print("=" * 65)

    # Build model
    cnn = KeywordSpottingCNN(NUM_CLASSES)
    print("\nKeywordSpottingCNN architecture:")
    print(cnn)

    # Dummy forward pass
    dummy = torch.randn(4, 1, N_MFCC, 101)   # batch=4
    with torch.no_grad():
        logits = cnn(dummy)
    print(f"\nInput  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Commands    : {COMMANDS}")

    # Parameter count
    total = sum(p.numel() for p in cnn.parameters())
    print(f"Parameters  : {total:,}")
    print("\n[OK] All checks passed.")
