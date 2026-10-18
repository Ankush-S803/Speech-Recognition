# 🎙️ Speech-Controlled Maze Navigation

> A **Soft Computing mini-project** that combines deep learning–based speech
> recognition with an interactive, randomised maze game.  
> Navigate a 🐭 mouse through a maze using only your voice.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](#)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## 📸 Features

| Feature | Details |
|---|---|
| 🧠 **CNN Speech Model** | Convolutional Neural Network trained on Google Speech Commands v0.02 |
| 🎤 **Browser Mic** | Captures 1.5 s of audio via the Web MediaRecorder API |
| 🗺️ **Randomised Maze** | Recursive-backtracking DFS generates a new maze every time |
| 📊 **Live Probability Bars** | See per-command softmax probabilities in real time |
| ⌨️ **Keyboard Fallback** | D-pad buttons + Arrow keys work without a microphone |
| 🏆 **Win Detection** | Overlay when the mouse reaches the exit 🚪 |

---

## 🏗️ Project Structure

```
Speech-Controlled-Maze/
│
├── soft_computing_model.py   ← ★ ACADEMIC MODULE (open for professor inspection)
│                                  MFCC extraction, CNN architecture, training loop
│
├── app.py                    ← Flask backend (REST endpoints)
├── maze_engine.py            ← Maze generation & game logic (pure Python)
│
├── templates/
│   └── index.html            ← Single-page UI
├── static/
│   ├── css/style.css         ← Dark glassmorphism theme
│   └── js/maze.js            ← Canvas renderer + MediaRecorder integration
│
├── experiments/              ← Model checkpoints saved here after training
│   └── best_model.pth        (auto-created; not committed)
│
├── requirements.txt
└── README.md
```

---

## 🧠 Soft Computing Module

**`soft_computing_model.py`** is the **dedicated academic file** containing all neural-network logic:

```
Section 1 — Configuration          (commands, hyper-parameters)
Section 2 — Feature Extraction     (MFCC pipeline via librosa)
Section 3 — Neural Network         (KeywordSpottingCNN + MLPClassifier)
Section 4 — Dataset                (PyTorch AudioCommandDataset)
Section 5 — Training Pipeline      (Adam, cross-entropy, early stopping)
Section 6 — Inference Engine       (SpeechCommandClassifier wrapper)
Section 7 — Self-Test              (run with python soft_computing_model.py)
```

### CNN Architecture

```
Input: (batch, 1, 40, T)   ← single-channel MFCC spectrogram

[CNN Feature Extractor]
  Conv2d(1→32, 3×3)  → BatchNorm → ReLU → MaxPool(2×2)
  Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  Conv2d(64→128,3×3) → BatchNorm → ReLU → MaxPool(2×2)
  AdaptiveAvgPool2d(1,1) → flatten → 128-d

[MLP Head]
  FC(128→256) → ReLU → Dropout(0.4)
  FC(256→128) → ReLU → Dropout(0.3)
  FC(128→6)                           ← 6 commands

Output: (batch, 6)   ← raw logits
```

### MFCC Pipeline

```
Audio (16 kHz, 1 s)
  → Pre-emphasis
  → STFT (FFT=512, hop=160 ≈ 10 ms)
  → 80 Mel filterbanks
  → Log energy
  → DCT → 40 MFCC coefficients
  → (40 × T) feature map → CNN input
```

### Training Setup

| Hyper-parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping | patience = 5 |
| LR scheduler | ReduceLROnPlateau (factor 0.5) |
| Loss | CrossEntropyLoss |

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/Ankush-S803/Speech-Recognition.git
cd Speech-Recognition
```

### 2. Virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app (demo mode — no training required)

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

> **Demo mode** — if no trained model checkpoint exists, the classifier
> returns a random prediction so you can still test the UI and maze mechanics
> immediately.

---

## 🎓 Training on Real Data (optional)

### Download Google Speech Commands dataset

```bash
cd data/raw
# Windows (PowerShell):
Invoke-WebRequest http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -OutFile speech.tar.gz
tar -xzf speech.tar.gz

# Linux / macOS:
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
```

### Train the model

```python
# In a Python script or REPL:
from soft_computing_model import train_model

# Build your sample lists: [(audio_path, label_string), ...]
train_model(train_samples, val_samples, model_type="cnn")
```

The best checkpoint is saved to `experiments/best_model.pth` and loaded
automatically on next `python app.py`.

### Verify the soft-computing module independently

```bash
python soft_computing_model.py
```

---

## 🎮 How to Play

| Action | Method |
|---|---|
| **Move mouse** | Say "left", "right", "up", or "down" |
| **Confirm / reject** | Say "yes" or "no" |
| **New maze** | Click **New Maze** button |
| **Keyboard fallback** | Arrow keys or on-screen D-pad |
| **Quick record** | Press **Space bar** |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network framework |
| `librosa` | Audio loading & MFCC extraction |
| `flask` | Web server / REST API |
| `numpy` | Numerical arrays |
| `sounddevice` | Server-side mic capture (optional) |
| `scipy` | Signal processing utilities |
| `scikit-learn` | Evaluation metrics |

---

## 🛠️ API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serve web UI |
| POST | `/predict` | Upload WAV audio → returns command + maze state |
| POST | `/maze/move` | `{direction}` → move mouse, return maze state |
| POST | `/maze/new` | Generate new maze |
| GET | `/maze/state` | Get current maze state as JSON |

---

## 📐 Academic Attribution

This project adapts the MFCC feature extraction and MLP baseline from:
> **mabdulam/speech-keyword-spotting** — PyTorch keyword spotting with Google Speech Commands  
> https://github.com/mabdulam/speech-keyword-spotting

The CNN architecture, Flask backend, maze engine, and UI were built from scratch for this mini-project.

---

## 📄 License

MIT © 2024 — Ankush-S803
