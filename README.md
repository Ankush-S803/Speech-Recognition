# 🎙️ Speech-Controlled Maze Navigation

> **A Soft Computing Mini-Project** combining deep learning-based speech recognition and real-time speech processing with an interactive, randomized maze game. Navigate a mouse (🐭) to the cheese/exit (🚪) using directional voice commands!

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](#)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

---

## 📌 Project Overview

This project implements an intelligent, speech-driven interactive gaming system. The objective is to navigate an agent through a procedurally generated maze using voice commands.

The system incorporates:
1. **Soft Computing / Deep Learning**: Audio feature extraction via **Mel-Frequency Cepstral Coefficients (MFCCs)** paired with a **Convolutional Neural Network (CNN)** built with PyTorch.
2. **Dual Speech Processing Pipeline**:
   - **Real-Time Client-Side Voice Engine**: Uses the HTML5 Web Speech API for low-latency, responsive command execution and continuous hands-free navigation.
   - **Server-Side Deep Learning Inference**: A PyTorch CNN inference engine running over Flask (`/predict`) capable of processing raw 16 kHz WAV audio buffers.
3. **Smart Procedural Maze Generation**: A pure Python maze engine implementing **Depth-First Search (DFS) recursive backtracking** with **Breadth-First Search (BFS) distance targeting** to guarantee accessible and demonstrative short paths (6–8 steps to exit).
4. **Modern Viewport-Fit Dashboard**: A clean, single-screen 3-column interface (no vertical scrolling) with a live activity log, interactive D-pad fallback, speech pipeline status indicator, and victory celebration overlay.

---

## 🎮 Supported Voice Commands

The system listens for directional voice commands:

| Command | Action |
|:---:|:---|
| **`"UP"`** | Moves the mouse one cell upward |
| **`"DOWN"`** | Moves the mouse one cell downward |
| **`"LEFT"`** | Moves the mouse one cell to the left |
| **`"RIGHT"`** | Moves the mouse one cell to the right |

*Keyboard arrow keys and on-screen D-pad buttons are also supported as fallback inputs.*

---

## 🛠️ Technologies Used

### Backend & Machine Learning
- **Python 3.10+**: Core programming language.
- **PyTorch**: Deep learning framework used for the Convolutional Neural Network (CNN) classifier.
- **Librosa / Scipy**: Audio signal processing, waveform reslicing, and MFCC feature extraction.
- **Flask**: Lightweight WSGI web application framework exposing RESTful control and inference endpoints.
- **NumPy & Scikit-learn**: Numerical computations and classification metrics.

### Frontend & UI
- **HTML5 & Canvas API**: High-performance grid and character rendering.
- **Vanilla CSS3**: Modern dark-mode glassmorphic theme with a responsive 3-column layout fitted to the viewport.
- **Vanilla JavaScript**: State synchronization, Canvas sprite rendering, Web Speech API integration, and audio stream management.

---

## 🏗️ Project Architecture & File Structure

```
Speech-Recognition/
│
├── soft_computing_model.py   # Core Academic Soft Computing Module:
│                             #   - Section 1: Configuration & Hyperparameters
│                             #   - Section 2: MFCC Feature Extraction (Librosa)
│                             #   - Section 3: CNN & MLP Neural Network Architectures
│                             #   - Section 4: PyTorch Audio Dataset & DataLoader
│                             #   - Section 5: Training Loop (Adam, Scheduler, Early Stopping)
│                             #   - Section 6: SpeechCommandClassifier Inference Engine
│                             #   - Section 7: Verification Self-Test
│
├── app.py                    # Flask Web Server & REST API endpoints
├── maze_engine.py            # Procedural DFS maze generator with BFS 6-8 step exit targeting
├── train_model.py            # Automated dataset download & training pipeline script
│
├── templates/
│   └── index.html            # Main dashboard UI template (3-column layout)
│
├── static/
│   ├── css/
│   │   └── style.css         # Dark glassmorphic styling, viewport-fit rules
│   └── js/
│       └── maze.js           # Canvas renderer, continuous voice recognition, event bindings
│
├── experiments/              # Checkpoint storage (best_model.pth)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🧠 Soft Computing & Neural Network Details

### 1. Audio Signal Preprocessing & MFCC Extraction
Raw 1-second audio waveforms sampled at **16 kHz** are transformed into 2D time-frequency representations:
- **Pre-emphasis Filter**: Amplifies high-frequency energy.
- **Short-Time Fourier Transform (STFT)**: Frame length $N_{\text{fft}} = 512$, hop length $= 160$ (~10 ms frames).
- **Mel Filterbank**: 80 triangular filters mapped across the perceptual Mel scale.
- **Log Compression & Discrete Cosine Transform (DCT)**: Yields **40 MFCC coefficients** per frame, producing a feature matrix of size `(40, T)`.

### 2. CNN Architecture
```
Input Feature Map: (Batch, 1, 40, T)
  │
  ├── Conv2D(1 → 32, kernel=3×3)  → BatchNorm → ReLU → MaxPool(2×2)
  ├── Conv2D(32 → 64, kernel=3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├── Conv2D(64 → 128, kernel=3×3)→ BatchNorm → ReLU → MaxPool(2×2)
  │
  ├── AdaptiveAvgPool2d(1, 1) → Flatten (128 units)
  │
  ├── Dense(128 → 256) → ReLU → Dropout(0.4)
  ├── Dense(256 → 128) → ReLU → Dropout(0.3)
  └── Dense(128 → 4)   → Softmax Output (Left, Right, Up, Down)
```

---

## 🚀 Step-by-Step Installation & Running Guide

Follow these instructions to run the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/Ankush-S803/Speech-Recognition.git
cd Speech-Recognition
```

### 2. Set Up a Python Virtual Environment

#### On Windows (PowerShell or Command Prompt):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

Start the Flask server:

```bash
python app.py
```

You should see output similar to:
```
INFO:root:Flask app ready.
INFO:root:Starting server on http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### 5. Access the Game

1. Open your web browser (Google Chrome or Microsoft Edge recommended for native Web Speech API support).
2. Navigate to: **`http://127.0.0.1:5000`**
3. When prompted, allow **Microphone Access**.
4. Click **"Speak Command"** or check **"Continuous Voice Mode"** to start navigating hands-free!

---

## 🎯 How to Play

1. **Speak**: Say `"up"`, `"down"`, `"left"`, or `"right"`. The mouse entity (🐭) will immediately advance in that direction if no wall blocks the path.
2. **Continuous Voice Mode**: Turn on the **Continuous Voice Mode** toggle to speak commands consecutively without having to click the button each time.
3. **New Maze**: Click the **"New Maze"** button at any time to regenerate a fresh maze with an optimal path.
4. **Manual Controls**: Click the on-screen **D-Pad buttons** or use your keyboard's **Arrow Keys** (`↑`, `↓`, `←`, `→`).
5. **Win the Game**: Guide the mouse to the exit cell (🚪) to trigger the victory banner!

---

## 🏋️ Optional: Model Training

To train the PyTorch CNN on the Google Speech Commands v0.02 dataset:

```bash
python train_model.py
```
This script will:
1. Automatically download the relevant command subsets (`left`, `right`, `up`, `down`) if not already present.
2. Extract MFCC features with train/validation/test splits.
3. Train the CNN with Adam optimizer and early stopping.
4. Save the top checkpoint to `experiments/best_model.pth`.

To independently verify the soft computing module:
```bash
python soft_computing_model.py
```

---

## 📡 REST API Reference

| Endpoint | Method | Payload / Format | Description |
|---|---|---|---|
| `/` | `GET` | — | Serves the single-page application |
| `/predict` | `POST` | `multipart/form-data` (audio file) | Returns predicted direction, confidence, and updated maze state |
| `/maze/move` | `POST` | `{"direction": "up" \| "down" \| "left" \| "right"}` | Executes a manual movement step |
| `/maze/new` | `POST` | `{"rows": 9, "cols": 9}` | Generates a new randomized maze |
| `/maze/state` | `GET` | — | Returns the current maze matrix, player coordinates, and status |

---

## 📄 License & Attribution

- **Base Dataset**: Google Speech Commands Dataset v0.02.
- **Reference**: PyTorch keyword spotting baseline principles inspired by [mabdulam/speech-keyword-spotting](https://github.com/mabdulam/speech-keyword-spotting).
- **License**: Released under the [MIT License](LICENSE).
