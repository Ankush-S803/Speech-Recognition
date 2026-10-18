"""
app.py — Flask backend for the Speech-Controlled Maze Navigation project.

Routes:
  GET  /              — serve index.html
  POST /predict       — receive WAV audio bytes, return predicted command
  POST /maze/move     — move the mouse given a direction command
  POST /maze/new      — generate a fresh randomised maze
  GET  /maze/state    — get current maze state as JSON
"""

import os
import json
import logging
from flask import Flask, request, jsonify, render_template

# Import from our dedicated soft-computing module
from soft_computing_model import SpeechCommandClassifier
from maze_engine import MazeEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Singleton instances (initialised once at startup) ─────────────────────────
classifier = SpeechCommandClassifier()
maze       = MazeEngine(rows=11, cols=11)

logger.info("Flask app ready.")


# =============================================================================
# HTML page
# =============================================================================

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


# =============================================================================
# Speech recognition endpoint
# =============================================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive a WAV audio clip (bytes) from the browser.
    Run it through the soft-computing classifier.
    Return the predicted command + confidence.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    wav_bytes = request.files["audio"].read()

    try:
        result = classifier.predict_from_bytes(wav_bytes)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

    # If it's a movement command, also move the mouse automatically
    command = result.get("command", "")
    move_result = None
    if command in ("left", "right", "up", "down"):
        move_result = maze.move(command)

    return jsonify({
        "command":    command,
        "confidence": result.get("confidence", 0),
        "all_probs":  result.get("all_probs", {}),
        "demo_mode":  result.get("demo_mode", False),
        "maze":       maze.state(),
        "move_result": move_result,
    })


# =============================================================================
# Maze control endpoints
# =============================================================================

@app.route("/maze/move", methods=["POST"])
def maze_move():
    """Move the mouse via a JSON body {direction: 'left'|'right'|'up'|'down'}."""
    data      = request.get_json(silent=True) or {}
    direction = data.get("direction", "")
    result    = maze.move(direction)
    return jsonify({"result": result, "maze": maze.state()})


@app.route("/maze/new", methods=["POST"])
def maze_new():
    """Generate a completely new randomised maze."""
    data = request.get_json(silent=True) or {}
    rows = int(data.get("rows", 11))
    cols = int(data.get("cols", 11))
    maze.generate(rows, cols)
    return jsonify({"maze": maze.state()})


@app.route("/maze/state", methods=["GET"])
def maze_state():
    """Return current maze state as JSON."""
    return jsonify({"maze": maze.state()})


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
