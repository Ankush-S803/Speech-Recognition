/**
 * maze.js — Frontend logic for the VoiceMaze application
 *
 * Responsibilities:
 *  • Fetch and render the maze grid on a <canvas>
 *  • Handle "Start Recording" button → MediaRecorder API → POST /predict
 *  • Handle "New Maze" button → POST /maze/new
 *  • Handle keyboard D-pad clicks → POST /maze/move
 *  • Update probability bars, command display, and activity log
 *  • Show win overlay when the mouse reaches the exit
 */

"use strict";

// ─── Constants ───────────────────────────────────────────────────────────────
const CELL_SIZE     = 28;   // pixels per cell on canvas
const COMMANDS      = ["left", "right", "up", "down", "yes", "no"];
const MAX_LOG_ITEMS = 20;   // activity log ceiling

// Colour palette (matches CSS)
const COLOURS = {
  wall:      "#1a2036",
  path:      "#0a0e1a",
  mouse:     "#f59e0b",
  exit:      "#22c55e",
  exitBg:    "rgba(34,197,94,0.15)",
  mouseShadow: "rgba(245,158,11,0.6)",
  grid:      "rgba(255,255,255,0.03)",
};

// ─── DOM references ───────────────────────────────────────────────────────────
const canvas        = document.getElementById("maze-canvas");
const ctx           = canvas.getContext("2d");
const btnRecord     = document.getElementById("btn-record");
const btnRecordLbl  = document.getElementById("btn-record-label");
const btnNewMaze    = document.getElementById("btn-new-maze");
const btnWinNew     = document.getElementById("btn-win-new");
const statusDot     = document.getElementById("status-dot");
const statusText    = document.getElementById("status-text");
const cmdWord       = document.getElementById("cmd-word");
const cmdConf       = document.getElementById("cmd-conf");
const demoBadge     = document.getElementById("demo-badge");
const stepCount     = document.getElementById("step-count");
const winOverlay    = document.getElementById("win-overlay");
const winSteps      = document.getElementById("win-steps");
const activityLog   = document.getElementById("activity-log");

// ─── State ────────────────────────────────────────────────────────────────────
let mazeState    = null;
let isRecording  = false;
let mediaRecorder = null;
let audioChunks  = [];

// ─── Maze rendering ───────────────────────────────────────────────────────────
function renderMaze(state) {
  if (!state) return;
  mazeState = state;

  const { grid, rows, cols, mouse, exit: exitCell, won, steps } = state;
  const w = cols * CELL_SIZE;
  const h = rows * CELL_SIZE;

  canvas.width  = w;
  canvas.height = h;
  stepCount.textContent = steps;

  // ── Draw cells ──────────────────────────────────────────────────────────
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * CELL_SIZE;
      const y = r * CELL_SIZE;
      const isWall  = grid[r][c] === 1;
      const isExit  = r === exitCell[0] && c === exitCell[1];
      const isMouse = r === mouse[0]    && c === mouse[1];

      if (isWall) {
        // Wall: dark slate with subtle gradient
        const grad = ctx.createLinearGradient(x, y, x + CELL_SIZE, y + CELL_SIZE);
        grad.addColorStop(0, "#1a2036");
        grad.addColorStop(1, "#111827");
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Inner edge highlight
        ctx.strokeStyle = "rgba(255,255,255,0.04)";
        ctx.lineWidth   = 0.5;
        ctx.strokeRect(x + 0.5, y + 0.5, CELL_SIZE - 1, CELL_SIZE - 1);
      } else {
        // Open path
        ctx.fillStyle = COLOURS.path;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Subtle grid line
        ctx.strokeStyle = COLOURS.grid;
        ctx.lineWidth   = 0.5;
        ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
      }

      // Exit cell
      if (isExit && !isMouse) {
        ctx.fillStyle = COLOURS.exitBg;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Door emoji
        ctx.font      = `${CELL_SIZE * 0.65}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("🚪", x + CELL_SIZE / 2, y + CELL_SIZE / 2);
      }
    }
  }

  // ── Draw mouse ───────────────────────────────────────────────────────────
  const [mr, mc] = mouse;
  const mx = mc * CELL_SIZE + CELL_SIZE / 2;
  const my = mr * CELL_SIZE + CELL_SIZE / 2;
  const r  = CELL_SIZE * 0.36;

  // Glow
  ctx.shadowColor = COLOURS.mouseShadow;
  ctx.shadowBlur  = 14;

  ctx.beginPath();
  ctx.arc(mx, my, r, 0, Math.PI * 2);
  const grad = ctx.createRadialGradient(mx - r * 0.3, my - r * 0.3, 0, mx, my, r);
  grad.addColorStop(0, "#fcd34d");
  grad.addColorStop(1, "#d97706");
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.shadowBlur = 0;

  // Mouse face emoji
  ctx.font         = `${CELL_SIZE * 0.6}px sans-serif`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("🐭", mx, my);

  // ── Win state ────────────────────────────────────────────────────────────
  if (won) {
    showWinOverlay(steps);
  } else {
    hideWinOverlay();
  }
}

function showWinOverlay(steps) {
  winSteps.textContent = steps;
  winOverlay.classList.remove("hidden");
}
function hideWinOverlay() {
  winOverlay.classList.add("hidden");
}

// ─── Fetch initial maze ───────────────────────────────────────────────────────
async function fetchMazeState() {
  try {
    const res  = await fetch("/maze/state");
    const data = await res.json();
    renderMaze(data.maze);
  } catch (e) {
    logActivity("Failed to load maze", "error");
  }
}

// ─── New Maze ─────────────────────────────────────────────────────────────────
async function newMaze() {
  try {
    btnNewMaze.disabled = true;
    const res  = await fetch("/maze/new", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: 11, cols: 11 }) });
    const data = await res.json();
    renderMaze(data.maze);
    logActivity("New maze generated", "");
  } catch (e) {
    logActivity("Failed to generate maze", "error");
  } finally {
    btnNewMaze.disabled = false;
  }
}

// ─── Keyboard move ────────────────────────────────────────────────────────────
async function sendMove(direction) {
  try {
    const res  = await fetch("/maze/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction }),
    });
    const data = await res.json();
    renderMaze(data.maze);
    const result = data.result;
    if (result === "moved") {
      logActivity(`Moved ${direction}`, "cmd");
    } else if (result === "blocked") {
      logActivity(`Blocked: ${direction}`, "error");
    } else if (result === "won") {
      logActivity("🏆 Maze solved!", "success");
    }
  } catch (e) {
    logActivity("Move error", "error");
  }
}

// ─── Speech recording ─────────────────────────────────────────────────────────
function setStatus(type, text) {
  statusDot.className = `status-dot ${type}`;
  statusText.textContent = text;
}

async function startRecording() {
  if (isRecording) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks   = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      stream.getTracks().forEach(t => t.stop());
      await sendAudioToServer(audioBlob);
    };

    mediaRecorder.start();
    isRecording = true;

    btnRecord.classList.add("recording");
    btnRecordLbl.textContent = "Listening…";
    setStatus("recording", "Recording — speak now…");
    logActivity("Recording started", "");

    // Auto-stop after 1.5 s
    setTimeout(() => {
      if (isRecording) stopRecording();
    }, 1500);

  } catch (err) {
    logActivity("Microphone access denied", "error");
    setStatus("error", "Mic permission denied");
    alert("Please allow microphone access to use voice commands.");
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  mediaRecorder.stop();
  isRecording = false;
  btnRecord.classList.remove("recording");
  btnRecord.classList.add("processing");
  btnRecordLbl.textContent = "Processing…";
  setStatus("processing", "Processing audio…");
}

async function sendAudioToServer(blob) {
  try {
    const form = new FormData();
    form.append("audio", blob, "command.wav");

    const res  = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    // ── Update command display ────────────────────────────────────────────
    const cmd  = data.command    || "unknown";
    const conf = data.confidence || 0;
    const isDemo = data.demo_mode;

    cmdWord.textContent = cmd;
    cmdWord.classList.remove("pop");
    void cmdWord.offsetWidth;   // force reflow
    cmdWord.classList.add("pop");

    cmdConf.textContent = `Confidence: ${(conf * 100).toFixed(1)}%`;
    demoBadge.classList.toggle("hidden", !isDemo);

    // ── Update probability bars ───────────────────────────────────────────
    if (data.all_probs) {
      updateProbBars(data.all_probs);
    }

    // ── Re-render maze if server moved the mouse ──────────────────────────
    if (data.maze) {
      renderMaze(data.maze);
    }

    logActivity(`Heard: "${cmd}" (${(conf * 100).toFixed(0)}%)`, "cmd");
    setStatus("success", `Command: "${cmd}"`);

  } catch (err) {
    logActivity(`Prediction error: ${err.message}`, "error");
    setStatus("error", "Error processing audio");
  } finally {
    btnRecord.classList.remove("processing");
    btnRecordLbl.textContent = "Start Recording";
  }
}

// ─── Probability bars ─────────────────────────────────────────────────────────
function updateProbBars(probs) {
  for (const cmd of COMMANDS) {
    const row  = document.querySelector(`.prob-row[data-cmd="${cmd}"]`);
    if (!row) continue;
    const fill = row.querySelector(".prob-fill");
    const pct  = row.querySelector(".prob-pct");
    const val  = (probs[cmd] || 0) * 100;
    fill.style.width      = `${val.toFixed(1)}%`;
    pct.textContent       = `${val.toFixed(0)}%`;
  }
}

// ─── Activity log ─────────────────────────────────────────────────────────────
function logActivity(msg, type = "") {
  const li     = document.createElement("li");
  li.className = type;
  const now    = new Date();
  const time   = `${now.getHours().toString().padStart(2,"0")}:${now.getMinutes().toString().padStart(2,"0")}:${now.getSeconds().toString().padStart(2,"0")}`;
  li.innerHTML = `<span class="time">${time}</span>${msg}`;
  activityLog.prepend(li);

  // Trim to max items
  while (activityLog.children.length > MAX_LOG_ITEMS) {
    activityLog.removeChild(activityLog.lastChild);
  }
}

// ─── Keyboard shortcut (arrow keys) ──────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  const map = {
    ArrowUp:    "up",
    ArrowDown:  "down",
    ArrowLeft:  "left",
    ArrowRight: "right",
  };
  if (map[e.key]) {
    e.preventDefault();
    sendMove(map[e.key]);
  }
  if (e.code === "Space") {
    e.preventDefault();
    if (!isRecording) startRecording();
  }
});

// ─── Event listeners ──────────────────────────────────────────────────────────
btnRecord.addEventListener("click", () => {
  if (isRecording) stopRecording();
  else startRecording();
});

btnNewMaze.addEventListener("click", newMaze);
btnWinNew.addEventListener("click", newMaze);

document.querySelectorAll(".kb-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const dir = btn.dataset.dir;
    if (dir) sendMove(dir);
  });
});

// ─── Initialise ───────────────────────────────────────────────────────────────
fetchMazeState();
logActivity("Application loaded", "success");
setStatus("idle", "Ready — click mic or press Space");
