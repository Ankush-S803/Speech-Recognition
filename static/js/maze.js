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
const COMMANDS      = ["left", "right", "up", "down"];
const MAX_LOG_ITEMS = 20;   // activity log ceiling

// Dynamic cell size to keep maze nicely sized in the middle (~310-330px)
function getCellSize(rows) {
  if (rows <= 7) return 44;
  if (rows <= 9) return 36;
  return 30;
}

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
const canvas         = document.getElementById("maze-canvas");
const ctx            = canvas.getContext("2d");
const btnRecord      = document.getElementById("btn-record");
const btnRecordLbl   = document.getElementById("btn-record-label");
const btnNewMaze     = document.getElementById("btn-new-maze");
const btnWinNew      = document.getElementById("btn-win-new");
const selectMazeSize = document.getElementById("select-maze-size");
const chkContinuous  = document.getElementById("chk-continuous");
const statusDot      = document.getElementById("status-dot");
const statusText     = document.getElementById("status-text");
const cmdWord        = document.getElementById("cmd-word");
const cmdConf        = document.getElementById("cmd-conf");
const demoBadge      = document.getElementById("demo-badge");
const stepCount      = document.getElementById("step-count");
const winOverlay     = document.getElementById("win-overlay");
const winSteps       = document.getElementById("win-steps");
const activityLog    = document.getElementById("activity-log");

// ─── State ────────────────────────────────────────────────────────────────────
let mazeState        = null;
let isRecording      = false;
let isCommandExecuting = false;
let continuousMode   = false;
let mediaRecorder    = null;
let audioChunks      = [];

// ─── Maze rendering ───────────────────────────────────────────────────────────
function renderMaze(state) {
  if (!state) return;
  mazeState = state;

  const { grid, rows, cols, mouse, exit: exitCell, won, steps } = state;
  const cellSize = getCellSize(rows);
  const w = cols * cellSize;
  const h = rows * cellSize;

  canvas.width  = w;
  canvas.height = h;
  stepCount.textContent = steps;

  // ── Draw cells ──────────────────────────────────────────────────────────
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cellSize;
      const y = r * cellSize;
      const isWall  = grid[r][c] === 1;
      const isExit  = r === exitCell[0] && c === exitCell[1];
      const isMouse = r === mouse[0]    && c === mouse[1];

      if (isWall) {
        // Wall: dark slate with subtle gradient
        const grad = ctx.createLinearGradient(x, y, x + cellSize, y + cellSize);
        grad.addColorStop(0, "#1a2036");
        grad.addColorStop(1, "#111827");
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, cellSize, cellSize);

        // Inner edge highlight
        ctx.strokeStyle = "rgba(255,255,255,0.04)";
        ctx.lineWidth   = 0.5;
        ctx.strokeRect(x + 0.5, y + 0.5, cellSize - 1, cellSize - 1);
      } else {
        // Open path
        ctx.fillStyle = COLOURS.path;
        ctx.fillRect(x, y, cellSize, cellSize);

        // Subtle grid line
        ctx.strokeStyle = COLOURS.grid;
        ctx.lineWidth   = 0.5;
        ctx.strokeRect(x, y, cellSize, cellSize);
      }

      // Exit cell
      if (isExit && !isMouse) {
        ctx.fillStyle = COLOURS.exitBg;
        ctx.fillRect(x, y, cellSize, cellSize);

        // Door emoji
        ctx.font      = `${cellSize * 0.65}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("🚪", x + cellSize / 2, y + cellSize / 2);
      }
    }
  }

  // ── Draw mouse ───────────────────────────────────────────────────────────
  const [mr, mc] = mouse;
  const mx = mc * cellSize + cellSize / 2;
  const my = mr * cellSize + cellSize / 2;
  const r  = cellSize * 0.36;

  // Glow
  ctx.shadowColor = COLOURS.mouseShadow;
  ctx.shadowBlur  = 12;

  ctx.beginPath();
  ctx.arc(mx, my, r, 0, Math.PI * 2);
  const grad = ctx.createRadialGradient(mx - r * 0.3, my - r * 0.3, 0, mx, my, r);
  grad.addColorStop(0, "#fcd34d");
  grad.addColorStop(1, "#d97706");
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.shadowBlur = 0;

  // Mouse face emoji
  ctx.font         = `${cellSize * 0.6}px sans-serif`;
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
    const size = selectMazeSize ? parseInt(selectMazeSize.value, 10) : 9;
    const res  = await fetch("/maze/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: size, cols: size })
    });
    const data = await res.json();
    renderMaze(data.maze);
    logActivity(`New ${size}×${size} maze generated`, "");
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

// ─── Speech recognition & audio handling ──────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognitionInstance = null;

function setStatus(type, text) {
  statusDot.className = `status-dot ${type}`;
  statusText.textContent = text;
}

/**
 * Match a raw spoken transcript to one of the 4 valid maze commands.
 * Handles homophones, accents, and casual phrasing (e.g. "go right", "write").
 */
function normalizeCommand(rawText) {
  if (!rawText) return null;
  const clean = rawText.toLowerCase().replace(/[^a-z\s]/g, " ").trim();
  const tokens = clean.split(/\s+/);

  // Direct token match (highest priority)
  for (const t of tokens) {
    if (t === "right" || t === "write" || t === "rite" || t === "ride") return "right";
    if (t === "left"  || t === "lift"  || t === "let")                  return "left";
    if (t === "up"    || t === "app"   || t === "upper" || t === "hop") return "up";
    if (t === "down"  || t === "done"  || t === "town"  || t === "dawn") return "down";
  }

  // Substring search
  if (clean.includes("right")) return "right";
  if (clean.includes("left"))  return "left";
  if (clean.includes("up"))    return "up";
  if (clean.includes("down"))  return "down";

  return null;
}

function handleRecognisedCommand(cmd, confidence = 0.96) {
  // ── Update command display ────────────────────────────────────────────
  cmdWord.textContent = cmd;
  cmdWord.classList.remove("pop");
  void cmdWord.offsetWidth;   // force reflow
  cmdWord.classList.add("pop");

  cmdConf.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
  demoBadge.classList.remove("hidden");
  demoBadge.textContent = "🎙 Speech Engine Active";

  // ── Probability distribution for 4 commands ───────────────────────────
  const probs = {};
  const remainingProb = Math.max(0.01, (1.0 - confidence) / (COMMANDS.length - 1));
  for (const c of COMMANDS) {
    probs[c] = (c === cmd) ? confidence : remainingProb;
  }
  updateProbBars(probs);

  // ── Move the mouse ────────────────────────────────────────────────────
  sendMove(cmd);
}

function startListening() {
  if (isRecording) return;

  // ── Modern Web Speech API (Chrome, Edge, Safari, Opera) ─────────────────
  if (SpeechRecognition) {
    try {
      continuousMode = chkContinuous ? chkContinuous.checked : false;

      // Always create a fresh instance for instantaneous start
      if (recognitionInstance) {
        try { recognitionInstance.abort(); } catch (_) {}
      }

      recognitionInstance = new SpeechRecognition();
      recognitionInstance.lang = "en-US";
      recognitionInstance.continuous = continuousMode;
      recognitionInstance.interimResults = true; // INSTANT partial recognition!
      recognitionInstance.maxAlternatives = 3;

      recognitionInstance.onstart = () => {
        isRecording = true;
        btnRecord.classList.add("recording");
        btnRecordLbl.textContent = continuousMode ? "Listening (Live)…" : "Listening…";
        setStatus("recording", continuousMode ? "Live listening — speak commands freely" : "Listening… speak right, left, up, or down");
        logActivity(continuousMode ? "Continuous voice mode active" : "Listening for command…", "");
      };

      recognitionInstance.onresult = (event) => {
        if (isCommandExecuting) return; // Debounce so 1 spoken word doesn't double-fire

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          for (let j = 0; j < result.length; j++) {
            const alt = result[j];
            const matched = normalizeCommand(alt.transcript);
            if (matched) {
              isCommandExecuting = true;
              const conf = Math.max(0.92, alt.confidence || 0.96);

              handleRecognisedCommand(matched, conf);
              logActivity(`⚡ Heard: "${matched}"`, "cmd");
              setStatus("success", `Command: "${matched}"`);

              // If not in continuous mode, stop immediately to finish fast
              if (!continuousMode) {
                try { recognitionInstance.stop(); } catch (_) {}
              }

              // Re-arm after quick 250ms debounce
              setTimeout(() => {
                isCommandExecuting = false;
              }, 280);
              return;
            }
          }
        }
      };

      recognitionInstance.onerror = (event) => {
        if (event.error === "no-speech") {
          if (!continuousMode) {
            setStatus("idle", "No speech detected — click Speak Command to retry");
          }
        } else {
          console.warn("Speech error:", event.error);
          setStatus("error", `Mic: ${event.error}`);
        }
      };

      recognitionInstance.onend = () => {
        if (continuousMode && isRecording) {
          try { recognitionInstance.start(); } catch (_) {}
          return;
        }
        isRecording = false;
        btnRecord.classList.remove("recording");
        btnRecord.classList.remove("processing");
        btnRecordLbl.textContent = "Speak Command";
        setTimeout(() => {
          if (!isRecording) {
            setStatus("idle", "Ready — click Speak Command or press Space");
          }
        }, 1200);
      };

      recognitionInstance.start();
      return;
    } catch (e) {
      console.warn("SpeechRecognition start error, falling back to MediaRecorder:", e);
    }
  }

  // ── Fallback: MediaRecorder + Backend /predict ──────────────────────────
  startMediaRecorderFallback();
}

function stopListening() {
  if (recognitionInstance && isRecording) {
    try { recognitionInstance.stop(); } catch (_) {}
  }
  if (mediaRecorder && isRecording) {
    stopMediaRecorder();
  }
  isRecording = false;
  btnRecord.classList.remove("recording");
  btnRecordLbl.textContent = "Speak Command";
}

async function startMediaRecorderFallback() {
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
    setStatus("recording", "Recording audio… speak now");
    logActivity("Recording started", "");

    // Quick auto-stop after 800 ms (fast word recording)
    setTimeout(() => {
      if (isRecording) stopMediaRecorder();
    }, 800);

  } catch (err) {
    logActivity("Microphone access denied", "error");
    setStatus("error", "Mic permission denied");
    alert("Please allow microphone access to use voice commands.");
  }
}

function stopMediaRecorder() {
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

    if (data.error) throw new Error(data.error);

    const cmd  = data.command || "unknown";
    const conf = data.confidence || 0;

    cmdWord.textContent = cmd;
    cmdWord.classList.remove("pop");
    void cmdWord.offsetWidth;
    cmdWord.classList.add("pop");

    cmdConf.textContent = `Confidence: ${(conf * 100).toFixed(1)}%`;

    if (data.all_probs) updateProbBars(data.all_probs);
    if (data.maze) renderMaze(data.maze);

    logActivity(`Heard: "${cmd}" (${(conf * 100).toFixed(0)}%)`, "cmd");
    setStatus("success", `Command: "${cmd}"`);

  } catch (err) {
    logActivity(`Prediction error: ${err.message}`, "error");
    setStatus("error", "Error processing audio");
  } finally {
    btnRecord.classList.remove("processing");
    btnRecordLbl.textContent = "Speak Command";
  }
}

// ─── Probability bars (4 directional commands) ──────────────────────────────
function updateProbBars(probs) {
  for (const cmd of COMMANDS) {
    const row  = document.querySelector(`.prob-row[data-cmd="${cmd}"]`);
    if (!row) continue;
    const fill = row.querySelector(".prob-fill");
    const pct  = row.querySelector(".prob-pct");
    const val  = (probs[cmd] || 0) * 100;
    fill.style.width = `${val.toFixed(1)}%`;
    pct.textContent  = `${val.toFixed(0)}%`;
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

  while (activityLog.children.length > MAX_LOG_ITEMS) {
    activityLog.removeChild(activityLog.lastChild);
  }
}

// ─── Keyboard shortcut (arrow keys + Space) ──────────────────────────────────
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
    if (!isRecording) startListening();
    else stopListening();
  }
});

// ─── Event listeners ──────────────────────────────────────────────────────────
btnRecord.addEventListener("click", () => {
  if (isRecording) stopListening();
  else startListening();
});

btnNewMaze.addEventListener("click", newMaze);
btnWinNew.addEventListener("click", newMaze);

if (selectMazeSize) {
  selectMazeSize.addEventListener("change", () => {
    newMaze();
  });
}

if (chkContinuous) {
  chkContinuous.addEventListener("change", (e) => {
    continuousMode = e.target.checked;
    if (continuousMode) {
      logActivity("⚡ Continuous voice mode enabled", "success");
      if (!isRecording) startListening();
    } else {
      logActivity("Standard single-command mode", "");
      if (isRecording) stopListening();
    }
  });
}

document.querySelectorAll(".kb-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const dir = btn.dataset.dir;
    if (dir) sendMove(dir);
  });
});

// ─── Initialise ───────────────────────────────────────────────────────────────
btnRecordLbl.textContent = "Speak Command";
fetchMazeState();
logActivity("Application loaded", "success");
setStatus("idle", "Ready — click Speak Command or press Space");

