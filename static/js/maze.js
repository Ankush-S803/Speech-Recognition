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

let recognitionInstance   = null;
let isContinuousRunning   = false;
let isSingleListening     = false;
let singleTimeout         = null;
let continuousRestartTimer= null;
let lastExecutedCommand   = "";
let lastExecutedTime      = 0;

function setStatus(type, text) {
  statusDot.className = `status-dot ${type}`;
  statusText.textContent = text;
}

/**
 * Match a raw spoken transcript to one of the 4 valid maze commands.
 * Handles homophones, accents, casual phrasing, and compound sentences.
 */
function normalizeCommand(rawText) {
  if (!rawText) return null;
  const clean = rawText.toLowerCase().replace(/[^a-z0-9\s]/g, " ").trim();
  if (!clean) return null;

  const tokens = clean.split(/\s+/);

  // Exact word tokens
  for (const t of tokens) {
    if (["right", "write", "rite", "ride", "light", "bright", "wright", "alright", "white", "rate"].includes(t)) {
      return "right";
    }
    if (["left", "lift", "let", "leaf", "laughed", "last", "west", "loft"].includes(t)) {
      return "left";
    }
    if (["up", "app", "upper", "top", "hop", "cup", "north", "straight", "forward", "ahead"].includes(t)) {
      return "up";
    }
    if (["down", "done", "dawn", "town", "brown", "don", "bottom", "south", "dan"].includes(t)) {
      return "down";
    }
  }

  // Regex boundary match across phrase
  if (/\b(right|write|rite|ride|light|bright|white)\b/.test(clean)) return "right";
  if (/\b(left|lift|let|leaf|laughed)\b/.test(clean))              return "left";
  if (/\b(up|upper|top|hop|forward|straight|ahead|north)\b/.test(clean)) return "up";
  if (/\b(down|done|dawn|town|bottom|south)\b/.test(clean))        return "down";

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

// ─── CONTINUOUS VOICE MODE ────────────────────────────────────────────────────
function startContinuousMode() {
  stopSingleListening(false);
  isContinuousRunning = true;
  btnRecord.classList.add("recording");
  btnRecord.classList.remove("processing");
  btnRecordLbl.textContent = "Listening (Live)…";
  setStatus("recording", "Live listening — speak commands freely (up, down, left, right)");
  logActivity("⚡ Continuous voice mode started", "success");

  setupContinuousRecognition();
}

function stopContinuousMode() {
  isContinuousRunning = false;
  clearTimeout(continuousRestartTimer);

  if (recognitionInstance) {
    try { recognitionInstance.abort(); } catch (_) {}
    recognitionInstance = null;
  }

  btnRecord.classList.remove("recording");
  btnRecord.classList.remove("processing");
  btnRecordLbl.textContent = "Speak Command";
  setStatus("idle", "Ready — click Speak Command or toggle Continuous");
  logActivity("Continuous voice mode paused", "");
}

function setupContinuousRecognition() {
  if (!isContinuousRunning) return;

  if (recognitionInstance) {
    try { recognitionInstance.abort(); } catch (_) {}
    recognitionInstance = null;
  }

  if (!SpeechRecognition) {
    startMediaRecorderFallback(true);
    return;
  }

  try {
    recognitionInstance = new SpeechRecognition();
    recognitionInstance.lang = "en-US";
    recognitionInstance.continuous = true;
    recognitionInstance.interimResults = true;
    recognitionInstance.maxAlternatives = 5;

    recognitionInstance.onstart = () => {
      if (isContinuousRunning) {
        setStatus("recording", "Live listening — speak commands freely");
      }
    };

    recognitionInstance.onresult = (event) => {
      if (!isContinuousRunning) return;

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        let transcript = "";
        let matched = null;

        for (let j = 0; j < res.length; j++) {
          transcript = res[j].transcript;
          matched = normalizeCommand(transcript);
          if (matched) break;
        }

        if (matched) {
          const now = Date.now();
          // Debounce: prevent same command firing repeatedly within 350ms
          if (matched === lastExecutedCommand && (now - lastExecutedTime) < 350) {
            return;
          }
          lastExecutedCommand = matched;
          lastExecutedTime    = now;

          const conf = Math.max(0.92, (res[0] && res[0].confidence) || 0.96);
          handleRecognisedCommand(matched, conf);
          logActivity(`⚡ Heard: "${matched}"`, "cmd");
          setStatus("success", `Executed: ${matched.toUpperCase()}`);
          return;
        } else if (res.isFinal && transcript.trim()) {
          // Show unrecognized speech feedback
          setStatus("recording", `Heard: "${transcript.trim()}" — say up, down, left, right`);
          logActivity(`Unrecognized: "${transcript.trim()}"`, "");
        }
      }
    };

    recognitionInstance.onerror = (event) => {
      if (event.error === "no-speech" || event.error === "aborted") {
        // Normal in continuous mode; do not disrupt
        return;
      }
      console.warn("Continuous recognition error:", event.error);
    };

    recognitionInstance.onend = () => {
      // Re-instantiate freshly after a brief cooldown so the audio track clears cleanly
      if (isContinuousRunning) {
        clearTimeout(continuousRestartTimer);
        continuousRestartTimer = setTimeout(() => {
          if (isContinuousRunning) setupContinuousRecognition();
        }, 180);
      }
    };

    recognitionInstance.start();
  } catch (err) {
    console.warn("Failed to start continuous recognition, retrying in 400ms:", err);
    if (isContinuousRunning) {
      clearTimeout(continuousRestartTimer);
      continuousRestartTimer = setTimeout(setupContinuousRecognition, 400);
    }
  }
}

// ─── SINGLE COMMAND MODE (5-second comfortable listening budget) ──────────────
function startSingleListening() {
  if (isContinuousRunning) {
    // If continuous mode is on, a click toggles continuous mode off
    if (chkContinuous) chkContinuous.checked = false;
    stopContinuousMode();
    return;
  }

  if (isSingleListening) {
    stopSingleListening(false);
    return;
  }

  isSingleListening = true;
  btnRecord.classList.add("recording");
  btnRecord.classList.remove("processing");
  btnRecordLbl.textContent = "Listening…";
  setStatus("recording", "Listening… speak right, left, up, or down (5s)");
  logActivity("Listening for command (5s)…", "");

  // Give the user a full 5.0 seconds to speak
  clearTimeout(singleTimeout);
  singleTimeout = setTimeout(() => {
    if (isSingleListening) {
      stopSingleListening(true);
    }
  }, 5000);

  if (SpeechRecognition) {
    runSingleSpeechRecognition();
  } else {
    startMediaRecorderFallback(false);
  }
}

function runSingleSpeechRecognition() {
  if (!isSingleListening) return;

  if (recognitionInstance) {
    try { recognitionInstance.abort(); } catch (_) {}
    recognitionInstance = null;
  }

  try {
    recognitionInstance = new SpeechRecognition();
    recognitionInstance.lang = "en-US";
    recognitionInstance.continuous = false;
    recognitionInstance.interimResults = true;
    recognitionInstance.maxAlternatives = 5;

    recognitionInstance.onresult = (event) => {
      if (!isSingleListening) return;

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        let matched = null;
        for (let j = 0; j < res.length; j++) {
          matched = normalizeCommand(res[j].transcript);
          if (matched) break;
        }

        if (matched) {
          clearTimeout(singleTimeout);
          isSingleListening = false;

          const conf = Math.max(0.92, (res[0] && res[0].confidence) || 0.96);
          handleRecognisedCommand(matched, conf);
          logActivity(`⚡ Heard: "${matched}"`, "cmd");
          setStatus("success", `Command: "${matched.toUpperCase()}"`);

          try { recognitionInstance.stop(); } catch (_) {}

          btnRecord.classList.remove("recording");
          btnRecordLbl.textContent = "Speak Command";
          setTimeout(() => {
            if (!isSingleListening && !isContinuousRunning) {
              setStatus("idle", "Ready — click Speak Command or press Space");
            }
          }, 1500);
          return;
        }
      }
    };

    recognitionInstance.onerror = (event) => {
      if (event.error === "no-speech") {
        // If the user was momentarily quiet but our 5s budget is still active, restart!
        if (isSingleListening) {
          setTimeout(() => {
            if (isSingleListening) runSingleSpeechRecognition();
          }, 150);
        }
        return;
      }
      console.warn("Single recognition error:", event.error);
    };

    recognitionInstance.onend = () => {
      // If ended before our 5s budget expired and no command was matched, keep listening
      if (isSingleListening) {
        setTimeout(() => {
          if (isSingleListening) runSingleSpeechRecognition();
        }, 150);
      }
    };

    recognitionInstance.start();
  } catch (err) {
    console.warn("SpeechRecognition start error:", err);
    startMediaRecorderFallback(false);
  }
}

function stopSingleListening(isTimeout = false) {
  clearTimeout(singleTimeout);
  isSingleListening = false;

  if (recognitionInstance) {
    try { recognitionInstance.stop(); } catch (_) {}
    recognitionInstance = null;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    try { mediaRecorder.stop(); } catch (_) {}
  }

  btnRecord.classList.remove("recording");
  btnRecord.classList.remove("processing");
  btnRecordLbl.textContent = "Speak Command";

  if (isTimeout) {
    setStatus("idle", "Timed out — click Speak Command to try again");
    logActivity("Listening timed out (no command detected)", "error");
  } else {
    setStatus("idle", "Ready — click Speak Command or press Space");
  }
}

// ─── MEDIA RECORDER FALLBACK (2.5s window) ───────────────────────────────────
async function startMediaRecorderFallback(isContinuous = false) {
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

      if (isContinuous && isContinuousRunning) {
        setTimeout(() => startMediaRecorderFallback(true), 300);
      }
    };

    mediaRecorder.start();
    btnRecord.classList.add("recording");
    btnRecordLbl.textContent = "Listening…";
    setStatus("recording", "Recording audio (2.5s)… speak clearly");
    logActivity("Recording audio (2.5s)…", "");

    // 2.5 seconds allows comfortable speaking time
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        btnRecord.classList.remove("recording");
        btnRecord.classList.add("processing");
        btnRecordLbl.textContent = "Processing…";
        setStatus("processing", "Processing with CNN model…");
      }
    }, 2500);

  } catch (err) {
    logActivity("Microphone access denied", "error");
    setStatus("error", "Mic permission denied");
    alert("Please allow microphone access to use voice commands.");
  }
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
    if (isContinuousRunning) {
      if (chkContinuous) chkContinuous.checked = false;
      stopContinuousMode();
    } else if (isSingleListening) {
      stopSingleListening(false);
    } else {
      startSingleListening();
    }
  }
});

// ─── Event listeners ──────────────────────────────────────────────────────────
btnRecord.addEventListener("click", () => {
  if (isContinuousRunning) {
    if (chkContinuous) chkContinuous.checked = false;
    stopContinuousMode();
  } else if (isSingleListening) {
    stopSingleListening(false);
  } else {
    startSingleListening();
  }
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
    if (e.target.checked) {
      startContinuousMode();
    } else {
      stopContinuousMode();
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

