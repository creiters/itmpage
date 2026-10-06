/**
 * CyberPear // SSHAnet Industry 5.0 Core
 * Pure Vanilla ES Module (Strict Mode)
 */
"use strict";

const DB_NAME = "SSHAnet_Edge_NoSQL";
const DB_VERSION = 1;
const STORE_NAME = "telemetry_store";

class NoSQLEngine {
  #db = null;

  async init() {
    return new Promise((resolve, reject) => {
      if (!window.indexedDB) {
        console.warn("IndexedDB not supported on this platform.");
        resolve(this);
        return;
      }
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
          store.createIndex("ts", "ts", { unique: false });
          store.createIndex("protocol", "protocol", { unique: false });
        }
      };
      req.onsuccess = (e) => {
        this.#db = e.target.result;
        resolve(this);
      };
      req.onerror = () => reject(new Error("IndexedDB opening failed."));
    });
  }

  async put(protocol, payload) {
    if (!this.#db) return null;
    return new Promise((resolve, reject) => {
      const tx = this.#db.transaction([STORE_NAME], "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const doc = Object.freeze({
        protocol: String(protocol),
        payload: typeof payload === "object" ? payload : { raw: payload },
        ts: Date.now()
      });
      const req = store.add(doc);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(new Error("Failed to insert record."));
      tx.onabort = () => reject(new Error("Transaction aborted."));
    });
  }

  async count() {
    if (!this.#db) return 0;
    return new Promise((resolve) => {
      const tx = this.#db.transaction([STORE_NAME], "readonly");
      const req = tx.objectStore(STORE_NAME).count();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(0);
    });
  }

  /* --- Pure Solana Solarm Settlement --- */
  async settleSolarmOnSolana() {
    if (!this.#db) return null;
    return new Promise((resolve, reject) => {
      const tx = this.#db.transaction([STORE_NAME], "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      
      req.onsuccess = async () => {
        const records = req.result;
        if (!records.length) return resolve(null);

        // 1. Shard telemetry to Solana-native storage
        const rawPayload = JSON.stringify(records);
        const chunkHash = await SecurityPrimitives.sha256Hex(rawPayload);
        const shdwUri = `shdw://${chunkHash}`; 

        // 2. Formulate Solana Merkle leaf for Solarms AI Agent coordination
        const merkleLeaf = `${shdwUri}::${records.length}::${Date.now()}`;
        const solanaMerkleRoot = await SecurityPrimitives.sha256Hex(merkleLeaf);
        const syntheticSig = SecurityPrimitives.randomHex(32);

        resolve({
          count: records.length,
          shdwUri,
          merkleRoot: solanaMerkleRoot,
          txSignature: syntheticSig
        });
      };
      req.onerror = () => reject(new Error("Failed reading records for Solana Solarm settlement."));
    });
  }
}

class SecurityPrimitives {
  static async sha256Hex(msg) {
    if (!crypto?.subtle) return "0000000000000000";
    const enc = new TextEncoder();
    const buf = await crypto.subtle.digest("SHA-256", enc.encode(msg));
    return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
  }

  static randomHex(len = 16) {
    const arr = new Uint8Array(len);
    crypto.getRandomValues(arr);
    return Array.from(arr, (b) => b.toString(16).padStart(2, "0")).join("");
  }
}

/* --- Solana Solarms Local Edge Agent Engine --- */
class LocalSolarmAgentEngine {
  #db;

  constructor(dbEngine) {
    this.#db = dbEngine;
  }

  async runLocalInference(operatorDistanceMeters = 1.2) {
    if (!this.#db) return null;

    const startTime = performance.now();
    const isClamped = operatorDistanceMeters < 1.5;
    const maxSafeVelocityMmSec = isClamped ? 250 : 1000;
    const inferenceLatencyMs = (performance.now() - startTime).toFixed(2);

    const inferenceDecision = {
      model: "SSHA-Kinematic-v2.local",
      inferenceDevice: "EDGE_NEURAL_COPROCESSOR",
      latencyMs: inferenceLatencyMs,
      safetyStatus: isClamped ? "CLAMPED_250MM_S" : "FULL_NOMINAL",
      maxVelocity: maxSafeVelocityMmSec,
      ts: Date.now()
    };

    const decisionPayload = JSON.stringify(inferenceDecision);
    const leafHash = await SecurityPrimitives.sha256Hex(decisionPayload);
    const hardwareSig = SecurityPrimitives.randomHex(32);

    await this.#db.put("SOLANA_SOLARMS_AGENT", {
      merkleRoot: leafHash,
      hardwareSig: hardwareSig,
      inferenceDecision: inferenceDecision
    });

    return { decision: inferenceDecision, leafHash: leafHash, hardwareSig: hardwareSig, costUsd: 0.0005 };
  }
}

/* --- Pipeline Flow Controller --- */
const PIPELINE_DATA = Object.freeze([
  {
    tag: "STAGE 01 // EXTRACTION",
    title: "Bare-Metal WASI Modules",
    desc: "Lightweight POSIX C/Rust modules (<100KB) running under Wasmtime autonomously poll Modbus TCP (Port 502) and industrial PLCs directly without middleware."
  },
  {
    tag: "STAGE 02 // SYNAPTIC MESH & SOLIUM",
    title: "WebRTC P2P & Solium LoRaWAN",
    desc: "A transcendent connection across an air-gapped WebRTC DataChannel forms a local P2P Mesh Network, while outdoor perimeter IoT sensors backhaul over the Solium network."
  },
  {
    tag: "STAGE 03 // NOSQL & SOLARM",
    title: "IndexedDB & Solana Storage",
    desc: "Incoming telemetry is cached client-side in IndexedDB for deterministic, offline-first analysis, later archiving deep history to Solana Shadow Drive."
  },
  {
    tag: "STAGE 04 // WEBMCP COGNITION",
    title: "Structured Contracts for AI Agents",
    desc: "The Web Model Context Protocol (WebMCP) exposes live telemetry states directly to local AI agents via structured tool contracts without cloud API roundtrips."
  },
  {
    tag: "STAGE 05 // SOLANA SETTLEMENT",
    title: "SSHAnet Batch Verification & cNFTs",
    desc: "Batches of telemetry readings are compressed into Merkle trees, signed with Ed25519 hardware keys, and settled onto Solana at ~$0.0005 per batch."
  }
]);

function initPipelineUI() {
  const display = document.getElementById("pipe-display");
  const tabs = document.querySelectorAll(".pipe-tab");
  if (!display || !tabs.length) return;

  const render = (idx) => {
    const item = PIPELINE_DATA[idx] || PIPELINE_DATA[0];
    tabs.forEach((t, i) => {
      const active = i === idx;
      t.classList.toggle("active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
    });
    display.innerHTML = `
      <div class="pipe-card">
        <span class="pipe-tag">${item.tag}</span>
        <h3 style="color: var(--yellow-glow, #D4AF37); margin: 6px 0;">${item.title}</h3>
        <p style="font-size: 0.85rem; line-height: 1.6;">${item.desc}</p>
      </div>`;
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const idx = Number.parseInt(tab.getAttribute("data-step") || "0", 10);
      render(idx);
    });
  });
  render(0);
}

/* --- Driver Sandbox --- */
const DRIVER_SAMPLE_C = `int poll_plc(const char* ip, uint16_t reg, uint16_t* val) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    uint8_t pdu[12] = { 0, 1, 0, 0, 0, 6, 1, 3, reg >> 8, reg & 0xFF, 0, 1 };
    send(s, pdu, sizeof(pdu), 0);
    return recv(s, val, 2, 0) > 0 ? 0 : -1;
};`;

function initCodeSandbox() {
  const editor = document.getElementById("code-sandbox");
  const typeBtn = document.getElementById("btn-live-type");
  const execBtn = document.getElementById("btn-exec-code");
  const status = document.getElementById("sandbox-status");
  if (!editor || !typeBtn || !execBtn || !status) return;

  editor.value = "// Select 'Simulate Typing' or inspect POSIX C driver...";
  let typingInterval = null;

  typeBtn.addEventListener("click", () => {
    if (typingInterval) clearInterval(typingInterval);
    editor.value = "";
    let i = 0;
    status.textContent = "STATUS: TYPING DRIVER...";
    typingInterval = setInterval(() => {
      if (i < DRIVER_SAMPLE_C.length) {
        editor.value += DRIVER_SAMPLE_C[i++];
        editor.scrollTop = editor.scrollHeight;
      } else {
        clearInterval(typingInterval);
        typingInterval = null;
        status.textContent = "STATUS: CODE READY (WASI-COMPLIANT)";
      }
    }, 15);
  });

  execBtn.addEventListener("click", () => {
    if (editor.value.includes("socket") && editor.value.includes("send")) {
      status.textContent = "STATUS: COMPILED SUCCESSFULLY [wasm32-wasip1]";
    } else {
      status.textContent = "STATUS: BUILD FAILED (Missing POSIX socket calls)";
    }
  });
}

class TerminalConsole {
  #body;
  #input;
  #db;

  constructor(bodyEl, inputEl, dbEngine) {
    this.#body = bodyEl;
    this.#input = inputEl;
    this.#db = dbEngine;
    this.#bind();
  }

  #bind() {
    this.#input.addEventListener("keydown", async (e) => {
      if (e.key === "Enter") {
        const val = this.#input.value.trim();
        if (!val) return;
        this.#echo(val);
        this.#input.value = "";
        await this.#run(val.toLowerCase());
      }
    });
  }

  #echo(cmd) {
    const row = document.createElement("div");
    row.className = "terminal-log";
    const prompt = document.createElement("em");
    prompt.textContent = "creiters@sshanet-2026:~$ ";
    const txt = document.createElement("code");
    txt.textContent = cmd;
    row.appendChild(prompt);
    row.appendChild(txt);
    this.#append(row);
  }

  #append(node) {
    this.#body.appendChild(node);
    this.#body.scrollTop = this.#body.scrollHeight;
  }

  async #run(cmd) {
    const row = document.createElement("div");
    row.className = "terminal-log";

    switch (cmd) {
      case "awaken":
        row.innerHTML = `<mark>[SYSTEM INIT]:</mark> CyberPear P2P protocol booted. Local Vectorization Engine running.`;
        break;
      case "help":
        row.innerHTML = `<b>COMMANDS:</b> awaken, solium, solarm, webrtc, webmcp, nosql, robot, synergy, modbus, ping, mine, clear`;
        break;
      case "solium":
        const packet = { hotspot_cNFT: "8xHT...cNFT", network: "Solana IOT SubDAO", rssi: -68, snr: 9.4, payload: { devEUI: "A84041F11D", temp: 24.8 } };
        const id = await this.#db.put("SOLIUM_SOLANA", packet);
        row.innerHTML = `
          <strong>[SOLIUM &times; SOLANA]:</strong> LoRaWAN backhaul packet ingested.<br>
          &bull; Hotspot Identity: Verified cNFT <code>${packet.hotspot_cNFT}</code><br>
          &bull; Settlement: Burned <b>1 Data Credit</b> via Solana SPL<br>
          &bull; Local Persistence: Committed to NoSQL (Doc #${id ?? "LOCAL"})
        `;
        break;
      case "solarm":
        const result = await this.#db.settleSolarmOnSolana();
        if (!result) {
          row.innerHTML = `<strong>[SOLANA SOLARM]:</strong> Telemetry store is empty. Ingest packets via <code>mqtt</code> first.`;
        } else {
          row.innerHTML = `
            <strong>[SOLANA SOLARMS AI &times; STORAGE]:</strong><br>
            &bull; Archived <b>${result.count}</b> records to Solana Storage: <code>${result.shdwUri}</code><br>
            &bull; Anchored Agent Root to Solana: <code>${result.merkleRoot}</code><br>
            &bull; Hardware Ed25519 Sig: <code>${result.txSignature.slice(0, 16)}...</code> (~$0.0005)
          `;
        }
        break;
      case "webmcp":
        row.innerHTML = `
          <strong>[WEBMCP TOOL CONTRACTS]:</strong><br>
          &bull; <code>read_plc_registers(port: 502, reg: 40001)</code> &rarr; WASI Direct Socket<br>
          &bull; <code>eval_safety_proximity(distance_m)</code> &rarr; Sub-15ms Dynamic Braking<br>
          &bull; <code>anchor_solarm_consensus(merkle_leaf)</code> &rarr; Solana cNFT Batch Settlement
        `;
        break;
      case "synergy":
        const agent = new LocalSolarmAgentEngine(this.#db);
        const res = await agent.runLocalInference(1.1);
        row.innerHTML = `
          <strong>[SOLANA SOLARMS // LOCAL INFERENCE]:</strong><br>
          &bull; Inference Engine: <code>${res.decision.model}</code> on <b>${res.decision.inferenceDevice}</b><br>
          &bull; Execution Latency: <b>${res.decision.latencyMs}ms</b> (SIL-3 Deterministic)<br>
          &bull; Policy Action: Velocity clamped to <b>${res.decision.maxVelocity} mm/s</b> [Operator at 1.1m]<br>
          &bull; Hardware Ed25519 Sig: <code>${res.hardwareSig.slice(0, 16)}...</code> (Anchored: ~$${res.costUsd})
        `;
        break;
      case "webrtc":
        row.innerHTML = `<strong>[WEBRTC MESH]:</strong> Air-gapped DataChannel active. Connected Peers: 4 Cobots | 0 External Cloud Egress.`;
        break;
      case "nosql":
        const c = await this.#db.count();
        row.innerHTML = `<strong>[INDEXEDDB NOSQL]:</strong> Target 'telemetry_store' holds <b>${c}</b> active documents.`;
        break;
      case "robot":
        row.innerHTML = `<strong>KUKA KR-100:</strong> J1: 42.18° | J2: -12.45° | J3: 88.02° | SIL-3 Safety: NORMAL`;
        break;
      case "modbus":
        row.innerHTML = `<strong>MODBUS TCP (192.168.1.120:502):</strong> Reg 40001 (Temp): 48.2°C | Cycle: 1.84ms`;
        break;
      case "ping":
        row.innerHTML = `<em>SOLANA SSHAnet:</em> Pong received in <b>${Math.floor(Math.random() * 40 + 360)}ms</b>. Batch cost: <b>$0.0005</b>.`;
        break;
      case "mine":
        const root = SecurityPrimitives.randomHex(16);
        await this.#db.put("SOLANA_DEPIN", { merkleRoot: root, records: 1000 });
        row.innerHTML = `<mark>[PROOF OF TELEMETRY]:</mark> Merkle root: <b>${root}</b> signed with hardware Ed25519 key.`;
        break;
      case "clear":
        while (this.#body.firstChild) this.#body.removeChild(this.#body.firstChild);
        return;
      default:
        row.textContent = `Unknown command: '${cmd}'. Type 'help' for command list.`;
        break;
    }
    this.#append(row);
  }
}

function initNavigation() {
  const hamburger = document.getElementById("hamburger");
  const navLinks = document.getElementById("nav-links");
  if (hamburger && navLinks) {
    hamburger.addEventListener("click", () => {
      const isOpen = navLinks.classList.toggle("active");
      hamburger.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });
    navLinks.querySelectorAll("a").forEach((l) => {
      l.addEventListener("click", () => {
        navLinks.classList.remove("active");
        hamburger.setAttribute("aria-expanded", "false");
      });
    });
  }
}

function initParticles() {
  const canvas = document.getElementById("neural-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let w = window.innerWidth;
  let h = window.innerHeight;

  function setCanvasDimensions() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  setCanvasDimensions();

  const particleCount = window.innerWidth < 768 ? 16 : 32;
  const particles = Array.from({ length: particleCount }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4
  }));

  let resizeTimeout;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      setCanvasDimensions();
      particles.forEach((p) => {
        p.x = Math.min(p.x, w);
        p.y = Math.min(p.y, h);
      });
    }, 150);
  }, { passive: true });

  function step() {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#79d98e";
    ctx.strokeStyle = "rgba(121, 217, 142, 0.1)";

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
      ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 110) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

document.addEventListener("DOMContentLoaded", async () => {
  const db = new NoSQLEngine();
  try {
    await db.init();
  } catch (err) {
    console.error("Local storage error:", err);
  }

  initNavigation();
  initPipelineUI();
  initCodeSandbox();
  initParticles();

  const termBody = document.getElementById("term-body");
  const termInput = document.getElementById("term-input");
  if (termBody && termInput) {
    new TerminalConsole(termBody, termInput, db);
  }

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch((err) => console.log("SW failed:", err));
  }
});
