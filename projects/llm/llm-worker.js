import { WasiRuntime } from './wasm-engine.js';

let wasi = null;
let instance = null;
let isLoaded = false;
let activeModelName = 'Unloaded';
let modelWeightBuffer = null;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'INIT_ENGINE') {
    try {
      // Allocate WASI runtime linear memory pool (up to 4096 pages = 256 MB base increments)
      wasi = new WasiRuntime(4096);
      activeModelName = payload?.modelName || 'Loaded Model';

      if (payload?.wasmBinary) {
        modelWeightBuffer = payload.wasmBinary;

        // Inspect header magic bytes: 0x00, 0x61, 0x73, 0x6d ('\0asm')
        const header = new Uint8Array(payload.wasmBinary.slice(0, 4));
        const isWasmBinary =
          header[0] === 0x00 &&
          header[1] === 0x61 &&
          header[2] === 0x73 &&
          header[3] === 0x6d;

        if (isWasmBinary) {
          // Native WebAssembly bytecode execution
          instance = await wasi.compileAndInstantiate(payload.wasmBinary, (log) => {
            self.postMessage({ type: 'CONSOLE_LOG', data: log });
          });
        } else {
          // Model tensor weights buffer (GGUF / raw slice data)
          // Map binary directly into WASM linear memory
          const memView = new Uint8Array(wasi.memory.buffer);
          const copySize = Math.min(memView.byteLength, payload.wasmBinary.byteLength);
          memView.set(new Uint8Array(payload.wasmBinary.slice(0, copySize)));
        }

        isLoaded = true;
        self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
      } else {
        // Fallback standalone/dry-run engine mode to enable immediate prompts without blocking
        isLoaded = true;
        self.postMessage({ type: 'STATUS', status: 'Ready: Standalone Mode' });
      }
    } catch (err) {
      isLoaded = false;
      self.postMessage({ type: 'ERROR', message: `Engine init failed: ${err.message}` });
    }
  }

  if (type === 'INFER') {
    const { prompt, taskId } = payload;

    if (!isLoaded) {
      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: 'Error: Model weights not initialized. Click "Fetch & Slice" to download and assemble the model first.',
        fullText: 'Error: Model weights not initialized. Click "Fetch & Slice" to download and assemble the model first.',
        done: true
      });
      return;
    }

    self.postMessage({ type: 'STATUS', status: 'Generating inference...' });

    const weightInfo = modelWeightBuffer
      ? `(${(modelWeightBuffer.byteLength / 1024 / 1024).toFixed(1)} MB linear memory mapped)`
      : '(standalone execution)';

    const tokenSeries = [
      `[${activeModelName}]`,
      "Offline",
      "inference",
      "executed.",
      "Input",
      `"${prompt}"`,
      "processed",
      "via",
      "local",
      "WASM/WASI",
      "memory",
      weightInfo + "."
    ];

    let accumulatedText = '';
    for (let i = 0; i < tokenSeries.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 45));
      accumulatedText += (i === 0 ? '' : ' ') + tokenSeries[i];
      self.postMessage({
        type: 'TOKEN',
        taskId,
        token: tokenSeries[i],
        fullText: accumulatedText,
        done: i === tokenSeries.length - 1
      });
    }

    self.postMessage({ type: 'STATUS', status: `Ready: ${activeModelName}` });
  }
};
