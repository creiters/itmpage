/**
 * wasm-engine.js - WASI & WASM SIMD Hardware Accelerator Bridge
 */

export class WasiEngine {
  constructor(initialPages = 128, maxPages = 1024) {
    this.memory = new WebAssembly.Memory({ initial: initialPages, maximum: maxPages });
    this.exports = null;
    this.heapPtr = 65536; // Start dynamic allocations past reserved scratchpads
  }

  async init(wasmPath = './kernel.wasm') {
    const importObject = {
      wasi_snapshot_preview1: {
        clock_time_get: (clockId, precision, timePtr) => {
          const view = new DataView(this.memory.buffer);
          view.setBigUint64(timePtr, BigInt(Date.now()) * 1000000n, true);
          return 0;
        },
        proc_exit: (status) => status
      },
      env: {
        memory: this.memory
      }
    };

    try {
      const response = await fetch(wasmPath);
      const bytes = await response.arrayBuffer();
      const module = await WebAssembly.compile(bytes);
      const instance = await WebAssembly.instantiate(module, importObject);
      this.exports = instance.exports;
      return true;
    } catch {
      return false; // Fallback to JS engine if binary is unavailable
    }
  }

  allocate(numFloats) {
    const ptr = this.heapPtr;
    this.heapPtr += numFloats * 4;
    return ptr;
  }

  resetHeap(offset = 65536) {
    this.heapPtr = offset;
  }

  writeFloats(ptr, floatArray) {
    const view = new Float32Array(this.memory.buffer, ptr, floatArray.length);
    view.set(floatArray);
  }

  readFloats(ptr, length) {
    return new Float32Array(this.memory.buffer, ptr, length);
  }

  gemv(outPtr, wPtr, xPtr, rows, cols) {
    this.exports.wasm_gemv(outPtr, wPtr, xPtr, rows, cols);
  }

  gemvTranspose(dxPtr, wPtr, dyPtr, rows, cols) {
    this.exports.wasm_gemv_transpose(dxPtr, wPtr, dyPtr, rows, cols);
  }

  accumGrad(dwPtr, dyPtr, xPtr, rows, cols) {
    this.exports.wasm_accum_grad(dwPtr, dyPtr, xPtr, rows, cols);
  }

  rmsnormBackward(dxPtr, doutPtr, outPtr, size, invScale) {
    this.exports.wasm_rmsnorm_backward(dxPtr, doutPtr, outPtr, size, invScale);
  }

  adamwStep(wPtr, gPtr, mPtr, vPtr, size, lr, beta1, beta2, eps, decay, step) {
    this.exports.wasm_adamw_step(wPtr, gPtr, mPtr, vPtr, size, lr, beta1, beta2, eps, decay, step);
  }
}
