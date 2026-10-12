/**
 * wasm-engine.js - WASI Snapshot Preview 1 Runtime & SIMD Hardware Bridge
 */

export class WasiRuntime {
  constructor(initialPages = 256, maxPages = 2048) {
    this.memory = new WebAssembly.Memory({ initial: initialPages, maximum: maxPages });
    this.exports = null;
    this.heapPtr = 65536; // Reserve first 64KB for internal kernel scratchpads
  }

  createImports(logHandler) {
    const mem = this.memory;

    return {
      wasi_snapshot_preview1: {
        args_get: () => 0,
        args_sizes_get: (argcPtr, argvBufSizePtr) => {
          const view = new DataView(mem.buffer);
          view.setUint32(argcPtr, 0, true);
          view.setUint32(argvBufSizePtr, 0, true);
          return 0;
        },
        environ_get: () => 0,
        environ_sizes_get: (countPtr, bufSizePtr) => {
          const view = new DataView(mem.buffer);
          view.setUint32(countPtr, 0, true);
          view.setUint32(bufSizePtr, 0, true);
          return 0;
        },
        clock_time_get: (clockId, precision, timePtr) => {
          const view = new DataView(mem.buffer);
          view.setBigUint64(timePtr, BigInt(Date.now()) * 1000000n, true);
          return 0;
        },
        fd_write: (fd, iovs, iovsLen, nwritten) => {
          const view = new DataView(mem.buffer);
          let totalBytes = 0;
          let output = '';

          for (let i = 0; i < iovsLen; i++) {
            const ptr = view.getUint32(iovs + i * 8, true);
            const len = view.getUint32(iovs + i * 8 + 4, true);
            const chunk = new Uint8Array(mem.buffer, ptr, len);
            output += new TextDecoder('utf-8').decode(chunk);
            totalBytes += len;
          }

          view.setUint32(nwritten, totalBytes, true);
          if (output.trim() && logHandler) logHandler(output);
          return 0;
        },
        fd_read: () => 0,
        fd_seek: () => 0,
        fd_close: () => 0,
        proc_exit: (status) => {
          if (logHandler) logHandler(`[WASI] Process exit: ${status}`);
          return 0;
        }
      },
      env: {
        memory: this.memory,
        abort: () => {
          if (logHandler) logHandler('[WASM] Abort execution triggered.');
        }
      }
    };
  }

  async compileAndInstantiate(binaryBuffer, logHandler) {
    const imports = this.createImports(logHandler);
    const module = await WebAssembly.compile(binaryBuffer);
    const instance = await WebAssembly.instantiate(module, imports);
    this.exports = instance.exports;
    return instance;
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
}
