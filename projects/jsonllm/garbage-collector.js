/**
 * garbage-collector.js
 * Explicit In-Memory Garbage Collector and Scratchpad Recycler
 */

export class MemoryGarbageCollector {
  constructor(initialBytes = 32 * 1024 * 1024) {
    this.totalAllocatedBytes = initialBytes;
    this.pools = new Map();
    this.activeRecords = new Set();
  }

  acquireBuffer(byteLength) {
    const bucket = this.pools.get(byteLength);
    if (bucket && bucket.length > 0) {
      const buffer = bucket.pop();
      new Uint8Array(buffer).fill(0);
      this.activeRecords.add(buffer);
      return buffer;
    }
    const newBuffer = new ArrayBuffer(byteLength);
    this.activeRecords.add(newBuffer);
    return newBuffer;
  }

  releaseBuffer(buffer) {
    if (!buffer || !this.activeRecords.has(buffer)) return;
    this.activeRecords.delete(buffer);
    const len = buffer.byteLength;
    if (!this.pools.has(len)) {
      this.pools.set(len, []);
    }
    const bucket = this.pools.get(len);
    if (bucket.length < 128) {
      bucket.push(buffer);
    }
  }

  sweepTape(tape) {
    if (!tape) return;
    if (tape.xInit && tape.xInit.buffer) {
      this.releaseBuffer(tape.xInit.buffer);
      tape.xInit = null;
    }
    if (tape.normFinal && tape.normFinal.out && tape.normFinal.out.buffer) {
      this.releaseBuffer(tape.normFinal.out.buffer);
      tape.normFinal.out = null;
    }
    if (tape.layers && Array.isArray(tape.layers)) {
      for (const l of tape.layers) {
        if (l.q && l.q.buffer) this.releaseBuffer(l.q.buffer);
        if (l.k && l.k.buffer) this.releaseBuffer(l.k.buffer);
        if (l.v && l.v.buffer) this.releaseBuffer(l.v.buffer);
        if (l.att && l.att.buffer) this.releaseBuffer(l.att.buffer);
        if (l.h1 && l.h1.buffer) this.releaseBuffer(l.h1.buffer);
        if (l.h2 && l.h2.buffer) this.releaseBuffer(l.h2.buffer);
        if (l.siluH1 && l.siluH1.buffer) this.releaseBuffer(l.siluH1.buffer);
        if (l.mlpAct && l.mlpAct.buffer) this.releaseBuffer(l.mlpAct.buffer);
        if (l.norm1 && l.norm1.out && l.norm1.out.buffer) this.releaseBuffer(l.norm1.out.buffer);
        if (l.norm2 && l.norm2.out && l.norm2.out.buffer) this.releaseBuffer(l.norm2.out.buffer);
      }
      tape.layers.length = 0;
    }
  }

  purgeIdlePools() {
    let freed = 0;
    for (const [size, list] of this.pools.entries()) {
      freed += size * list.length;
      list.length = 0;
    }
    this.pools.clear();
    return freed;
  }
}
