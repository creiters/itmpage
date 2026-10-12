/**
 * garbage-collector.js
 * Explicit In-Memory Garbage Collector and Scratchpad Allocator
 */

export class MemoryGarbageCollector {
  constructor(maxPoolBytes = 64 * 1024 * 1024) {
    this.maxPoolBytes = maxPoolBytes;
    this.allocatedBytes = 0;
    this.pools = new Map(); // Size -> Array of ArrayBuffers
    this.activeRecords = new Set();
  }

  // Acquire or recycle a pinned buffer of an exact byte length
  acquireBuffer(byteLength) {
    const bucket = this.pools.get(byteLength);
    if (bucket && bucket.length > 0) {
      const buf = bucket.pop();
      new Uint8Array(buf).fill(0);
      this.activeRecords.add(buf);
      return buf;
    }

    const newBuf = new ArrayBuffer(byteLength);
    this.allocatedBytes += byteLength;
    this.activeRecords.add(newBuf);
    return newBuf;
  }

  // Release a buffer back into the reuse pool
  releaseBuffer(buf) {
    if (!buf || !this.activeRecords.has(buf)) return;
    this.activeRecords.delete(buf);

    const byteLength = buf.byteLength;
    if (!this.pools.has(byteLength)) {
      this.pools.set(byteLength, []);
    }
    const bucket = this.pools.get(byteLength);
    if (bucket.length < 64) {
      bucket.push(buf);
    } else {
      this.allocatedBytes -= byteLength; // Allow native GC to collect excess
    }
  }

  // Mark-and-sweep cleanup over autograd forward tapes
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
        if (l.xMid && l.xMid.buffer) this.releaseBuffer(l.xMid.buffer);
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

  // Purge all idle buffer pools
  purgePools() {
    let freed = 0;
    for (const [size, list] of this.pools.entries()) {
      freed += size * list.length;
      list.length = 0;
    }
    this.pools.clear();
    this.allocatedBytes -= freed;
    return freed;
  }

  getMetrics() {
    let pooled = 0;
    for (const [size, list] of this.pools.entries()) {
      pooled += size * list.length;
    }
    return {
      activeAllocatedBytes: this.allocatedBytes - pooled,
      pooledStandbyBytes: pooled,
      totalTrackedBytes: this.allocatedBytes
    };
  }
}
