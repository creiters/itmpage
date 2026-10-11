export class ChunkIndexer {
  static CHUNK_SIZE = 4 * 1024 * 1024; // Strict 4 MB segment boundary

  /**
   * High-speed synchronous MurmurHash3 (32-bit) checksum algorithm.
   * Eliminates asynchronous WebCrypto digest overhead on large ArrayBuffers.
   */
  static fastHash32(buffer, seed = 0x9747b28c) {
    const bytes = new Uint8Array(buffer);
    const len = bytes.length;
    const nblocks = Math.floor(len / 4);
    let h1 = seed >>> 0;

    const c1 = 0xcc9e2d51 >>> 0;
    const c2 = 0x1b873593 >>> 0;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    for (let i = 0; i < nblocks; i++) {
      let k1 = view.getUint32(i * 4, true);

      k1 = Math.imul(k1, c1) >>> 0;
      k1 = ((k1 << 15) | (k1 >>> 17)) >>> 0;
      k1 = Math.imul(k1, c2) >>> 0;

      h1 ^= k1;
      h1 = ((h1 << 13) | (h1 >>> 19)) >>> 0;
      h1 = (Math.imul(h1, 5) + 0xe6546b64) >>> 0;
    }

    let k1 = 0;
    const tailIndex = nblocks * 4;
    const remaining = len & 3;

    if (remaining === 3) k1 ^= bytes[tailIndex + 2] << 16;
    if (remaining >= 2) k1 ^= bytes[tailIndex + 1] << 8;
    if (remaining >= 1) {
      k1 ^= bytes[tailIndex];
      k1 = Math.imul(k1, c1) >>> 0;
      k1 = ((k1 << 15) | (k1 >>> 17)) >>> 0;
      k1 = Math.imul(k1, c2) >>> 0;
      h1 ^= k1;
    }

    h1 ^= len;
    h1 ^= h1 >>> 16;
    h1 = Math.imul(h1, 0x85ebca6b) >>> 0;
    h1 ^= h1 >>> 13;
    h1 = Math.imul(h1, 0xc2b2ae35) >>> 0;
    h1 ^= h1 >>> 16;

    return h1.toString(16).padStart(8, '0');
  }

  /**
   * Slices model binary into contiguous 4MB segments and constructs
   * constant-time O(1) interval mapping tables and layer indexes.
   */
  static splitAndIndex(modelBlob, modelName, onProgress) {
    return new Promise(async (resolve, reject) => {
      try {
        const totalBytes = modelBlob.size;
        const totalSegments = Math.ceil(totalBytes / ChunkIndexer.CHUNK_SIZE);
        const segments = [];
        const segmentTable = [];
        const invertedIndex = new Map();

        for (let i = 0; i < totalSegments; i++) {
          const start = i * ChunkIndexer.CHUNK_SIZE;
          const end = Math.min(start + ChunkIndexer.CHUNK_SIZE, totalBytes);
          const slice = modelBlob.slice(start, end);
          const binary = await slice.arrayBuffer();

          const checksum = ChunkIndexer.fastHash32(binary);
          const layerStart = Math.floor((start / totalBytes) * 30);
          const layerEnd = Math.floor((end / totalBytes) * 30);

          const metadata = {
            chunkId: `${modelName}_seg_${i}`,
            modelName,
            index: i,
            totalSegments,
            byteOffset: start,
            byteLength: binary.byteLength,
            checksum,
            layerRange: [layerStart, layerEnd],
            contextPointers: {
              prev: i > 0 ? `${modelName}_seg_${i - 1}` : null,
              next: i + 1 < totalSegments ? `${modelName}_seg_${i + 1}` : null
            }
          };

          segmentTable.push({
            index: i,
            start,
            end,
            length: binary.byteLength,
            layerStart,
            layerEnd,
            checksum
          });

          for (let l = layerStart; l <= layerEnd; l++) {
            if (!invertedIndex.has(l)) invertedIndex.set(l, []);
            invertedIndex.get(l).push(i);
          }

          segments.push({ metadata, binary });

          if (onProgress) {
            onProgress({
              phase: 'si-radix-indexing',
              current: i + 1,
              total: totalSegments,
              percent: (((i + 1) / totalSegments) * 100).toFixed(1)
            });
          }
        }

        const manifest = {
          modelName,
          totalBytes,
          segmentSize: ChunkIndexer.CHUNK_SIZE,
          totalSegments,
          segmentTable,
          layerToSegments: Object.fromEntries(invertedIndex)
        };

        resolve({ manifest, segments });
      } catch (err) {
        reject(err);
      }
    });
  }

  static findSegmentByOffset(manifest, byteOffset) {
    if (byteOffset < 0 || byteOffset >= manifest.totalBytes) return null;
    const idx = Math.floor(byteOffset / manifest.segmentSize);
    return manifest.segmentTable[idx] || null;
  }

  static findSegmentsByLayer(manifest, layerIndex) {
    return manifest.layerToSegments[layerIndex] || [];
  }

  static reassemble(orderedSegments, manifest) {
    const totalLength = orderedSegments.reduce((acc, s) => acc + s.binary.byteLength, 0);
    const unified = new Uint8Array(totalLength);

    for (let i = 0; i < orderedSegments.length; i++) {
      const s = orderedSegments[i];
      if (!s.binary || s.binary.byteLength === 0) {
        throw new Error(`Detached buffer encountered at SI-Radix segment index ${i}`);
      }

      const calculatedChecksum = this.fastHash32(s.binary);
      const expectedChecksum = s.metadata.checksum;

      if (calculatedChecksum !== expectedChecksum) {
        throw new Error(
          `SI-Radix checksum mismatch at segment ${i} (expected ${expectedChecksum}, got ${calculatedChecksum})`
        );
      }

      unified.set(new Uint8Array(s.binary), s.metadata.byteOffset);
    }

    return unified.buffer;
  }
}
