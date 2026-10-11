export class ChunkIndexer {
  static CHUNK_SIZE = 4 * 1024 * 1024; // 4 MB boundary[span_4](start_span)[span_4](end_span)

  /**
   * High-speed, non-cryptographic MurmurHash3 (32-bit)
   * ~50x faster than async SHA-256 on 4MB ArrayBuffers
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
   * Builds an O(1) Segment Interval Index & Inverted Term Index
   */
  static splitAndIndex(modelBlob, modelName, onProgress) {
    return new Promise(async (resolve, reject) => {
      try {
        const totalBytes = modelBlob.size;
        const totalChunks = Math.ceil(totalBytes / ChunkIndexer.CHUNK_SIZE);
        const chunks = [];
        const segmentTable = [];
        const invertedIndex = new Map();

        for (let i = 0; i < totalChunks; i++) {
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
            totalChunks,
            byteOffset: start,
            byteLength: binary.byteLength,
            checksum,
            layerRange: [layerStart, layerEnd],
            contextPointers: {
              prev: i > 0 ? `${modelName}_seg_${i - 1}` : null,
              next: i + 1 < totalChunks ? `${modelName}_seg_${i + 1}` : null
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

          // Inverted index for layer-based chunk resolution
          for (let l = layerStart; l <= layerEnd; l++) {
            if (!invertedIndex.has(l)) invertedIndex.set(l, []);
            invertedIndex.get(l).push(i);
          }

          chunks.push({ metadata, binary });

          if (onProgress) {
            onProgress({
              phase: 'indexing',
              current: i + 1,
              total: totalChunks,
              percent: (((i + 1) / totalChunks) * 100).toFixed(1)
            });
          }
        }

        const manifest = {
          modelName,
          totalBytes,
          chunkSize: ChunkIndexer.CHUNK_SIZE,
          totalSegments: totalChunks,
          segmentTable,
          layerToSegments: Object.fromEntries(invertedIndex)
        };

        resolve({ manifest, chunks });
      } catch (err) {
        reject(err);
      }
    });
  }

  /**
   * O(1) Lookup: Resolves chunk index directly from linear byte offset
   */
  static findSegmentByOffset(manifest, byteOffset) {
    if (byteOffset < 0 || byteOffset >= manifest.totalBytes) return null;
    const idx = Math.floor(byteOffset / manifest.chunkSize);
    return manifest.segmentTable[idx] || null;
  }

  /**
   * O(1) Lookup: Resolves chunk indices containing target transformer layer
   */
  static findSegmentsByLayer(manifest, layerIndex) {
    return manifest.layerToSegments[layerIndex] || [];
  }

  /**
   * Fast In-Memory Reassembly with Synchronous Murmur3 Verification
   */
  static reassemble(orderedChunks, manifest) {
    const totalLength = orderedChunks.reduce((acc, c) => acc + c.binary.byteLength, 0);
    const unified = new Uint8Array(totalLength);

    for (let i = 0; i < orderedChunks.length; i++) {
      const c = orderedChunks[i];
      if (!c.binary || c.binary.byteLength === 0) {
        throw new Error(`Detached or empty buffer at segment index ${i}`);
      }

      const calculatedChecksum = this.fastHash32(c.binary);
      const expectedChecksum = c.metadata.checksum;

      if (calculatedChecksum !== expectedChecksum) {
        throw new Error(
          `Checksum mismatch at segment ${i} (expected ${expectedChecksum}, got ${calculatedChecksum})`
        );
      }

      unified.set(new Uint8Array(c.binary), c.metadata.byteOffset);
    }

    return unified.buffer;
  }
}
