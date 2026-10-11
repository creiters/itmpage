export class ChunkIndexer {
  static CHUNK_SIZE = 4 * 1024 * 1024; // Strict 4 MB Segment Interval Boundary

  static murmur3(buffer) {
    const bytes = new Uint8Array(buffer);
    let h1 = 0xdeadbeef ^ 0;
    const c1 = 0xcc9e2d51;
    const c2 = 0x1b873593;
    const len = bytes.length;
    const nblocks = Math.floor(len / 4);

    const view = new DataView(buffer, bytes.byteOffset, bytes.byteLength);

    for (let i = 0; i < nblocks; i++) {
      let k1 = view.getUint32(i * 4, true);
      k1 = Math.imul(k1, c1);
      k1 = (k1 << 15) | (k1 >>> 17);
      k1 = Math.imul(k1, c2);

      h1 ^= k1;
      h1 = (h1 << 13) | (h1 >>> 19);
      h1 = Math.imul(h1, 5) + 0xe6546b64;
    }

    let k1 = 0;
    const tail = nblocks * 4;
    switch (len & 3) {
      case 3: k1 ^= bytes[tail + 2] << 16;
      case 2: k1 ^= bytes[tail + 1] << 8;
      case 1:
        k1 ^= bytes[tail];
        k1 = Math.imul(k1, c1);
        k1 = (k1 << 15) | (k1 >>> 17);
        k1 = Math.imul(k1, c2);
        h1 ^= k1;
    }

    h1 ^= len;
    h1 ^= h1 >>> 16;
    h1 = Math.imul(h1, 0x85ebca6b);
    h1 ^= h1 >>> 13;
    h1 = Math.imul(h1, 0xc2b2ae35);
    h1 ^= h1 >>> 16;
    return (h1 >>> 0).toString(16).padStart(8, '0');
  }

  static async splitToSlices(blob, modelName, onProgress = null) {
    const totalBytes = blob.size;
    const totalChunks = Math.ceil(totalBytes / this.CHUNK_SIZE);
    const chunks = [];
    const intervalMap = [];

    for (let i = 0; i < totalChunks; i++) {
      const start = i * this.CHUNK_SIZE;
      const end = Math.min(start + this.CHUNK_SIZE, totalBytes);
      const sliceBlob = blob.slice(start, end);
      const buffer = await sliceBlob.arrayBuffer();

      const checksum = this.murmur3(buffer);
      const chunkId = `${modelName}_si_${i}`;

      const minLayer = Math.floor((i / totalChunks) * 32);
      const maxLayer = Math.floor(((i + 1) / totalChunks) * 32);

      const metadata = {
        chunkId,
        modelName,
        index: i,
        totalChunks,
        byteOffset: start,
        byteLength: buffer.byteLength,
        checksum,
        context: {
          layerRange: [minLayer, maxLayer],
          targetNextContext: i < totalChunks - 1 ? `${modelName}_si_${i + 1}` : null
        }
      };

      chunks.push({ metadata, binary: buffer });
      intervalMap.push({
        index: i,
        chunkId,
        offset: start,
        length: buffer.byteLength,
        checksum,
        layers: [minLayer, maxLayer]
      });

      if (onProgress) {
        onProgress({ phase: 'SLICING', current: i + 1, total: totalChunks });
      }
    }

    const manifest = {
      modelName,
      totalBytes,
      totalChunks,
      intervalMap,
      createdAt: new Date().toISOString()
    };

    return { manifest, chunks };
  }

  static reassemble(chunks, manifest = null) {
    const sorted = [...chunks].sort((a, b) => a.metadata.index - b.metadata.index);
    let totalLength = 0;
    for (const c of sorted) totalLength += c.binary.byteLength;

    const assembled = new Uint8Array(totalLength);
    let offset = 0;

    for (let i = 0; i < sorted.length; i++) {
      const c = sorted[i];
      const verifiedChecksum = this.murmur3(c.binary);
      if (c.metadata.checksum && verifiedChecksum !== c.metadata.checksum) {
        throw new Error(`SI-Radix checksum mismatch at slice ${c.metadata.chunkId}`);
      }
      assembled.set(new Uint8Array(c.binary), offset);
      offset += c.binary.byteLength;
    }

    return assembled.buffer;
  }
}
