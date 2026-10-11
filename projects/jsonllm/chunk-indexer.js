export class ChunkIndexer {
  static CHUNK_SIZE = 4 * 1024 * 1024; // Strict 4MB SI-Radix Boundary

  static murmur3(str) {
    let h1 = 0xdeadbeef ^ 0;
    for (let i = 0; i < str.length; i++) {
      let k1 = str.charCodeAt(i);
      k1 = Math.imul(k1, 0xcc9e2d51);
      k1 = (k1 << 15) | (k1 >>> 17);
      k1 = Math.imul(k1, 0x1b873593);

      h1 ^= k1;
      h1 = (h1 << 13) | (h1 >>> 19);
      h1 = Math.imul(h1, 5) + 0xe6546b64;
    }
    h1 ^= str.length;
    h1 ^= h1 >>> 16;
    h1 = Math.imul(h1, 0x85ebca6b);
    h1 ^= h1 >>> 13;
    h1 = Math.imul(h1, 0xc2b2ae35);
    h1 ^= h1 >>> 16;
    return (h1 >>> 0).toString(16).padStart(8, '0');
  }

  static slicePayload(payloadString, modelName) {
    const totalChars = payloadString.length;
    const segments = [];
    let startChar = 0;
    let segIdx = 0;

    while (startChar < totalChars) {
      const endChar = Math.min(startChar + this.CHUNK_SIZE, totalChars);
      const sliceData = payloadString.substring(startChar, endChar);
      const checksum = this.murmur3(sliceData);

      segments.push({
        segmentId: `${modelName}_si_${segIdx}`,
        modelName,
        index: segIdx,
        startOffset: startChar,
        endOffset: endChar,
        totalBytes: totalChars,
        checksum,
        payload: sliceData
      });

      startChar = endChar;
      segIdx++;
    }

    return segments;
  }

  static reassemble(segments) {
    const sorted = [...segments].sort((a, b) => a.index - b.index);
    let output = '';
    for (const seg of sorted) {
      const calc = this.murmur3(seg.payload);
      if (calc !== seg.checksum) {
        throw new Error(`SI-Radix checksum mismatch at segment ${seg.segmentId}`);
      }
      output += seg.payload;
    }
    return output;
  }
}
