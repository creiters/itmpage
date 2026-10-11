export class JSONLLMCodec {
  static BLOCK_SIZE = 32;

  static quantizeQ4(floatArray) {
    const totalBlocks = Math.ceil(floatArray.length / this.BLOCK_SIZE);
    const scales = new Float32Array(totalBlocks);
    const mins = new Float32Array(totalBlocks);
    const packedBytes = new Uint8Array(totalBlocks * (this.BLOCK_SIZE / 2));

    let byteIdx = 0;
    for (let b = 0; b < totalBlocks; b++) {
      const start = b * this.BLOCK_SIZE;
      let minVal = Infinity;
      let maxVal = -Infinity;

      for (let i = 0; i < this.BLOCK_SIZE; i++) {
        const val = start + i < floatArray.length ? floatArray[start + i] : 0;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }

      const scale = (maxVal - minVal) / 15.0 || 1.0;
      scales[b] = scale;
      mins[b] = minVal;

      for (let i = 0; i < this.BLOCK_SIZE; i += 2) {
        const v0 = start + i < floatArray.length ? floatArray[start + i] : minVal;
        const v1 = start + i + 1 < floatArray.length ? floatArray[start + i + 1] : minVal;
        const q0 = Math.max(0, Math.min(15, Math.round((v0 - minVal) / scale)));
        const q1 = Math.max(0, Math.min(15, Math.round((v1 - minVal) / scale)));
        packedBytes[byteIdx++] = (q1 << 4) | (q0 & 0x0f);
      }
    }
    return { scales, mins, packedBytes, len: floatArray.length };
  }

  static dequantizeQ4(scales, mins, packedBytes, originalLength) {
    const out = new Float32Array(originalLength);
    const totalBlocks = scales.length;
    let outIdx = 0;
    let byteIdx = 0;

    for (let b = 0; b < totalBlocks && outIdx < originalLength; b++) {
      const scale = scales[b];
      const minVal = mins[b];
      for (let i = 0; i < this.BLOCK_SIZE / 2 && outIdx < originalLength; i++) {
        const byte = packedBytes[byteIdx++];
        const q0 = byte & 0x0f;
        const q1 = (byte >> 4) & 0x0f;
        out[outIdx++] = q0 * scale + minVal;
        if (outIdx < originalLength) out[outIdx++] = q1 * scale + minVal;
      }
    }
    return out;
  }

  static async compress(buffer) {
    if (typeof CompressionStream !== 'undefined') {
      const cs = new CompressionStream('deflate');
      const writer = cs.writable.getWriter();
      writer.write(buffer);
      writer.close();
      return await new Response(cs.readable).arrayBuffer();
    }
    return buffer;
  }

  static async decompress(buffer) {
    if (typeof DecompressionStream !== 'undefined') {
      const ds = new DecompressionStream('deflate');
      const writer = ds.writable.getWriter();
      writer.write(buffer);
      writer.close();
      return await new Response(ds.readable).arrayBuffer();
    }
    return buffer;
  }

  static bytesToBase64(bytes) {
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  static base64ToBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  static async exportToJSONLLM(config, tokenizer, weights) {
    const tensors = {};
    for (const [name, arr] of Object.entries(weights)) {
      const { scales, mins, packedBytes, len } = this.quantizeQ4(arr);
      const deflated = await this.compress(packedBytes.buffer);
      tensors[name] = {
        l: len,
        s: Array.from(scales).map((v) => Number(v.toFixed(5))),
        m: Array.from(mins).map((v) => Number(v.toFixed(5))),
        d: this.bytesToBase64(new Uint8Array(deflated))
      };
    }

    const payload = {
      format: 'jsonllm/v2',
      createdAt: new Date().toISOString(),
      config,
      tokenizer: {
        tokenToId: Object.fromEntries(tokenizer.tokenToId),
        idToToken: Object.fromEntries(tokenizer.idToToken)
      },
      tensors
    };
    return JSON.stringify(payload);
  }

  static async importFromJSONLLM(jsonString) {
    const parsed = typeof jsonString === 'string' ? JSON.parse(jsonString) : jsonString;
    const weights = {};
    for (const [name, meta] of Object.entries(parsed.tensors)) {
      const deflated = this.base64ToBytes(meta.d);
      const decompressed = await this.decompress(deflated.buffer);
      weights[name] = this.dequantizeQ4(
        new Float32Array(meta.s),
        new Float32Array(meta.m),
        new Uint8Array(decompressed),
        meta.l
      );
    }
    return {
      config: parsed.config,
      tokenizer: parsed.tokenizer,
      weights
    };
  }
}
