export class JsonLlmCodec {
  static BLOCK_SIZE = 32;

  static quantizeQ4_0(float32Array) {
    const totalBlocks = Math.ceil(float32Array.length / this.BLOCK_SIZE);
    const scales = new Float32Array(totalBlocks);
    const mins = new Float32Array(totalBlocks);
    const packedNibbles = new Uint8Array(totalBlocks * (this.BLOCK_SIZE / 2));

    let nibbleIdx = 0;
    for (let b = 0; b < totalBlocks; b++) {
      const start = b * this.BLOCK_SIZE;
      let minVal = Infinity;
      let maxVal = -Infinity;

      for (let i = 0; i < this.BLOCK_SIZE; i++) {
        const val = start + i < float32Array.length ? float32Array[start + i] : 0;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }

      const scale = (maxVal - minVal) / 15.0 || 1.0;
      scales[b] = scale;
      mins[b] = minVal;

      for (let i = 0; i < this.BLOCK_SIZE; i += 2) {
        const v0 = start + i < float32Array.length ? float32Array[start + i] : 0;
        const v1 = start + i + 1 < float32Array.length ? float32Array[start + i + 1] : 0;

        const q0 = Math.max(0, Math.min(15, Math.round((v0 - minVal) / scale)));
        const q1 = Math.max(0, Math.min(15, Math.round((v1 - minVal) / scale)));

        packedNibbles[nibbleIdx++] = (q0 & 0x0f) | ((q1 & 0x0f) << 4);
      }
    }

    return { scales, mins, packedNibbles };
  }

  static dequantizeQ4_0(scales, mins, packedNibbles, originalLength) {
    const out = new Float32Array(originalLength);
    const totalBlocks = scales.length;
    let outIdx = 0;
    let byteIdx = 0;

    for (let b = 0; b < totalBlocks && outIdx < originalLength; b++) {
      const scale = scales[b];
      const minVal = mins[b];

      for (let i = 0; i < this.BLOCK_SIZE / 2 && outIdx < originalLength; i++) {
        const byte = packedNibbles[byteIdx++];
        const q0 = byte & 0x0f;
        const q1 = (byte >> 4) & 0x0f;

        out[outIdx++] = q0 * scale + minVal;
        if (outIdx < originalLength) {
          out[outIdx++] = q1 * scale + minVal;
        }
      }
    }
    return out;
  }

  static async compressBuffer(buffer) {
    const cs = new CompressionStream('deflate');
    const writer = cs.writable.getWriter();
    writer.write(buffer);
    writer.close();
    const response = new Response(cs.readable);
    return await response.arrayBuffer();
  }

  static async decompressBuffer(buffer) {
    const ds = new DecompressionStream('deflate');
    const writer = ds.writable.getWriter();
    writer.write(buffer);
    writer.close();
    const response = new Response(ds.readable);
    return await response.arrayBuffer();
  }

  static bytesToBase64(uint8) {
    let binary = '';
    const len = uint8.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(uint8[i]);
    }
    return btoa(binary);
  }

  static base64ToBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  static async encodeModelToJSONLLM(config, tokenizerData, rawTensors) {
    const compressedTensors = {};

    for (const [tensorName, floatArray] of Object.entries(rawTensors)) {
      const { scales, mins, packedNibbles } = this.quantizeQ4_0(floatArray);
      const compressedBytes = await this.compressBuffer(packedNibbles.buffer);

      compressedTensors[tensorName] = {
        length: floatArray.length,
        scales: Array.from(scales).map((v) => Number(v.toFixed(5))),
        mins: Array.from(mins).map((v) => Number(v.toFixed(5))),
        q4_base64: this.bytesToBase64(new Uint8Array(compressedBytes))
      };
    }

    const packageEnvelope = {
      format: 'jsonllm/v1',
      createdAt: new Date().toISOString(),
      config,
      tokenizer: tokenizerData,
      compressed_tensors: compressedTensors
    };

    return JSON.stringify(packageEnvelope);
  }

  static async decodeJSONLLM(jsonString) {
    const parsed = typeof jsonString === 'string' ? JSON.parse(jsonString) : jsonString;
    const tensors = {};

    for (const [name, meta] of Object.entries(parsed.compressed_tensors)) {
      const deflatedBytes = this.base64ToBytes(meta.q4_base64);
      const rawNibblesBuf = await this.decompressBuffer(deflatedBytes.buffer);
      const packedNibbles = new Uint8Array(rawNibblesBuf);

      tensors[name] = this.dequantizeQ4_0(
        new Float32Array(meta.scales),
        new Float32Array(meta.mins),
        packedNibbles,
        meta.length
      );
    }

    return {
      config: parsed.config,
      tokenizer: parsed.tokenizer,
      tensors
    };
  }
}
