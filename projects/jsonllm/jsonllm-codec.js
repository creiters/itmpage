/**
 * jsonllm-codec.js
 * Pure FP16 / FP4 E2M1 Codec, Deflate Base64 Streams, and Token-Delta Packaging
 */

export const HalfFloatArray = typeof Float16Array !== 'undefined'
  ? Float16Array
  : class HalfFloatArrayPolyfill extends Uint16Array {
      constructor(lengthOrArray) {
        if (typeof lengthOrArray === 'number') {
          super(lengthOrArray);
        } else {
          super(lengthOrArray.length);
          for (let i = 0; i < lengthOrArray.length; i++) {
            this[i] = HalfFloatArrayPolyfill.toHalf(lengthOrArray[i]);
          }
        }
      }

      static toHalf(val) {
        const buf = new ArrayBuffer(4);
        new DataView(buf).setFloat32(0, val, true);
        const u32 = new DataView(buf).getUint32(0, true);
        const sign = (u32 >>> 31) & 0x1;
        const exp = (u32 >>> 23) & 0xff;
        const frac = u32 & 0x7fffff;
        if (exp === 0xff) return (sign << 15) | (frac ? 0x7e00 : 0x7c00);
        if (exp === 0) return (sign << 15);
        const newExp = exp - 127 + 15;
        if (newExp >= 31) return (sign << 15) | 0x7c00;
        if (newExp <= 0) return (sign << 15);
        return (sign << 15) | (newExp << 10) | (frac >>> 13);
      }

      static toFloat(h) {
        const sign = (h & 0x8000) ? -1 : 1;
        const exp = (h >>> 10) & 0x1f;
        const frac = h & 0x3ff;
        if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
        if (exp === 31) return frac ? NaN : sign * Infinity;
        return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
      }
    };

const FP4_GRID = [
  0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
  -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
];

const FP4_BYTE_LUT_LOW = new HalfFloatArray(256);
const FP4_BYTE_LUT_HIGH = new HalfFloatArray(256);
for (let b = 0; b < 256; b++) {
  FP4_BYTE_LUT_LOW[b] = FP4_GRID[b & 0x0f];
  FP4_BYTE_LUT_HIGH[b] = FP4_GRID[(b >> 4) & 0x0f];
}

export class JSONLLMCodec {
  static BLOCK_SIZE = 32;

  static quantizeFP4(halfArray) {
    const totalBlocks = Math.ceil(halfArray.length / this.BLOCK_SIZE);
    const scales = new HalfFloatArray(totalBlocks);
    const packedBytes = new Uint8Array(totalBlocks * (this.BLOCK_SIZE / 2));
    let byteIdx = 0;

    for (let b = 0; b < totalBlocks; b++) {
      const start = b * this.BLOCK_SIZE;
      let maxAbs = 0.0;
      for (let i = 0; i < this.BLOCK_SIZE; i++) {
        const v = Math.abs(start + i < halfArray.length ? halfArray[start + i] : 0.0);
        if (v > maxAbs) maxAbs = v;
      }

      const scale = maxAbs > 1e-6 ? maxAbs / 6.0 : 1.0;
      scales[b] = scale;
      const invScale = 1.0 / scale;

      for (let i = 0; i < this.BLOCK_SIZE; i += 2) {
        const v0 = (start + i < halfArray.length ? halfArray[start + i] : 0.0) * invScale;
        const v1 = (start + i + 1 < halfArray.length ? halfArray[start + i + 1] : 0.0) * invScale;
        const q0 = this._closestFP4(v0);
        const q1 = this._closestFP4(v1);
        packedBytes[byteIdx++] = (q1 << 4) | (q0 & 0x0f);
      }
    }
    return { scales, packedBytes, len: halfArray.length };
  }

  static _closestFP4(val) {
    let best = 0, minDiff = Infinity;
    for (let idx = 0; idx < 16; idx++) {
      const diff = Math.abs(val - FP4_GRID[idx]);
      if (diff < minDiff) { minDiff = diff; best = idx; }
    }
    return best;
  }

  static dequantizeFP4(scales, packedBytes, originalLength) {
    const out = new HalfFloatArray(originalLength);
    const totalBlocks = scales.length;
    let outIdx = 0, byteIdx = 0;

    for (let b = 0; b < totalBlocks && outIdx < originalLength; b++) {
      const scale = scales[b];
      for (let i = 0; i < this.BLOCK_SIZE / 2 && outIdx < originalLength; i++) {
        const byte = packedBytes[byteIdx++];
        out[outIdx++] = FP4_BYTE_LUT_LOW[byte] * scale;
        if (outIdx < originalLength) out[outIdx++] = FP4_BYTE_LUT_HIGH[byte] * scale;
      }
    }
    return out;
  }

  static async compress(buffer) {
    const cs = new CompressionStream('deflate');
    const writer = cs.writable.getWriter();
    writer.write(buffer);
    writer.close();
    return await new Response(cs.readable).arrayBuffer();
  }

  static async decompress(buffer) {
    const ds = new DecompressionStream('deflate');
    const writer = ds.writable.getWriter();
    writer.write(buffer);
    writer.close();
    return await new Response(ds.readable).arrayBuffer();
  }

  static bytesToBase64(bytes) {
    let binary = '';
    const chunkSz = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSz) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSz));
    }
    return btoa(binary);
  }

  static base64ToBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  static async exportDialogueDataset(tokenizer, dialoguePairs) {
    const sequences = [];
    for (const pair of dialoguePairs) {
      const uTokens = tokenizer.encode(pair.u, false);
      const bTokens = tokenizer.encode(pair.b, false);
      sequences.push([uTokens.length, ...uTokens, bTokens.length, ...bTokens]);
    }
    const flat = new Uint16Array(sequences.flat());
    const compressed = await this.compress(flat.buffer);

    return JSON.stringify({
      format: 'jsonllm/dialog-v3',
      createdAt: new Date().toISOString(),
      pairCount: dialoguePairs.length,
      vocabSize: tokenizer.vocabSize,
      data: this.bytesToBase64(new Uint8Array(compressed))
    });
  }

  static async exportToJSONLLM(config, tokenizer, weights, mode = 'fp4') {
    const tensors = {};
    for (const [name, arr] of Object.entries(weights)) {
      const { scales, packedBytes, len } = this.quantizeFP4(arr);
      const deflated = await this.compress(packedBytes.buffer);
      tensors[name] = {
        l: len,
        s: Array.from(scales).map((v) => Number(v.toFixed(4))),
        d: this.bytesToBase64(new Uint8Array(deflated))
      };
    }

    return JSON.stringify({
      format: 'jsonllm/v3-clean',
      quantMode: mode,
      createdAt: new Date().toISOString(),
      config,
      tensors
    });
  }

  static async importFromJSONLLM(jsonString) {
    const parsed = typeof jsonString === 'string' ? JSON.parse(jsonString) : jsonString;
    const weights = {};
    for (const [name, meta] of Object.entries(parsed.tensors)) {
      const deflated = this.base64ToBytes(meta.d);
      const decompressed = await this.decompress(deflated.buffer);
      weights[name] = this.dequantizeFP4(
        new HalfFloatArray(meta.s),
        new Uint8Array(decompressed),
        meta.l
      );
    }
    return { config: parsed.config, quantMode: parsed.quantMode || 'fp4', weights };
  }
}
