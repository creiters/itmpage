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

        const q0 = this._findClosestFP4(v0);
        const q1 = this._findClosestFP4(v1);
        packedBytes[byteIdx++] = (q1 << 4) | (q0 & 0x0f);
      }
    }
    return { scales, packedBytes, len: halfArray.length, type: 'fp4' };
  }

  static _findClosestFP4(val) {
    let bestIdx = 0;
    let minDiff = Infinity;
    for (let idx = 0; idx < 16; idx++) {
      const diff = Math.abs(val - FP4_GRID[idx]);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = idx;
      }
    }
    return bestIdx;
  }

  static dequantizeFP4(scales, packedBytes, originalLength) {
    const out = new HalfFloatArray(originalLength);
    const totalBlocks = scales.length;
    let outIdx = 0;
    let byteIdx = 0;

    for (let b = 0; b < totalBlocks && outIdx < originalLength; b++) {
      const scale = scales[b];
      for (let i = 0; i < this.BLOCK_SIZE / 2 && outIdx < originalLength; i++) {
        const byte = packedBytes[byteIdx++];
        out[outIdx++] = FP4_BYTE_LUT_LOW[byte] * scale;
        if (outIdx < originalLength) {
          out[outIdx++] = FP4_BYTE_LUT_HIGH[byte] * scale;
        }
      }
    }
    return out;
  }

  static quantizeQ4(halfArray) {
    const totalBlocks = Math.ceil(halfArray.length / this.BLOCK_SIZE);
    const scales = new HalfFloatArray(totalBlocks);
    const mins = new HalfFloatArray(totalBlocks);
    const packedBytes = new Uint8Array(totalBlocks * (this.BLOCK_SIZE / 2));

    let byteIdx = 0;
    for (let b = 0; b < totalBlocks; b++) {
      const start = b * this.BLOCK_SIZE;
      let minVal = Infinity;
      let maxVal = -Infinity;

      for (let i = 0; i < this.BLOCK_SIZE; i++) {
        const val = start + i < halfArray.length ? halfArray[start + i] : 0.0;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }

      const scale = (maxVal - minVal) / 15.0 || 1.0;
      scales[b] = scale;
      mins[b] = minVal;

      for (let i = 0; i < this.BLOCK_SIZE; i += 2) {
        const v0 = start + i < halfArray.length ? halfArray[start + i] : minVal;
        const v1 = start + i + 1 < halfArray.length ? halfArray[start + i + 1] : minVal;
        const q0 = Math.max(0, Math.min(15, Math.round((v0 - minVal) / scale)));
        const q1 = Math.max(0, Math.min(15, Math.round((v1 - minVal) / scale)));
        packedBytes[byteIdx++] = (q1 << 4) | (q0 & 0x0f);
      }
    }
    return { scales, mins, packedBytes, len: halfArray.length, type: 'q4' };
  }

  static dequantizeQ4(scales, mins, packedBytes, originalLength) {
    const out = new HalfFloatArray(originalLength);
    const totalBlocks = scales.length;
    let outIdx = 0;
    let byteIdx = 0;

    for (let b = 0; b < totalBlocks && outIdx < originalLength; b++) {
      const scale = scales[b];
      const minVal = mins[b];
      for (let i = 0; i < this.BLOCK_SIZE / 2 && outIdx < originalLength; i++) {
        const byte = packedBytes[byteIdx++];
        out[outIdx++] = (byte & 0x0f) * scale + minVal;
        if (outIdx < originalLength) {
          out[outIdx++] = ((byte >> 4) & 0x0f) * scale + minVal;
        }
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
    const chunkSz = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSz) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSz));
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

  static async exportDialogueDataset(tokenizer, dialoguePairs) {
    const tokenSequences = [];
    for (const pair of dialoguePairs) {
      const uTokens = tokenizer.encode(pair.u);
      const bTokens = tokenizer.encode(pair.b);
      const packed = [uTokens.length, ...uTokens, bTokens.length, ...bTokens];
      tokenSequences.push(packed);
    }

    const flatArray = new Uint16Array(tokenSequences.flat());
    const compressed = await this.compress(flatArray.buffer);

    const payload = {
      format: 'jsonllm/dialog-v3',
      createdAt: new Date().toISOString(),
      pairCount: dialoguePairs.length,
      vocab: Object.fromEntries(tokenizer.tokenToId),
      data: this.bytesToBase64(new Uint8Array(compressed))
    };
    return JSON.stringify(payload);
  }

  static async importDialogueDataset(jsonString, tokenizer) {
    const parsed = typeof jsonString === 'string' ? JSON.parse(jsonString) : jsonString;
    const bytes = this.base64ToBytes(parsed.data);
    const decompressed = await this.decompress(bytes.buffer);
    const flat = new Uint16Array(decompressed);

    if (parsed.vocab) {
      for (const [word, id] of Object.entries(parsed.vocab)) {
        tokenizer.tokenToId.set(word, id);
        tokenizer.idToToken.set(id, word);
      }
      tokenizer.vocabSize = Math.max(tokenizer.vocabSize, tokenizer.tokenToId.size);
    }

    const pairs = [];
    let idx = 0;
    while (idx < flat.length) {
      const uLen = flat[idx++];
      const uToks = Array.from(flat.subarray(idx, idx + uLen));
      idx += uLen;

      const bLen = flat[idx++];
      const bToks = Array.from(flat.subarray(idx, idx + bLen));
      idx += bLen;

      pairs.push({
        u: tokenizer.decode(uToks),
        b: tokenizer.decode(bToks)
      });
    }
    return pairs;
  }

  static async exportToJSONLLM(config, tokenizer, weights, mode = 'fp4') {
    const tensors = {};

    for (const [name, arr] of Object.entries(weights)) {
      if (mode === 'fp4') {
        const { scales, packedBytes, len } = this.quantizeFP4(arr);
        const deflated = await this.compress(packedBytes.buffer);
        tensors[name] = {
          t: 'fp4',
          l: len,
          s: Array.from(scales).map((v) => Number(v.toFixed(4))),
          d: this.bytesToBase64(new Uint8Array(deflated))
        };
      } else {
        const { scales, mins, packedBytes, len } = this.quantizeQ4(arr);
        const deflated = await this.compress(packedBytes.buffer);
        tensors[name] = {
          t: 'q4',
          l: len,
          s: Array.from(scales).map((v) => Number(v.toFixed(4))),
          m: Array.from(mins).map((v) => Number(v.toFixed(4))),
          d: this.bytesToBase64(new Uint8Array(deflated))
        };
      }
    }

    const payload = {
      format: 'jsonllm/v3-pure16',
      quantMode: mode,
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
      const rawBytes = new Uint8Array(decompressed);

      if (meta.t === 'fp4') {
        weights[name] = this.dequantizeFP4(new HalfFloatArray(meta.s), rawBytes, meta.l);
      } else {
        weights[name] = this.dequantizeQ4(new HalfFloatArray(meta.s), new HalfFloatArray(meta.m), rawBytes, meta.l);
      }
    }

    return {
      config: parsed.config,
      tokenizer: parsed.tokenizer,
      quantMode: parsed.quantMode || 'fp4',
      weights
    };
  }
}
