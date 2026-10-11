import { ChunkIndexer } from './chunk-indexer.js';

export class JSONLLMCodec {
  static FORMAT_VERSION = 'jsonllm/v1';

  /**
   * Block Quantization: Converts Float32Array to 4-bit nibbles (Q4_0)
   */
  static quantizeQ4(floatArray, blockSize = 32) {
    const len = floatArray.length;
    const numBlocks = Math.ceil(len / blockSize);
    const packedBytes = new Uint8Array(numBlocks * (blockSize / 2));
    const scales = new Float32Array(numBlocks);
    const mins = new Float32Array(numBlocks);

    let bytePtr = 0;

    for (let b = 0; b < numBlocks; b++) {
      const start = b * blockSize;
      const end = Math.min(start + blockSize, len);

      let min = Infinity;
      let max = -Infinity;

      for (let i = start; i < end; i++) {
        const val = floatArray[i];
        if (val < min) min = val;
        if (val > max) max = val;
      }

      const range = max - min;
      const scale = range > 1e-7 ? range / 15.0 : 0.0;
      scales[b] = scale;
      mins[b] = min;

      for (let i = start; i < end; i += 2) {
        const v0 = floatArray[i];
        const v1 = i + 1 < end ? floatArray[i + 1] : min;

        const q0 = scale > 0 ? Math.min(15, Math.max(0, Math.round((v0 - min) / scale))) : 0;
        const q1 = scale > 0 ? Math.min(15, Math.max(0, Math.round((v1 - min) / scale))) : 0;

        packedBytes[bytePtr++] = (q1 << 4) | (q0 & 0x0f);
      }
    }

    return {
      numElements: len,
      scales: Array.from(scales),
      mins: Array.from(mins),
      packedBytes
    };
  }

  /**
   * Dequantizes 4-bit nibbles back to full precision Float32Array
   */
  static dequantizeQ4(quantData, blockSize = 32) {
    const { numElements, scales, mins, packedBytes } = quantData;
    const out = new Float32Array(numElements);
    const numBlocks = scales.length;

    let bytePtr = 0;
    let outPtr = 0;

    for (let b = 0; b < numBlocks; b++) {
      const scale = scales[b];
      const min = mins[b];

      for (let i = 0; i < blockSize / 2; i++) {
        if (outPtr >= numElements) break;
        const byte = packedBytes[bytePtr++];

        const q0 = byte & 0x0f;
        const q1 = (byte >> 4) & 0x0f;

        out[outPtr++] = q0 * scale + min;
        if (outPtr < numElements) {
          out[outPtr++] = q1 * scale + min;
        }
      }
    }

    return out;
  }

  /**
   * Deflate compression stream to Base64 string
   */
  static async compressBytesToBase64(bytes) {
    if (typeof CompressionStream !== 'undefined') {
      const cs = new CompressionStream('deflate');
      const writer = cs.writable.getWriter();
      writer.write(bytes);
      writer.close();

      const response = new Response(cs.readable);
      const compressedBuffer = await response.arrayBuffer();
      const compBytes = new Uint8Array(compressedBuffer);

      let binary = '';
      const len = compBytes.byteLength;
      for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(compBytes[i]);
      }
      return { base64: btoa(binary), compressed: true };
    }

    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return { base64: btoa(binary), compressed: false };
  }

  /**
   * Base64 to decompressed bytes via DecompressionStream
   */
  static async decompressBase64ToBytes(base64Str, wasCompressed = true) {
    const binary = atob(base64Str);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    if (wasCompressed && typeof DecompressionStream !== 'undefined') {
      const ds = new DecompressionStream('deflate');
      const writer = ds.writable.getWriter();
      writer.write(bytes);
      writer.close();

      const response = new Response(ds.readable);
      const decompressedBuffer = await response.arrayBuffer();
      return new Uint8Array(decompressedBuffer);
    }

    return bytes;
  }

  /**
   * Converts a model instance to a JSONLLM envelope, slices it via SI-Radix,
   * and saves it to IndexedDB.
   */
  static async convertAndStoreToIndexedDB(modelInstance, storageService, modelName = 'browser-nano-llm') {
    const config = {
      dim: modelInstance.dim,
      layers: modelInstance.layers,
      heads: modelInstance.heads,
      vocabSize: modelInstance.vocabSize,
      maxSeqLen: modelInstance.maxSeqLen,
      stepCount: modelInstance.stepCount
    };

    const tokenizer = {
      tokenToId: Object.fromEntries(modelInstance.tokenizer.tokenToId),
      idToToken: Object.fromEntries(modelInstance.tokenizer.idToToken)
    };

    const compressedTensors = {};

    for (const key in modelInstance.weights) {
      const rawWeights = modelInstance.weights[key];
      const q4 = this.quantizeQ4(rawWeights);
      const { base64, compressed } = await this.compressBytesToBase64(q4.packedBytes);

      compressedTensors[key] = {
        numElements: q4.numElements,
        scales: q4.scales,
        mins: q4.mins,
        compressed,
        data_b64: base64
      };
    }

    const jsonllmObject = {
      format: this.FORMAT_VERSION,
      modelName,
      createdAt: new Date().toISOString(),
      config,
      tokenizer,
      compressedTensors
    };

    const jsonString = JSON.stringify(jsonllmObject);
    const jsonBlob = new Blob([jsonString], { type: 'application/json' });

    // Build SI-Radix 4 MB segments from the serialized JSON
    const { manifest, segments } = await ChunkIndexer.splitAndIndex(jsonBlob, modelName);

    // Persist full JSON record, manifest, and segment slices into IndexedDB
    await storageService.putJSONLLM({
      modelName,
      format: this.FORMAT_VERSION,
      createdAt: jsonllmObject.createdAt,
      config,
      jsonString
    });

    await storageService.saveManifest(manifest);
    await storageService.batchPutChunks(segments);

    return {
      jsonllmObject,
      jsonString,
      manifest,
      segments
    };
  }

  /**
   * Rehydrates a model instance from an IndexedDB-stored JSONLLM package
   */
  static async loadFromIndexedDB(storageService, modelName) {
    const record = await storageService.getJSONLLM(modelName);
    if (!record || !record.jsonString) {
      throw new Error(`JSONLLM record for "${modelName}" not found in IndexedDB.`);
    }

    const payload = JSON.parse(record.jsonString);
    const { config, tokenizer, compressedTensors } = payload;
    const restoredWeights = {};

    for (const key in compressedTensors) {
      const t = compressedTensors[key];
      const packedBytes = await this.decompressBase64ToBytes(t.data_b64, t.compressed);
      restoredWeights[key] = this.dequantizeQ4({
        numElements: t.numElements,
        scales: t.scales,
        mins: t.mins,
        packedBytes
      });
    }

    return {
      config,
      tokenizer,
      weights: restoredWeights
    };
  }
}
