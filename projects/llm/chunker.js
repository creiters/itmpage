export class ChunkManager {
  static CHUNK_SIZE = 1024 * 1024; // 1 MB slice boundary

  static async digestSHA256(buffer) {
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }

  /**
   * Slices binary blob into 1MB linked chunks with contextual graph metadata
   */
  static async splitBlob(modelBlob, modelName, onProgress) {
    const totalBytes = modelBlob.size;
    const totalChunks = Math.ceil(totalBytes / this.CHUNK_SIZE);
    const chunkList = [];
    let prevHash = 'GENESIS_ROOT';

    for (let index = 0; index < totalChunks; index++) {
      const byteStart = index * this.CHUNK_SIZE;
      const byteEnd = Math.min(byteStart + this.CHUNK_SIZE, totalBytes);
      const sliceBlob = modelBlob.slice(byteStart, byteEnd);
      const binary = await sliceBlob.arrayBuffer();

      const hash = await this.digestSHA256(binary);
      const estimatedLayer = Math.floor((index / totalChunks) * 32);

      const metadata = {
        modelName,
        chunkId: `${modelName}_c${index}`,
        index,
        totalChunks,
        byteOffset: byteStart,
        byteLength: binary.byteLength,
        prevHash,
        nextHash: null,
        hash,
        context: {
          layerId: estimatedLayer,
          byteSpan: { start: byteStart, end: byteEnd },
          tensorNamespace: `smollm.layers.${estimatedLayer}.weights`
        }
      };

      if (chunkList.length > 0) {
        chunkList[chunkList.length - 1].metadata.nextHash = hash;
      }

      chunkList.push({ metadata, binary });
      prevHash = hash;

      if (onProgress) {
        onProgress({
          index: index + 1,
          total: totalChunks,
          percent: ((index + 1) / totalChunks) * 100
        });
      }
    }

    return chunkList;
  }

  /**
   * Reassembles sequential chunks and validates contextual DAG integrity
   */
  static async reassemble(orderedChunks) {
    if (!orderedChunks || orderedChunks.length === 0) {
      throw new Error('No chunks provided for reassembly.');
    }

    let calculatedTotalLength = 0;

    for (let i = 0; i < orderedChunks.length; i++) {
      const chunk = orderedChunks[i];
      const recalculated = await this.digestSHA256(chunk.binary);

      if (recalculated !== chunk.metadata.hash) {
        throw new Error(`Data corruption detected at chunk index ${i}`);
      }

      if (i > 0 && chunk.metadata.prevHash !== orderedChunks[i - 1].metadata.hash) {
        throw new Error(`Broken DAG link between chunk ${i - 1} and ${i}`);
      }

      calculatedTotalLength += chunk.binary.byteLength;
    }

    const contiguousBuffer = new Uint8Array(calculatedTotalLength);

    for (const chunk of orderedChunks) {
      contiguousBuffer.set(new Uint8Array(chunk.binary), chunk.metadata.byteOffset);
    }

    return contiguousBuffer.buffer;
  }

  /**
   * Encapsulates a chunk into a standalone contextual container file
   */
  static exportChunkEnvelope(chunkRecord) {
    const metaString = JSON.stringify(chunkRecord.metadata);
    const metaBytes = new TextEncoder().encode(metaString);
    const totalSize = 4 + metaBytes.byteLength + chunkRecord.binary.byteLength;
    const outputBuffer = new Uint8Array(totalSize);

    new DataView(outputBuffer.buffer).setUint32(0, metaBytes.byteLength, true);
    outputBuffer.set(metaBytes, 4);
    outputBuffer.set(new Uint8Array(chunkRecord.binary), 4 + metaBytes.byteLength);

    return new Blob([outputBuffer], { type: 'application/octet-stream' });
  }
}
