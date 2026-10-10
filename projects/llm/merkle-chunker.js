export class MerkleChunker {
  static CHUNK_SIZE = 4 * 1024 * 1024; // 4 MB boundary

  /**
   * Computes SHA-256 hash string for an ArrayBuffer or TypedArray
   */
  static async sha256(buffer) {
    if (!buffer || buffer.byteLength === 0) return 'EMPTY_BUFFER';
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }

  /**
   * Builds a binary Merkle tree over an array of leaf hash strings
   */
  static async buildMerkleTree(leafHashes) {
    if (!leafHashes || leafHashes.length === 0) {
      return { rootHash: 'EMPTY_ROOT', treeLevels: [] };
    }

    let currentLevel = [...leafHashes];
    const tree = [currentLevel];

    while (currentLevel.length > 1) {
      const nextLevel = [];
      for (let i = 0; i < currentLevel.length; i += 2) {
        const left = currentLevel[i];
        const right = i + 1 < currentLevel.length ? currentLevel[i + 1] : left;
        const combined = new TextEncoder().encode(left + right);
        const parentHash = await this.sha256(combined);
        nextLevel.push(parentHash);
      }
      tree.push(nextLevel);
      currentLevel = nextLevel;
    }

    return {
      rootHash: currentLevel[0] || 'EMPTY_ROOT',
      treeLevels: tree
    };
  }

  /**
   * Splits a model blob into 4 MB contextual envelopes bound by a Merkle tree
   */
  static async splitToMerkleChunks(modelBlob, modelName, onProgress) {
    const totalBytes = modelBlob.size;
    const totalChunks = Math.ceil(totalBytes / this.CHUNK_SIZE);
    const rawChunks = [];
    const leafHashes = [];

    // 1. Slice and calculate leaf hashes
    for (let index = 0; index < totalChunks; index++) {
      const start = index * this.CHUNK_SIZE;
      const end = Math.min(start + this.CHUNK_SIZE, totalBytes);
      const slice = modelBlob.slice(start, end);
      const binary = await slice.arrayBuffer();
      const hash = await this.sha256(binary);

      leafHashes.push(hash);
      rawChunks.push({ index, start, end, binary, hash });

      if (onProgress) {
        onProgress({ phase: 'hashing', current: index + 1, total: totalChunks });
      }
    }

    // 2. Compute root hash
    const { rootHash } = await this.buildMerkleTree(leafHashes);
    const finalizedChunks = [];

    // 3. Assemble envelopes with context DAG links
    for (let i = 0; i < totalChunks; i++) {
      const item = rawChunks[i];
      const nextChunkId = i + 1 < totalChunks ? `${modelName}_chunk_4mb_${i + 1}` : null;
      const prevChunkId = i > 0 ? `${modelName}_chunk_4mb_${i - 1}` : null;

      const metadata = {
        chunkId: `${modelName}_chunk_4mb_${i}`,
        modelName,
        index: i,
        totalChunks,
        byteOffset: item.start,
        byteLength: item.binary.byteLength,
        chunkHash: item.hash,
        hash: item.hash,
        merkleRoot: rootHash,
        prevChunkId,
        nextChunkId,
        context: {
          layerRange: [
            Math.floor((item.start / totalBytes) * 30),
            Math.floor((item.end / totalBytes) * 30)
          ],
          canYieldNext: nextChunkId !== null,
          targetNextContext: nextChunkId ? `context://chunk/${nextChunkId}` : 'context://tail'
        }
      };

      finalizedChunks.push({
        metadata,
        binary: item.binary
      });

      if (onProgress) {
        onProgress({ phase: 'linking', current: i + 1, total: totalChunks });
      }
    }

    return { rootHash, chunks: finalizedChunks };
  }

  /**
   * Verifies individual chunk hashes and reassembles them into linear memory
   */
  static async reassemble(orderedChunks, expectedRoot = null) {
    if (!orderedChunks || orderedChunks.length === 0) {
      throw new Error('Reassembly failed: No chunks provided.');
    }

    const leafHashes = [];
    let calculatedTotalLength = 0;

    for (let i = 0; i < orderedChunks.length; i++) {
      const c = orderedChunks[i];
      if (!c.binary || c.binary.byteLength === 0) {
        throw new Error(`Chunk at index ${i} has an empty or detached buffer.`);
      }

      const expectedHash = c.metadata.chunkHash || c.metadata.hash;
      const calculatedHash = await this.sha256(c.binary);

      if (calculatedHash !== expectedHash) {
        throw new Error(
          `Data corruption detected at chunk index ${i} (expected ${expectedHash.slice(0, 8)}, got ${calculatedHash.slice(0, 8)})`
        );
      }

      leafHashes.push(calculatedHash);
      calculatedTotalLength += c.binary.byteLength;
    }

    if (expectedRoot) {
      const { rootHash } = await this.buildMerkleTree(leafHashes);
      if (rootHash !== expectedRoot) {
        throw new Error(`Merkle Root mismatch: expected ${expectedRoot}, got ${rootHash}`);
      }
    }

    const unifiedBuffer = new Uint8Array(calculatedTotalLength);
    for (const c of orderedChunks) {
      unifiedBuffer.set(new Uint8Array(c.binary), c.metadata.byteOffset);
    }

    return unifiedBuffer.buffer;
  }

  /**
   * Bundles a single chunk and its contextual metadata into an exportable binary
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
