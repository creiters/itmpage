export class MerkleChunker {
  static CHUNK_SIZE = 4 * 1024 * 1024; // Strict 4 MB boundary

  static async sha256(buffer) {
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }

  /**
   * Computes binary Merkle tree root and branch proofs
   */
  static async buildMerkleTree(leafHashes) {
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
   * Splits GGUF model into 4MB linked contextual envelopes with Merkle verification
   */
  static async splitToMerkleChunks(modelBlob, modelName, onProgress) {
    const totalBytes = modelBlob.size;
    const totalChunks = Math.ceil(totalBytes / this.CHUNK_SIZE);
    const rawChunks = [];
    const leafHashes = [];

    // Phase 1: Slice and calculate leaf hashes
    for (let index = 0; index < totalChunks; index++) {
      const start = index * this.CHUNK_SIZE;
      const end = Math.min(start + this.CHUNK_SIZE, totalBytes);
      const binary = await modelBlob.slice(start, end).arrayBuffer();
      const hash = await this.sha256(binary);

      leafHashes.push(hash);
      rawChunks.push({
        index,
        start,
        end,
        binary,
        hash
      });

      if (onProgress) {
        onProgress({ phase: 'hashing', current: index + 1, total: totalChunks });
      }
    }

    // Phase 2: Compute Merkle Tree
    const { rootHash } = await this.buildMerkleTree(leafHashes);

    // Phase 3: Build contextual DAG envelope with next-chunk context links
    const finalizedChunks = [];
    for (let i = 0; i < totalChunks; i++) {
      const item = rawChunks[i];
      const nextChunkId = i + 1 < totalChunks ? `${modelName}_chunk_4mb_${i + 1}` : null;
      const prevChunkId = i > 0 ? `${modelName}_chunk_4mb_${i - 1}` : null;

      // Approximate layer bounds inside this 4MB chunk (SmolLM ~30 layers)
      const startLayer = Math.floor((item.start / totalBytes) * 30);
      const endLayer = Math.floor((item.end / totalBytes) * 30);

      const metadata = {
        chunkId: `${modelName}_chunk_4mb_${i}`,
        modelName,
        index: i,
        totalChunks,
        byteOffset: item.start,
        byteLength: item.binary.byteLength,
        chunkHash: item.hash,
        merkleRoot: rootHash,
        prevChunkId,
        nextChunkId,
        // Embedded context link
        context: {
          layerRange: [startLayer, endLayer],
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
   * Reassembles chunks in Merkle order with hash integrity check
   */
  static async reassemble(orderedChunks, expectedRoot) {
    const leafHashes = [];
    for (const c of orderedChunks) {
      const h = await this.sha256(c.binary);
      if (h !== c.metadata.chunkHash) {
        throw new Error(`Merkle Leaf Tampering detected at chunk ${c.metadata.index}`);
      }
      leafHashes.push(h);
    }

    const { rootHash } = await this.buildMerkleTree(leafHashes);
    if (expectedRoot && rootHash !== expectedRoot) {
      throw new Error(`Merkle Root mismatch: expected ${expectedRoot}, got ${rootHash}`);
    }

    const totalLength = orderedChunks.reduce((acc, c) => acc + c.binary.byteLength, 0);
    const unified = new Uint8Array(totalLength);

    for (const c of orderedChunks) {
      unified.set(new Uint8Array(c.binary), c.metadata.byteOffset);
    }

    return unified.buffer;
  }
}
