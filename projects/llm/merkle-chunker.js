export class MerkleChunker {
  static CHUNK_SIZE = 4 * 1024 * 1024; // Strict 4 MB boundary

  static async sha256(buffer) {
    if (!buffer || buffer.byteLength === 0) return 'EMPTY_BUFFER';
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }

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

  static async splitToMerkleChunks(modelBlob, modelName, onProgress) {
    const totalBytes = modelBlob.size;
    const totalChunks = Math.ceil(totalBytes / this.CHUNK_SIZE);
    const rawChunks = [];
    const leafHashes = [];

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

    const { rootHash } = await this.buildMerkleTree(leafHashes);
    const finalizedChunks = [];

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

      finalizedChunks.push({ metadata, binary: item.binary });

      if (onProgress) {
        onProgress({ phase: 'linking', current: i + 1, total: totalChunks });
      }
    }

    return { rootHash, chunks: finalizedChunks };
  }

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
        throw new Error(`Data corruption detected at chunk index ${i}`);
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
}
