import { HalfFloatArray } from './jsonllm-codec.js';

export class LocalRAG {
  constructor(dim = 64) {
    this.dim = dim;
    this.storage = null;
  }

  async init(storageService) {
    this.storage = storageService;
  }

  embed(text) {
    const vec = new HalfFloatArray(this.dim);
    const cleaned = text.toLowerCase().trim();
    if (!cleaned) return vec;

    const n = 3;
    for (let i = 0; i <= cleaned.length - n; i++) {
      const gram = cleaned.substring(i, i + n);
      let hash = 2166136261;
      for (let j = 0; j < gram.length; j++) {
        hash ^= gram.charCodeAt(j);
        hash = Math.imul(hash, 16777619);
      }
      const idx = Math.abs(hash) % this.dim;
      vec[idx] += 1.0;
    }

    let norm = 0.0;
    for (let i = 0; i < this.dim; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < this.dim; i++) vec[i] /= norm;
    }
    return vec;
  }

  cosineSimilarity(vA, vB) {
    let dot = 0.0;
    for (let i = 0; i < this.dim; i++) dot += vA[i] * vB[i];
    return dot;
  }

  async indexDocument(docId, agentId = 'vimto-dialog', text = '', metadata = {}) {
    const vector = Array.from(this.embed(text)).map((v) => Number(v.toFixed(4)));
    return await this.storage.putRAGDoc({ docId, agentId, text, metadata, vector });
  }

  async retrieve(query, agentId = 'vimto-dialog', topK = 1) {
    const queryVec = this.embed(query);
    const docs = await this.storage.getAllRAGDocs();
    if (!docs.length) return [];

    // Filter by target agent or universal entries
    const agentScoped = docs.filter((d) => d.agentId === agentId || d.agentId === 'all');
    if (!agentScoped.length) return [];

    const scored = agentScoped.map((d) => ({
      ...d,
      score: this.cosineSimilarity(queryVec, new HalfFloatArray(d.vector))
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }
}
