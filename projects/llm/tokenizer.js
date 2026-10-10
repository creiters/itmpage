export class SimpleBPETokenizer {
  constructor(tokens, scores) {
    this.tokens = tokens || [];
    this.tokenToId = new Map();
    this.idToToken = new Map();

    for (let i = 0; i < this.tokens.length; i++) {
      const t = this.tokens[i];
      this.tokenToId.set(t, i);
      this.idToToken.set(i, t);
    }
  }

  encode(text) {
    if (!text) return [];
    const tokenIds = [];
    // Clean token splitting for sentence fragments
    const parts = text.split(/(\s+|[^\w\s])/);

    for (const p of parts) {
      if (!p) continue;
      const formatted = 'Ġ' + p; // BPE leading whitespace mark
      if (this.tokenToId.has(formatted)) {
        tokenIds.push(this.tokenToId.get(formatted));
      } else if (this.tokenToId.has(p)) {
        tokenIds.push(this.tokenToId.get(p));
      } else {
        const bytes = new TextEncoder().encode(p);
        for (const b of bytes) {
          const ch = String.fromCharCode(b);
          if (this.tokenToId.has(ch)) {
            tokenIds.push(this.tokenToId.get(ch));
          }
        }
      }
    }
    return tokenIds.length > 0 ? tokenIds : [1];
  }

  decode(tokenId) {
    const raw = this.idToToken.get(tokenId);
    if (!raw) return '';

    // Ignore special system control tokens
    if (raw.startsWith('<|') && raw.endsWith('|>')) return '';
    if (raw === '<s>' || raw === '</s>') return '';

    // Replace byte-pair whitespace markers with regular space
    return raw.replace(/Ġ/g, ' ').replace(/ /g, ' ');
  }
}
