export class SimpleBPETokenizer {
  constructor(tokens, scores) {
    this.tokens = tokens || [];
    this.tokenToId = new Map();
    this.idToToken = new Map();

    for (let i = 0; i < this.tokens.length; i++) {
      this.tokenToId.set(this.tokens[i], i);
      this.idToToken.set(i, this.tokens[i]);
    }
  }

  encode(text) {
    const tokenIds = [];
    // Basic greedy/whitespace byte fallback
    const words = text.split(/(\s+|[^\w\s])/);
    for (const w of words) {
      if (!w) continue;
      const formatted = w.replace(' ', 'Ġ');
      if (this.tokenToId.has(formatted)) {
        tokenIds.push(this.tokenToId.get(formatted));
      } else if (this.tokenToId.has(w)) {
        tokenIds.push(this.tokenToId.get(w));
      } else {
        // Fallback byte-by-byte encoding
        const bytes = new TextEncoder().encode(w);
        for (const b of bytes) {
          const charRepr = String.fromCharCode(b);
          tokenIds.push(this.tokenToId.get(charRepr) || 0);
        }
      }
    }
    return tokenIds;
  }

  decode(tokenId) {
    const token = this.idToToken.get(tokenId) || '';
    return token.replace('Ġ', ' ').replace(/<[^>]+>/g, '');
  }
}
