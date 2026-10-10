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
    if (!text) return [1];
    const cleanText = text.trim();
    const tokenIds = [];

    // Check BOS
    if (this.tokenToId.has('<|im_start|>')) {
      tokenIds.push(this.tokenToId.get('<|im_start|>'));
    }

    // Split words preserving whitespace
    const words = cleanText.split(/(\s+)/);
    for (let word of words) {
      if (!word) continue;
      const bpeWord = word.startsWith(' ') ? 'Ġ' + word.trim() : word;

      if (this.tokenToId.has(bpeWord)) {
        tokenIds.push(this.tokenToId.get(bpeWord));
      } else if (this.tokenToId.has(word)) {
        tokenIds.push(this.tokenToId.get(word));
      } else {
        // Character fallback
        for (let i = 0; i < word.length; i++) {
          const ch = word[i];
          if (this.tokenToId.has(ch)) {
            tokenIds.push(this.tokenToId.get(ch));
          } else {
            tokenIds.push(0); // unknown
          }
        }
      }
    }
    return tokenIds.length > 0 ? tokenIds : [1];
  }

  decode(tokenId) {
    const raw = this.idToToken.get(tokenId);
    if (!raw) return '';

    // Ignore special system control tags
    if (raw.startsWith('<|') && raw.endsWith('|>')) return '';
    if (raw === '<s>' || raw === '</s>') return '';

    // Convert BPE space markers back to space
    return raw.replace(/Ġ/g, ' ').replace(/ /g, ' ');
  }
}
