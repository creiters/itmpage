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

    if (this.tokenToId.has('<|im_start|>')) {
      tokenIds.push(this.tokenToId.get('<|im_start|>'));
    }

    const words = cleanText.split(/(\s+)/);
    for (const word of words) {
      if (!word) continue;
      const bpeWord = word.startsWith(' ') ? 'Ġ' + word.trim() : word;

      if (this.tokenToId.has(bpeWord)) {
        tokenIds.push(this.tokenToId.get(bpeWord));
      } else if (this.tokenToId.has(word)) {
        tokenIds.push(this.tokenToId.get(word));
      } else {
        for (let i = 0; i < word.length; i++) {
          const ch = word[i];
          tokenIds.push(this.tokenToId.has(ch) ? this.tokenToId.get(ch) : 0);
        }
      }
    }
    return tokenIds.length > 0 ? tokenIds : [1];
  }

  decode(tokenId) {
    const raw = this.idToToken.get(tokenId);
    if (!raw) return '';
    if (raw.startsWith('<|') && raw.endsWith('|>')) return '';
    if (raw === '<s>' || raw === '</s>') return '';
    return raw.replace(/Ġ/g, ' ').replace(/ /g, ' ');
  }
}
