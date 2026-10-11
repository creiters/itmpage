export const DOMAIN_SPECS = {
  dialog: {
    modelName: 'agent-dialog',
    dim: 64,
    layers: 2,
    heads: 2,
    maxSeqLen: 48,
    specials: ['<pad>', '<bos>', '<eos>', '<user>', '<bot>'],
    asciiRange: [32, 126]
  },
  clang: {
    modelName: 'agent-clang',
    dim: 64,
    layers: 2,
    heads: 2,
    maxSeqLen: 48,
    specials: ['<pad>', '<bos>', '<eos>', '<context>', '<c_code>'],
    customTokens: [
      '\n', '\t', ' ', '#include', '<stdio.h>', 'int', 'void', 'char', 'float',
      'struct', 'if', 'else', 'for', 'while', 'return', 'printf', '(', ')',
      '{', '}', '[', ']', ';', ',', '*', '&', '=', '+', '-', '/', '"', "'"
    ]
  },
  html5: {
    modelName: 'agent-html5',
    dim: 64,
    layers: 2,
    heads: 2,
    maxSeqLen: 48,
    specials: ['<pad>', '<bos>', '<eos>', '<context>', '<html_code>'],
    customTokens: [
      '\n', '\t', ' ', '<!DOCTYPE html>', '<html>', '</html>', '<head>', '</head>',
      '<title>', '</title>', '<body>', '</body>', '<main>', '</main>', '<div', '</div>',
      '<p>', '</p>', '<span', '</span>', 'class="', 'id="', '>', '<', '/>', '=', '"'
    ]
  }
};

export class DomainTokenizer {
  constructor(spec) {
    this.spec = spec;
    this.tokenToId = new Map();
    this.idToToken = new Map();
    this.build();
  }

  build() {
    let id = 0;
    for (const s of this.spec.specials) {
      this.tokenToId.set(s, id);
      this.idToToken.set(id, s);
      id++;
    }
    if (this.spec.customTokens) {
      for (const t of this.spec.customTokens) {
        if (!this.tokenToId.has(t)) {
          this.tokenToId.set(t, id);
          this.idToToken.set(id, t);
          id++;
        }
      }
    }
    if (this.spec.asciiRange) {
      const [start, end] = this.spec.asciiRange;
      for (let c = start; c <= end; c++) {
        const ch = String.fromCharCode(c);
        if (!this.tokenToId.has(ch)) {
          this.tokenToId.set(ch, id);
          this.idToToken.set(id, ch);
          id++;
        }
      }
    }
    this.vocabSize = id;
  }

  encode(text) {
    const tokens = [1];
    let cursor = 0;
    while (cursor < text.length) {
      let matched = false;
      if (this.spec.customTokens) {
        for (const tok of this.spec.customTokens) {
          if (text.startsWith(tok, cursor)) {
            tokens.push(this.tokenToId.get(tok));
            cursor += tok.length;
            matched = true;
            break;
          }
        }
      }
      if (!matched) {
        const ch = text[cursor];
        tokens.push(this.tokenToId.has(ch) ? this.tokenToId.get(ch) : 0);
        cursor++;
      }
    }
    return tokens;
  }

  decode(tokens) {
    return tokens
      .filter((t) => t > 2)
      .map((t) => this.idToToken.get(t) || '')
      .join('');
  }
}
