/**
 * nano-json-tokenizer.js
 * Curated 1,024-Token Vocabulary with Structural JSON Primitives & ASCII Fallbacks
 */

export class NanoJsonTokenizer {
  constructor(config = {}) {
    this.vocabSize = config.vocabSize || 1024;
    this.tokenToId = new Map();
    this.idToToken = new Map();

    this.specials = [
      '<pad>', '<bos>', '<eos>', '<unk>',
      '<user>', '<bot>', '<thought>', '</thought>',
      '<tool_call>', '<tool_resp>', '<context>', '<response>'
    ];

    this.jsonTokens = [
      '{"u":', '{"b":', '"u":', '"b":', '"u"', '"b"',
      '": "', '","', '":', '",', '"}', '}]', '[{',
      '{"name":', '"name":', '"params":', '"arguments":', '"result":', '"error":',
      '{', '}', '[', ']', ':', ',', '"',
      'true', 'false', 'null',
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ];

    this.coreWords = [
      "Hello", "Hi", "Hey", "How", "are", "you", "I", "am", "doing", "well",
      "great", "fine", "good", "What", "is", "your", "name", "My", "JSONLLM",
      "vimtoLLM", "Miriam", "wife", "partner", "love", "darling", "sweetheart",
      "think", "thinking", "assist", "help", "together", "offline", "browser",
      "AI", "assistant", "Yes", "No", "can", "with", "code", "HTML", "C",
      "and", "dialogue", "today", "Who", "created", "runs", "entirely", "on",
      "device", "Thanks", "Thank", "welcome", "Goodbye", "Bye", "ready",
      "System", "Status", "OK", "Please", "ask", "a", "question", "nice",
      "to", "meet", "parameters", "model", "4M", "memory", "space", "inside",
      "minus", "84", "10", "fast", "WASM", "SIMD", "WASI", "WebMCP", "RAG",
      "vector", "search", "store", "save", "data", "dataset", "epoch", "epochs",
      "coroutine", "step", "loss", "the", "be", "of", "in", "that", "have",
      "it", "for", "not", "he", "as", "do", "at", "this", "but", "by", "from",
      "they", "we", "or", "an", "will", "my", "one", "all", "would", "there",
      "their", "so", "up", "out", "if", "about", "which", "get", "when", "make",
      "like", "time", "just", "know", "take", "into", "year", "your", "some",
      "could", "them", "see", "other", "than", "then", "now", "look", "only",
      "come", "its", "over", "also", "back", "after", "use", "two", "our",
      "first", "way", "even", "new", "want", "because", "any", "these", "give",
      "\n", "\t", " "
    ];

    this.buildVocab();
  }

  buildVocab() {
    let id = 0;
    for (const spec of this.specials) {
      this.tokenToId.set(spec, id);
      this.idToToken.set(id, spec);
      id++;
    }

    const sortedJsonTokens = [...this.jsonTokens].sort((a, b) => b.length - a.length);
    for (const tok of sortedJsonTokens) {
      if (!this.tokenToId.has(tok)) {
        this.tokenToId.set(tok, id);
        this.idToToken.set(id, tok);
        id++;
      }
    }

    for (const word of this.coreWords) {
      if (!this.tokenToId.has(word) && id < this.vocabSize) {
        this.tokenToId.set(word, id);
        this.idToToken.set(id, word);
        id++;
      }
    }

    for (let c = 32; c <= 126 && id < this.vocabSize; c++) {
      const ch = String.fromCharCode(c);
      if (!this.tokenToId.has(ch)) {
        this.tokenToId.set(ch, id);
        this.idToToken.set(id, ch);
        id++;
      }
    }

    this.activeVocabSize = id;
    this.multiCharTokens = Array.from(this.tokenToId.keys())
      .filter((tok) => tok.length > 1)
      .sort((a, b) => b.length - a.length);
  }

  encode(text, addBos = true) {
    if (!text) return addBos ? [1] : [];
    const tokens = addBos ? [1] : [];
    let cursor = 0;
    const len = text.length;

    while (cursor < len) {
      let matched = false;
      for (let i = 0; i < this.multiCharTokens.length; i++) {
        const candidate = this.multiCharTokens[i];
        if (text.startsWith(candidate, cursor)) {
          tokens.push(this.tokenToId.get(candidate));
          cursor += candidate.length;
          matched = true;
          break;
        }
      }

      if (!matched) {
        const ch = text[cursor];
        tokens.push(this.tokenToId.has(ch) ? this.tokenToId.get(ch) : 3);
        cursor++;
      }
    }
    return tokens;
  }

  decode(tokens, stripSpecials = false) {
    if (!tokens || !tokens.length) return '';
    let out = '';
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      if (stripSpecials && id < this.specials.length) continue;
      out += this.idToToken.get(id) || '';
    }
    return out;
  }

  encodeJSON(obj, addBos = true) {
    return this.encode(JSON.stringify(obj), addBos);
  }

  decodeJSON(tokens) {
    try {
      return JSON.parse(this.decode(tokens, true));
    } catch {
      return null;
    }
  }
}
