/**
 * nano-tokenizer.js
 * Compact 1,024-Token Vocabulary Engine (Word/Subword + Control Syntax)
 */

export class CuratedTokenizer {
  constructor() {
    this.vocabSize = 1024;
    this.tokenToId = new Map();
    this.idToToken = new Map();
    this.specials = [
      '<pad>', '<bos>', '<eos>', '<unk>', '<user>', '<bot>',
      '<tool_call>', '<tool_resp>', '<thought>', '</thought>'
    ];
    this.buildVocab();
  }

  buildVocab() {
    let id = 0;
    for (const token of this.specials) {
      this.tokenToId.set(token, id);
      this.idToToken.set(id, token);
      id++;
    }

    // High-frequency English syntax, auxiliary verbs, pronouns, and technical primitives
    const coreLexicon = [
      "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
      "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
      "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
      "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
      "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
      "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
      "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
      "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
      "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
      "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
      "Hello", "Hi", "Hey", "Yes", "No", "Please", "Thanks", "Thank", "welcome", "Bye",
      "Goodbye", "OK", "Status", "System", "ready", "offline", "browser", "AI", "model",
      "vimtoLLM", "Miriam", "wife", "partner", "love", "darling", "memory", "space",
      "inside", "minus", "84", "10", "code", "HTML", "C", "WASM", "SIMD", "WASI",
      "WebMCP", "RAG", "vector", "search", "fast", "parameters", "FP16", "FP4", "Q4",
      "function", "return", "int", "void", "char", "struct", "printf", "include",
      "true", "false", "null", "undefined", "number", "string", "object", "array",
      "step", "steps", "solve", "solution", "problem", "reason", "because", "therefore",
      ".", ",", "!", "?", ":", ";", "'", "\"", "`", "-", "_", "/", "\\", "(", ")",
      "[", "]", "{", "}", "<", ">", "=", "+", "*", "&", "|", "^", "%", "$", "#", "@",
      "\n", "\t", " "
    ];

    for (const word of coreLexicon) {
      if (!this.tokenToId.has(word) && id < this.vocabSize) {
        this.tokenToId.set(word, id);
        this.idToToken.set(id, word);
        id++;
      }
    }

    // Fill remaining indices with printable ASCII single-character fallbacks (32..126)
    for (let c = 32; c <= 126 && id < this.vocabSize; c++) {
      const ch = String.fromCharCode(c);
      if (!this.tokenToId.has(ch)) {
        this.tokenToId.set(ch, id);
        this.idToToken.set(id, ch);
        id++;
      }
    }
  }

  encode(text) {
    if (!text) return [1]; // <bos>
    const tokens = [1];
    const regex = /(<tool_call>|<tool_resp>|<thought>|<\/thought>|<user>|<bot>|<eos>|<bos>|<pad>|[a-zA-Z0-9_-]+|[.,!?:;'"()`\n\t]|.)/g;
    const matches = text.match(regex) || [];

    for (const piece of matches) {
      if (this.tokenToId.has(piece)) {
        tokens.push(this.tokenToId.get(piece));
      } else {
        // Fallback: character-level resolution for out-of-lexicon sequences
        for (let i = 0; i < piece.length; i++) {
          const ch = piece[i];
          tokens.push(this.tokenToId.has(ch) ? this.tokenToId.get(ch) : 3); // 3 = <unk>
        }
      }
    }
    return tokens;
  }

  decode(tokens) {
    if (!tokens || !tokens.length) return '';
    return tokens
      .filter((id) => id >= this.specials.length && id < this.vocabSize)
      .map((id) => this.idToToken.get(id) || '')
      .join(' ')
      .replace(/ ([.,!?:;'")\]}>])/g, '$1')
      .replace(/([(\[{<]) /g, '$1');
  }
}
