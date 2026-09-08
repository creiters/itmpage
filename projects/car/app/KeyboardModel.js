class TrieNode {
  constructor() {
    this.children = new Map();
    this.isWord = false;
    this.fullWord = '';
  }
}

export class KeyboardModel {
  constructor() {
    this.audioCtx = null;
    this.tries = new Map();
  }

  async fetchConfig() {
    const res = await fetch('./data/keyboard.json');
    if (!res.ok) throw new Error('Cannot load data/keyboard.json');
    const config = await res.json();
    this.buildLanguageTries(config);
    return config;
  }

  buildLanguageTries(config) {
    for (const [langKey, langObj] of Object.entries(config.languages)) {
      const root = new TrieNode();
      const wordSet = new Set();

      // Expand base words with linguistic inflections (~1000+ daily permutations)
      for (const base of langObj.dictionary) {
        wordSet.add(base.toUpperCase());
        if (langObj.inflections) {
          for (const inf of langObj.inflections) {
            wordSet.add((base + inf).toUpperCase());
          }
        }
      }

      // Populate Trie
      for (const word of wordSet) {
        let curr = root;
        for (let i = 0; i < word.length; i++) {
          const char = word[i];
          if (!curr.children.has(char)) {
            curr.children.set(char, new TrieNode());
          }
          curr = curr.children.get(char);
        }
        curr.isWord = true;
        curr.fullWord = word;
      }

      this.tries.set(langKey, { root, totalCount: wordSet.size });
    }
  }

  getPrediction(langKey, query) {
    const trieData = this.tries.get(langKey);
    if (!trieData) return { allowedChars: new Set(), suggestions: [], totalWords: 0 };

    const cleanQuery = query.toUpperCase();
    let curr = trieData.root;

    if (cleanQuery.length === 0) {
      return {
        allowedChars: new Set(curr.children.keys()),
        suggestions: ['PRAHA', 'NAVIGACE', 'RADIO', 'CLIMATE', 'CYBERPEAR'].slice(0, 5),
        totalWords: trieData.totalCount
      };
    }

    for (let i = 0; i < cleanQuery.length; i++) {
      const char = cleanQuery[i];
      if (!curr.children.has(char)) {
        return { allowedChars: new Set(), suggestions: [], totalWords: trieData.totalCount };
      }
      curr = curr.children.get(char);
    }

    const allowedChars = new Set(curr.children.keys());
    const suggestions = [];

    // Collect nearest candidate words via Depth-First Search
    const collectWords = (node) => {
      if (suggestions.length >= 6) return;
      if (node.isWord) suggestions.push(node.fullWord);
      for (const child of node.children.values()) {
        collectWords(child);
        if (suggestions.length >= 6) return;
      }
    };
    collectWords(curr);

    return { allowedChars, suggestions, totalWords: trieData.totalCount };
  }

  playAudioFeedback(type = 'tap') {
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      if (type === 'tap') {
        osc.frequency.setValueAtTime(940, this.audioCtx.currentTime);
        gain.gain.setValueAtTime(0.04, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + 0.04);
        osc.start();
        osc.stop(this.audioCtx.currentTime + 0.04);
      } else if (type === 'action') {
        osc.frequency.setValueAtTime(1400, this.audioCtx.currentTime);
        gain.gain.setValueAtTime(0.06, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + 0.08);
        osc.start();
        osc.stop(this.audioCtx.currentTime + 0.08);
      }

      if (navigator.vibrate) {
        navigator.vibrate(type === 'tap' ? 10 : 30);
      }
    } catch {
      // Audio limitations ignored in silent mode
    }
  }
}
