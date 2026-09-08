export class KeyboardViewModel {
  constructor(model) {
    this.model = model;
    this.listeners = [];
    this.config = null;
    this.state = {
      language: 'cs',
      inputBuffer: '',
      allowedKeys: new Set(),
      suggestions: [],
      rows: []
    };
  }

  subscribe(fn) {
    this.listeners.push(fn);
  }

  notify() {
    this.listeners.forEach((fn) => fn(this.state));
  }

  async init() {
    this.config = await this.model.fetchConfig();
    this.state.rows = this.config.rows;
    this.recalculateAllowedKeys();
    this.notify();
  }

  toggleLanguage() {
    const langs = Object.keys(this.config.languages);
    const currIdx = langs.indexOf(this.state.language);
    this.state.language = langs[(currIdx + 1) % langs.length];
    this.state.inputBuffer = '';
    this.recalculateAllowedKeys();
    this.notify();
  }

  pressKey(char) {
    this.state.inputBuffer += char;
    this.recalculateAllowedKeys();
    this.notify();
  }

  backspace() {
    if (this.state.inputBuffer.length > 0) {
      this.state.inputBuffer = this.state.inputBuffer.slice(0, -1);
      this.recalculateAllowedKeys();
      this.notify();
    }
  }

  clear() {
    this.state.inputBuffer = '';
    this.recalculateAllowedKeys();
    this.notify();
  }

  selectSuggestion(word) {
    this.state.inputBuffer = word;
    this.recalculateAllowedKeys();
    this.notify();
  }

  recalculateAllowedKeys() {
    const dict = this.config.languages[this.state.language].dictionary;
    const buffer = this.state.inputBuffer.toUpperCase();

    if (buffer.length === 0) {
      // First character: Allow all initials present in the dictionary
      const initials = new Set(dict.map((word) => word[0]));
      this.state.allowedKeys = initials;
      this.state.suggestions = [];
      return;
    }

    // Filter words matching current input prefix
    const matches = dict.filter((word) => word.startsWith(buffer));
    const nextChars = new Set();

    matches.forEach((word) => {
      if (word.length > buffer.length) {
        nextChars.add(word[buffer.length]);
      }
    });

    this.state.allowedKeys = nextChars;
    this.state.suggestions = matches.slice(0, 4);
  }
}

