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
    this.calculateNextKeys();
    this.notify();
  }

  toggleLanguage() {
    this.model.playCarFeedback('select');
    const langs = Object.keys(this.config.languages);
    const currIdx = langs.indexOf(this.state.language);
    this.state.language = langs[(currIdx + 1) % langs.length];
    this.state.inputBuffer = '';
    this.calculateNextKeys();
    this.notify();
  }

  pressKey(char) {
    this.model.playCarFeedback('tap');
    this.state.inputBuffer += char;
    this.calculateNextKeys();
    this.notify();
  }

  backspace() {
    this.model.playCarFeedback('tap');
    if (this.state.inputBuffer.length > 0) {
      this.state.inputBuffer = this.state.inputBuffer.slice(0, -1);
      this.calculateNextKeys();
      this.notify();
    }
  }

  clear() {
    this.model.playCarFeedback('tap');
    this.state.inputBuffer = '';
    this.calculateNextKeys();
    this.notify();
  }

  selectSuggestion(phrase) {
    this.model.playCarFeedback('select');
    this.state.inputBuffer = phrase;
    this.calculateNextKeys();
    this.notify();
  }

  calculateNextKeys() {
    const dict = this.config.languages[this.state.language].dictionary;
    const query = this.state.inputBuffer.toUpperCase();

    if (query.length === 0) {
      // At start: All initials across dictionary words + space disabled
      const initials = new Set();
      dict.forEach((entry) => {
        const first = entry.trim()[0];
        if (first) initials.add(first);
      });
      this.state.allowedKeys = initials;
      this.state.suggestions = dict.slice(0, 4); // Default hot targets
      return;
    }

    const matches = dict.filter((entry) => entry.toUpperCase().startsWith(query));
    const nextChars = new Set();

    matches.forEach((entry) => {
      const upper = entry.toUpperCase();
      if (upper.length > query.length) {
        nextChars.add(upper[query.length]);
      }
    });

    this.state.allowedKeys = nextChars;
    this.state.suggestions = matches.slice(0, 5);
  }
}
