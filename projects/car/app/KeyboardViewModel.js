export class KeyboardViewModel {
  constructor(model) {
    this.model = model;
    this.subscribers = [];
    this.config = null;
    this.state = {
      language: 'cs',
      inputBuffer: '',
      allowedKeys: new Set(),
      suggestions: [],
      rows: [],
      indexedWordCount: 0
    };
  }

  subscribe(fn) {
    this.subscribers.push(fn);
  }

  notify() {
    this.subscribers.forEach((fn) => fn(this.state));
  }

  async init() {
    this.config = await this.model.fetchConfig();
    this.state.rows = this.config.rows;
    this.updatePredictions();
    this.notify();
  }

  toggleLanguage() {
    this.model.playAudioFeedback('action');
    const langs = Object.keys(this.config.languages);
    const currIdx = langs.indexOf(this.state.language);
    this.state.language = langs[(currIdx + 1) % langs.length];
    this.state.inputBuffer = '';
    this.updatePredictions();
    this.notify();
  }

  pressKey(char) {
    this.model.playAudioFeedback('tap');
    this.state.inputBuffer += char;
    this.updatePredictions();
    this.notify();
  }

  backspace() {
    this.model.playAudioFeedback('tap');
    if (this.state.inputBuffer.length > 0) {
      this.state.inputBuffer = this.state.inputBuffer.slice(0, -1);
      this.updatePredictions();
      this.notify();
    }
  }

  clear() {
    this.model.playAudioFeedback('action');
    this.state.inputBuffer = '';
    this.updatePredictions();
    this.notify();
  }

  selectSuggestion(word) {
    this.model.playAudioFeedback('action');
    this.state.inputBuffer = word;
    this.updatePredictions();
    this.notify();
  }

  updatePredictions() {
    const res = this.model.getPrediction(this.state.language, this.state.inputBuffer);
    this.state.allowedKeys = res.allowedChars;
    this.state.suggestions = res.suggestions;
    this.state.indexedWordCount = res.totalWords;
  }
}
