export class KeyboardView {
  constructor(viewModel, container) {
    this.vm = viewModel;
    this.container = container;
    this.vm.subscribe((state) => this.render(state));
  }

  render(state) {
    this.container.innerHTML = `
      <div class="hmi-keyboard-container">
        <!-- Top Status Bar & CyberPear Branding -->
        <div class="hmi-brand-panel">
          <div class="brand-wrapper">
            <img src="assets/images/creiters_cyberpear.svg" alt="CyberPear Logo" class="hmi-logo-asset" />
            <div>
              <span class="brand-label">CYBERPEAR // HMI SPELLER</span>
              <div class="corpus-indicator">ACTIVE REPOSITORY: ${state.indexedWordCount} WORDS [~1K PERMUTATIONS]</div>
            </div>
          </div>
          <button class="hmi-lang-toggle" id="lang-btn">LANG: [ ${state.language.toUpperCase()} ]</button>
        </div>

        <!-- Cockpit Search Terminal -->
        <div class="hmi-search-display">
          <div class="hmi-input-terminal">
            ${state.inputBuffer ? `${state.inputBuffer}<span class="cursor"></span>` : '<span class="placeholder">ENTER NAVIGATION OR COMMAND TARGET...</span>'}
          </div>
          <button class="hmi-action-key-square" id="clear-btn">CLR</button>
          <button class="hmi-action-key-square" id="del-btn">DEL</button>
        </div>

        <!-- Predictive Suggestions Carousel -->
        <div class="hmi-prediction-strip">
          ${state.suggestions.length > 0 
            ? state.suggestions.map((entry) => `<div class="prediction-chip" data-phrase="${entry}">⚡ ${entry}</div>`).join('')
            : '<span class="no-prediction">NO MATCHING LEXICON ENTRIES</span>'
          }
        </div>

        <!-- Automotive Matrix Keyboard -->
        <div class="hmi-speller-grid">
          ${state.rows.map((row) => `
            <div class="hmi-row">
              ${row.map((char) => {
                const isEnabled = state.allowedKeys.has(char);
                const isSpace = char === ' ';
                const label = isSpace ? 'SPACE [ _ ]' : char;
                return `
                  <button 
                    class="hmi-key ${isSpace ? 'hmi-space-key' : ''}" 
                    data-char="${char}" 
                    ${isEnabled ? '' : 'disabled'}>
                    ${label}
                  </button>
                `;
              }).join('')}
            </div>
          `).join('')}
        </div>
      </div>
    `;

    this.bindEvents();
  }

  bindEvents() {
    this.container.querySelector('#lang-btn')?.addEventListener('click', () => this.vm.toggleLanguage());
    this.container.querySelector('#del-btn')?.addEventListener('click', () => this.vm.backspace());
    this.container.querySelector('#clear-btn')?.addEventListener('click', () => this.vm.clear());

    this.container.querySelectorAll('.hmi-key[data-char]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const char = btn.getAttribute('data-char');
        if (char) this.vm.pressKey(char);
      });
    });

    this.container.querySelectorAll('.prediction-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const phrase = chip.getAttribute('data-phrase');
        if (phrase) this.vm.selectSuggestion(phrase);
      });
    });
  }
}
