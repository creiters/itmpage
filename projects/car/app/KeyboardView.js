export class KeyboardView {
  constructor(viewModel, container) {
    this.vm = viewModel;
    this.container = container;
    this.vm.subscribe((state) => this.render(state));
  }

  render(state) {
    this.container.innerHTML = `
      <div class="hmi-keyboard-container">
        <!-- Top Status & Logo Bar -->
        <div class="hmi-brand-panel">
          <div class="brand-wrapper">
            <!-- Animated CyberPear Cockpit Asset -->
            <svg class="hmi-logo" viewBox="0 0 500 550" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <filter id="cyan-glow-hmi" x="-30%" y="-30%" width="160%" height="160%">
                  <feGaussianBlur stdDeviation="6" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>
              <g transform="translate(10, 10)">
                <g fill="#8B5A2B"><rect x="230" y="70" width="12" height="60" /></g>
                <g fill="#158837"><rect x="182" y="94" width="48" height="24" /></g>
                <g fill="#156B2E"><rect x="254" y="130" width="60" height="200" /></g>
                <g fill="#22B14C"><rect x="182" y="130" width="80" height="200" /></g>
                <g fill="#1affff" filter="url(#cyan-glow-hmi)"><rect x="134" y="214" width="36" height="80" /></g>
                <g fill="#D4AF37"><rect x="170" y="250" width="12" height="12" /></g>
              </g>
            </svg>
            <span class="brand-label">CYBERPEAR // HMI SPELLER</span>
          </div>
          <span class="driver-mode-tag">DISTRACTION-LOCK: ACTIVE</span>
        </div>

        <!-- Current Query Target Screen -->
        <div class="hmi-search-display">
          <div class="hmi-input-terminal">
            ${state.inputBuffer ? `${state.inputBuffer}<span class="cursor"></span>` : '<span style="opacity:0.25;">SELECT DESTINATION...</span>'}
          </div>
          <button class="hmi-lang-toggle" id="lang-btn">[ ${state.language.toUpperCase()} ]</button>
        </div>

        <!-- Predictive Words Strip -->
        <div class="hmi-prediction-strip">
          ${state.suggestions.map((entry) => `
            <div class="prediction-chip" data-phrase="${entry}">⚡ ${entry}</div>
          `).join('')}
        </div>

        <!-- Virtual Matrix Speller -->
        <div class="hmi-speller-grid">
          ${state.rows.map((row, idx) => `
            <div class="hmi-row">
              ${idx === 2 ? `<button class="hmi-key hmi-action-key" id="clear-btn">CLEAR</button>` : ''}
              ${row.map((char) => {
                const isEnabled = state.allowedKeys.has(char);
                const isSpace = char === ' ';
                const label = isSpace ? 'SPACE' : char;
                return `
                  <button 
                    class="hmi-key ${isSpace ? 'hmi-space-key' : ''}" 
                    data-char="${char}" 
                    ${isEnabled ? '' : 'disabled'}>
                    ${label}
                  </button>
                `;
              }).join('')}
              ${idx === 2 ? `<button class="hmi-key hmi-action-key" id="del-btn">DELETE</button>` : ''}
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
