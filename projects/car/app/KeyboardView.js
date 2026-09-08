export class KeyboardView {
  constructor(viewModel, container) {
    this.vm = viewModel;
    this.container = container;
    this.vm.subscribe((state) => this.render(state));
  }

  render(state) {
    this.container.innerHTML = `
      <div class="hmi-keyboard-container">
        <!-- Top Display -->
        <div class="hmi-header">
          <div class="hmi-input-display">${state.inputBuffer || '<span style="opacity:0.3;">ENTER TARGET...</span>'}</div>
          <button class="hmi-lang-btn" id="lang-btn">[ ${state.language.toUpperCase()} ]</button>
        </div>

        <!-- Dynamic Suggestions Bar -->
        <div class="hmi-suggestions">
          ${state.suggestions.map((s) => `<div class="hmi-suggest-chip" data-word="${s}">${s}</div>`).join('')}
        </div>

        <!-- Keypad Grid -->
        <div class="hmi-grid">
          ${state.rows.map((row, idx) => `
            <div class="hmi-row">
              ${idx === 2 ? `<button class="hmi-key hmi-action-key" id="clear-btn">CLR</button>` : ''}
              ${row.map((char) => {
                const isEnabled = state.allowedKeys.has(char);
                return `<button class="hmi-key" data-char="${char}" ${isEnabled ? '' : 'disabled'}>${char}</button>`;
              }).join('')}
              ${idx === 2 ? `<button class="hmi-key hmi-action-key" id="bksp-btn">DEL</button>` : ''}
            </div>
          `).join('')}
        </div>
      </div>
    `;

    this.bindEvents();
  }

  bindEvents() {
    this.container.querySelector('#lang-btn')?.addEventListener('click', () => this.vm.toggleLanguage());
    this.container.querySelector('#bksp-btn')?.addEventListener('click', () => this.vm.backspace());
    this.container.querySelector('#clear-btn')?.addEventListener('click', () => this.vm.clear());

    this.container.querySelectorAll('.hmi-key[data-char]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const char = btn.getAttribute('data-char');
        if (char) this.vm.pressKey(char);
      });
    });

    this.container.querySelectorAll('.hmi-suggest-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const word = chip.getAttribute('data-word');
        if (word) this.vm.selectSuggestion(word);
      });
    });
  }
}

