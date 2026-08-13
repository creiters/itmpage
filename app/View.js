export class AppView {
  constructor(viewModel) {
    this.viewModel = viewModel;
    
    this.heroTitle = document.getElementById('hero-title');
    this.heroSubtitle = document.getElementById('hero-subtitle');
    this.heroDesc = document.getElementById('hero-desc');
    this.gridContainer = document.getElementById('ecosystem-grid');
    this.termBody = document.getElementById('term-body');
    this.termInput = document.getElementById('term-input');

    this.bindEvents();
    this.viewModel.subscribe((state) => this.render(state));
  }

  bindEvents() {
    this.termInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        this.viewModel.executeCommand(this.termInput.value);
        this.termInput.value = '';
      }
    });

    window.executeCmd = (cmd) => this.viewModel.executeCommand(cmd);
  }

  render(state) {
    if (state.data) {
      if (this.heroTitle) this.heroTitle.textContent = state.data.system.title;
      if (this.heroSubtitle) this.heroSubtitle.textContent = state.data.system.subtitle;
      if (this.heroDesc) this.heroDesc.innerHTML = state.data.system.description;

      if (this.gridContainer) {
        this.gridContainer.innerHTML = state.data.ecosystem.map(card => `
          <div class="card ${card.isTranscendent ? 'transcendence-card' : ''}">
            <h3>${card.title}</h3>
            <p>${card.description}</p>
          </div>
        `).join('');
      }
    }

    if (this.termBody) {
      this.termBody.innerHTML = state.terminalLogs.map(log => 
        `<div class="terminal-log">${log.text}</div>`
      ).join('');
      this.termBody.scrollTop = this.termBody.scrollHeight;
    }
  }
}

