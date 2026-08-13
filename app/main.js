/**
 * MVVM ViewModel & Initialization Layer
 */

class AppViewModel {
  constructor() {
    this.data = null;
  }

  async init() {
    try {
      const response = await fetch('data/content.json');
      this.data = await response.json();
      this.render();
      this.bindTerminalEvents();
    } catch (err) {
      console.error('Failed to load JSON content:', err);
    }
  }

  render() {
    this.renderNav();
    this.renderHero();
    this.renderManifesto();
    this.renderEcosystem();
    this.renderBalance();
    this.renderTerminalHeader();
    this.renderContact();
    this.renderFooter();
  }

  renderNav() {
    const brandEl = document.getElementById('nav-brand-title');
    if (brandEl) brandEl.textContent = this.data.brand;

    const navContainer = document.getElementById('nav-links-list');
    if (navContainer) {
      navContainer.innerHTML = this.data.navigation.map(item => `
        <li>
          <a href="${item.href}" class="${item.isTranscend ? 'transcend-link' : ''}">
            ${item.label}
          </a>
        </li>
      `).join('');
    }
  }

  renderHero() {
    const { title, subtitle, description, ctaText } = this.data.hero;
    document.getElementById('hero-title').textContent = title;
    document.getElementById('hero-subtitle').textContent = subtitle;
    document.getElementById('hero-desc').innerHTML = description;
    
    const ctaBtn = document.getElementById('hero-cta');
    if (ctaBtn) {
      ctaBtn.textContent = ctaText;
      ctaBtn.onclick = () => window.executeCmd('awaken');
    }
  }

  renderManifesto() {
    const { title, paragraphs } = this.data.manifesto;
    document.getElementById('manifesto-title').textContent = title;
    const bodyContainer = document.getElementById('manifesto-body');
    if (bodyContainer) {
      bodyContainer.innerHTML = paragraphs.map(p => `<p class="body-text">${p}</p>`).join('');
    }
  }

  renderEcosystem() {
    const { title, cards } = this.data.ecosystem;
    document.getElementById('ecosystem-title').textContent = title;
    const gridContainer = document.getElementById('ecosystem-grid');
    if (gridContainer) {
      gridContainer.innerHTML = cards.map(card => `
        <div class="card ${card.isTranscendence ? 'transcendence-card' : ''}">
          <h3>${card.title}</h3>
          <p>${card.description}</p>
        </div>
      `).join('');
    }
  }

  renderBalance() {
    const { title, paragraphs } = this.data.balance;
    document.getElementById('balance-title').textContent = title;
    const bodyContainer = document.getElementById('balance-body');
    if (bodyContainer) {
      bodyContainer.innerHTML = paragraphs.map(p => `<p class="body-text">${p}</p>`).join('');
    }
  }

  renderTerminalHeader() {
    const { title, version, status, prompt } = this.data.terminal;
    document.getElementById('terminal-title').textContent = title;
    document.getElementById('term-version').textContent = version;
    document.getElementById('term-status').textContent = status;
    document.getElementById('term-prompt').textContent = prompt;
  }

  renderContact() {
    const { title, cardTitle, node, operator, signal, buttonText } = this.data.contact;
    document.getElementById('contact-title').textContent = title;
    document.getElementById('contact-card-title').textContent = cardTitle;
    document.getElementById('contact-node').textContent = node;
    document.getElementById('contact-operator').textContent = operator;
    document.getElementById('contact-signal').textContent = signal;
    
    const btn = document.getElementById('contact-btn');
    if (btn) {
      btn.textContent = buttonText;
      btn.onclick = () => window.executeCmd('ping');
    }
  }

  renderFooter() {
    const { copyright, status, transcendence, threat } = this.data.footer;
    document.getElementById('footer-copy').textContent = copyright;
    document.getElementById('footer-status').textContent = status;
    document.getElementById('footer-transcendence').textContent = transcendence;
    document.getElementById('footer-threat').textContent = threat;
  }

  bindTerminalEvents() {
    const termInput = document.getElementById('term-input');
    if (!termInput) return;

    termInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const cmd = termInput.value.trim().toLowerCase();
        if (cmd) {
          const promptText = this.data.terminal.prompt;
          window.appendLog(`<span style="color:var(--pink-neon);">${promptText}</span> ${cmd}`);
          window.executeCmd(cmd);
        }
        termInput.value = '';
      }
    });
  }
}

/* Terminal Global Handlers */
window.appendLog = function(text) {
  const termBody = document.getElementById('term-body');
  if (!termBody) return;
  const div = document.createElement('div');
  div.className = 'terminal-log';
  div.innerHTML = text;
  termBody.appendChild(div);
  termBody.scrollTop = termBody.scrollHeight;
};

window.executeCmd = function(cmd) {
  switch (cmd) {
    case 'help':
      window.appendLog(`Available commands: <span style="color:var(--cyan-neon);">status</span>, <span style="color:var(--cyan-neon);">ping</span>, <span style="color:var(--gold-transcend);">awaken</span>, <span style="color:var(--cyan-neon);">clear</span>`);
      break;
    case 'status':
      window.appendLog(`> SSHANET CORE [2026]: ONLINE<br>> TRANSCENDENCE QUOTIENT: <span style="color:var(--gold-transcend);">99.8%</span><br>> ACTIVE NEURONS: 100,000,000,000`);
      break;
    case 'ping':
      window.appendLog(`PING sshanet.cz (127.0.0.1): 56 bytes.<br><span style="color:#00ffcc;">Reply from SSHAnet node #2026: time=0.42ms [SIGNAL STABLE]</span>`);
      break;
    case 'awaken':
      window.appendLog(`<span style="color:var(--gold-transcend);">[!] SYNAPSE SURGE INITIATED IN SSHANET...</span><br>[+] Awakening 2026 neural pathways in SSHAnet backbone...<br>[+] 100B Neurons synchronized.`);
      break;
    case 'clear':
      const termBody = document.getElementById('term-body');
      if (termBody) termBody.innerHTML = '';
      break;
    default:
      window.appendLog(`<span style="color:var(--pink-neon);">ERROR:</span> Unknown command '${cmd}'. Type 'help' for options.`);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  const app = new AppViewModel();
  app.init();
});
