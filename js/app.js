import { registerWebMCP } from './webmcp.js';

class CreitersCyberPearApp {
  constructor() {
    this.meta = null;
    this.content = null;
    this.projects = [];
    this.currentSlideIndex = 0;

    // Element bindings
    this.navBrand = document.getElementById('nav-brand');
    this.navLinksList = document.getElementById('nav-links-list');
    this.heroTitle = document.getElementById('hero-title');
    this.heroSubtitle = document.getElementById('hero-subtitle');
    this.heroDesc = document.getElementById('hero-desc');
    this.heroCta = document.getElementById('hero-cta');

    this.manifestoTitle = document.getElementById('manifesto-title');
    this.manifestoBody = document.getElementById('manifesto-body');

    this.slideBadge = document.getElementById('slide-badge');
    this.slideTitle = document.getElementById('slide-title');
    this.slideDesc = document.getElementById('slide-desc');
    this.slideMeta = document.getElementById('slide-meta');
    this.slideRepoLink = document.getElementById('slide-repo-link');
    this.btnPrev = document.getElementById('btn-prev');
    this.btnNext = document.getElementById('btn-next');

    this.projectsGrid = document.getElementById('projects-grid');
    this.ecosystemTitle = document.getElementById('ecosystem-title');
    this.ecosystemGrid = document.getElementById('ecosystem-grid');

    this.balanceTitle = document.getElementById('balance-title');
    this.balanceBody = document.getElementById('balance-body');

    this.terminalBody = document.getElementById('term-body');
    this.terminalInput = document.getElementById('term-input');

    this.contactTitle = document.getElementById('contact-title');
    this.contactCardHeader = document.getElementById('contact-card-header');
    this.contactNode = document.getElementById('contact-node');
    this.contactOperator = document.getElementById('contact-operator');
    this.contactSignal = document.getElementById('contact-signal');
    this.contactBtn = document.getElementById('contact-btn');

    this.footerCopy = document.getElementById('footer-copy');
    this.footerStatus = document.getElementById('footer-status');
    this.footerTranscendence = document.getElementById('footer-transcendence');
    this.footerThreat = document.getElementById('footer-threat');
  }

  async init() {
    await this.loadData();
    this.applyMetadata();
    this.renderContent();
    this.renderDeck();
    this.renderProjectsGrid();
    this.bindEvents();
    this.initPWA();
    registerWebMCP(this);

    this.appendLog("SYSTEM: CyberPear SSHAnet Neural Interface initialized [2026].");
    this.appendLog("SYSTEM: Type 'help', 'projects', or 'awaken' for available commands.");
  }

  async loadData() {
    try {
      const [metaRes, contentRes, projRes] = await Promise.all([
        fetch('./data/meta.json'),
        fetch('./data/content.json'),
        fetch('./data/projects.json')
      ]);
      this.meta = await metaRes.json();
      this.content = await contentRes.json();
      this.projects = await projRes.json();
    } catch (err) {
      console.error("Data load failure:", err);
    }
  }

  applyMetadata() {
    if (!this.meta) return;
    document.title = this.meta.title;

    const setMeta = (name, content) => {
      let el = document.querySelector(`meta[name="${name}"]`);
      if (!el) {
        el = document.createElement('meta');
        el.setAttribute('name', name);
        document.head.appendChild(el);
      }
      el.setAttribute('content', content);
    };

    setMeta('description', this.meta.description);
    setMeta('keywords', this.meta.keywords);
    setMeta('geo.region', this.meta.geo.region);
    setMeta('geo.placename', this.meta.geo.placename);
    setMeta('geo.position', this.meta.geo.position);
    setMeta('ICBM', this.meta.geo.icbm);

    const schemaTag = document.createElement('script');
    schemaTag.type = 'application/ld+json';
    schemaTag.textContent = JSON.stringify(this.meta.schema);
    document.head.appendChild(schemaTag);
  }

  renderContent() {
    if (!this.content) return;

    this.navLinksList.innerHTML = '';
    this.content.navLinks.forEach((item) => {
      const li = document.createElement('li');
      li.setAttribute('role', 'none');
      const a = document.createElement('a');
      a.setAttribute('role', 'menuitem');
      a.href = item.href;
      a.textContent = item.label;
      if (item.isTranscend) a.className = 'transcend-link';
      li.appendChild(a);
      this.navLinksList.appendChild(li);
    });

    this.heroTitle.textContent = this.content.hero.title;
    this.heroSubtitle.textContent = this.content.hero.subtitle;
    this.heroDesc.innerHTML = this.content.hero.description;
    this.heroCta.textContent = this.content.hero.ctaText;

    this.manifestoTitle.textContent = this.content.manifesto.title;
    this.manifestoBody.innerHTML = '';
    this.content.manifesto.paragraphs.forEach((pText) => {
      const p = document.createElement('p');
      p.className = 'body-text';
      p.innerHTML = pText;
      this.manifestoBody.appendChild(p);
    });

    this.ecosystemTitle.textContent = this.content.ecosystem.title;
    this.ecosystemGrid.innerHTML = '';
    this.content.ecosystem.cards.forEach((item) => {
      const card = document.createElement('article');
      card.className = item.isTranscend ? 'card transcendence-card' : 'card';
      card.setAttribute('tabindex', '0');

      const wrap = document.createElement('div');
      const tag = document.createElement('div');
      tag.className = 'card-tag';
      tag.textContent = item.tag;

      const h3 = document.createElement('h3');
      h3.textContent = item.title;

      const p = document.createElement('p');
      p.textContent = item.description;

      wrap.appendChild(tag);
      wrap.appendChild(h3);
      wrap.appendChild(p);
      card.appendChild(wrap);
      this.ecosystemGrid.appendChild(card);
    });

    this.balanceTitle.textContent = this.content.balance.title;
    this.balanceBody.innerHTML = '';
    this.content.balance.paragraphs.forEach((pText) => {
      const p = document.createElement('p');
      p.className = 'body-text';
      p.innerHTML = pText;
      this.balanceBody.appendChild(p);
    });

    this.contactTitle.textContent = this.content.contact.title;
    this.contactCardHeader.textContent = this.content.contact.cardHeader;
    this.contactNode.textContent = this.content.contact.node;
    this.contactOperator.textContent = this.content.contact.operator;
    this.contactSignal.textContent = this.content.contact.signal;
    this.contactBtn.textContent = this.content.contact.buttonText;

    this.footerCopy.textContent = this.content.footer.copyright;
    this.footerStatus.textContent = this.content.footer.status;
    this.footerTranscendence.textContent = this.content.footer.transcendence;
    this.footerThreat.textContent = this.content.footer.threat;
  }

  goToSlide(target) {
    if (typeof target === 'number') {
      this.currentSlideIndex = Math.max(0, Math.min(target, this.projects.length - 1));
    } else if (typeof target === 'string') {
      const idx = this.projects.findIndex(
        (p) => p.id === target || p.title.toLowerCase().includes(target.toLowerCase())
      );
      if (idx !== -1) {
        this.currentSlideIndex = idx;
      } else {
        const parsed = parseInt(target, 10);
        if (!isNaN(parsed)) {
          this.currentSlideIndex = Math.max(0, Math.min(parsed, this.projects.length - 1));
        }
      }
    }
    this.renderDeck();
    return { activeIndex: this.currentSlideIndex, project: this.projects[this.currentSlideIndex] };
  }

  nextSlide() {
    if (this.projects.length === 0) return;
    this.currentSlideIndex = (this.currentSlideIndex + 1) % this.projects.length;
    this.renderDeck();
  }

  prevSlide() {
    if (this.projects.length === 0) return;
    this.currentSlideIndex = (this.currentSlideIndex - 1 + this.projects.length) % this.projects.length;
    this.renderDeck();
  }

  renderDeck() {
    const item = this.projects[this.currentSlideIndex];
    if (!item) return;

    this.slideBadge.textContent = `SLIDE [ ${this.currentSlideIndex + 1} / ${this.projects.length} ] // ${item.tag}`;
    this.slideTitle.textContent = item.title;
    this.slideDesc.textContent = item.description;
    this.slideMeta.textContent = `ARCHITECTURE SPEC: ${item.metrics}`;
    this.slideRepoLink.href = item.repo;
  }

  renderProjectsGrid() {
    this.projectsGrid.innerHTML = '';
    this.projects.forEach((item, index) => {
      const card = document.createElement('article');
      card.className = 'card';
      card.setAttribute('tabindex', '0');

      const contentWrap = document.createElement('div');
      const tag = document.createElement('div');
      tag.className = 'card-tag';
      tag.textContent = `[ ${item.badge} ] ${item.tag}`;

      const title = document.createElement('h3');
      title.textContent = item.title;

      const desc = document.createElement('p');
      desc.textContent = item.description;

      contentWrap.appendChild(tag);
      contentWrap.appendChild(title);
      contentWrap.appendChild(desc);

      const footer = document.createElement('div');
      footer.className = 'card-footer';

      const slideBtn = document.createElement('button');
      slideBtn.className = 'btn';
      slideBtn.textContent = 'View Slide';
      slideBtn.setAttribute('aria-label', `Navigate presentation to ${item.title}`);
      slideBtn.addEventListener('click', () => {
        this.goToSlide(index);
        document.getElementById('presentation').scrollIntoView({ behavior: 'smooth' });
      });

      const repoLink = document.createElement('a');
      repoLink.className = 'card-link';
      repoLink.href = item.repo;
      repoLink.target = '_blank';
      repoLink.rel = 'noopener noreferrer';
      repoLink.textContent = '[ Repository → ]';

      footer.appendChild(slideBtn);
      footer.appendChild(repoLink);

      card.appendChild(contentWrap);
      card.appendChild(footer);
      this.projectsGrid.appendChild(card);
    });
  }

  bindEvents() {
    this.btnPrev.addEventListener('click', () => this.prevSlide());
    this.btnNext.addEventListener('click', () => this.nextSlide());
    this.heroCta.addEventListener('click', () => this.executeCommand('awaken'));
    this.contactBtn.addEventListener('click', () => this.executeCommand('ping'));

    this.terminalInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const val = this.terminalInput.value.trim();
        if (val) {
          this.executeCommand(val);
          this.terminalInput.value = '';
        }
      }
    });
  }

  executeCommand(rawCmd) {
    const cmd = rawCmd.trim();
    if (!cmd) return;

    this.appendLog(`creiters@sshanet-2026:~$ ${cmd}`);
    const lower = cmd.toLowerCase();

    if (lower === 'help') {
      this.appendLog("Available commands: status, ping, awaken, projects, slide <n>, clear");
    } else if (lower === 'status') {
      this.appendLog("> SSHANET CORE [2026]: ONLINE | TRANSCENDENCE QUOTIENT: 99.8% | ACTIVE NEURONS: 100,000,000,000");
    } else if (lower === 'ping') {
      this.appendLog("PING sshanet.cz: bytes=56 time=0.42ms [SIGNAL STABLE]");
    } else if (lower === 'awaken') {
      this.appendLog("[!] SYNAPSE SURGE INITIATED IN SSHANET... 100B Neurons synchronized.");
    } else if (lower === 'projects') {
      this.projects.forEach((p, idx) => {
        this.appendLog(`[${idx}] ${p.title} (${p.tag}) -> ${p.repo}`);
      });
    } else if (lower.startsWith('slide ')) {
      const arg = lower.replace('slide ', '').trim();
      this.goToSlide(arg);
      this.appendLog(`Navigated presentation viewport to slide: ${arg}`);
    } else if (lower === 'clear') {
      this.terminalBody.innerHTML = '';
    } else {
      this.appendLog(`ERROR: Unknown command '${cmd}'. Type 'help' for options.`);
    }
  }

  appendLog(msg) {
    const line = document.createElement('div');
    line.className = 'terminal-log-line';
    line.textContent = msg;
    this.terminalBody.appendChild(line);
    this.terminalBody.scrollTop = this.terminalBody.scrollHeight;
  }

  initPWA() {
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js')
          .then((reg) => console.log('ServiceWorker active scope:', reg.scope))
          .catch((err) => console.warn('ServiceWorker registration error:', err));
      });
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const app = new CreitersCyberPearApp();
  window.__APP__ = app;
  app.init();
});
