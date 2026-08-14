export class AppViewModel {
  constructor(model) {
    this.model = model;
    this.listeners = [];
    this.state = {
      navTitle: '',
      navLinks: [],
      system: {},
      manifestoTitle: '',
      manifestoParagraphs: [],
      ecosystem: [],
      balanceTitle: '',
      balanceParagraphs: [],
      contactTitle: '',
      nodeText: '',
      operatorText: '',
      signalText: '',
      buttonText: '',
      footer: {},
      terminalLogs: [
        { text: "<span style='color: var(--cyan-neon);'>SYSTEM:</span> Cyberpear SSHAnet Neural Interface initialized [2026]." },
        { text: "<span style='color: var(--cyan-neon);'>SYSTEM:</span> Type <span style='color: var(--yellow-glow);'>'help'</span> for available commands." }
      ],
      currentCommand: ''
    };
  }

  subscribe(listener) {
    this.listeners.push(listener);
  }

  notify() {
    this.listeners.forEach((listener) => listener(this.state));
  }

  async init() {
    const data = await this.model.loadInitialData();
    Object.assign(this.state, data);
    this.notify();
  }

  pingSignal() {
    this.executeCommand('ping');
  }

  awaken() {
    this.executeCommand('awaken');
  }

  handleCommandInput(event) {
    this.state.currentCommand = event.target.value;
    if (event.key === 'Enter') {
      const cmd = this.state.currentCommand.trim().toLowerCase();
      this.executeCommand(cmd);
      this.state.currentCommand = '';
      event.target.value = '';
    }
  }

  executeCommand(cmd) {
    if (!cmd) return;
    this.addLog(`<span style="color:var(--pink-neon);">creiters@sshanet-2026:~$</span> ${cmd}`);

    switch (cmd) {
      case 'help':
        this.addLog("Available commands: <span style='color:var(--cyan-neon);'>status</span>, <span style='color:var(--cyan-neon);'>ping</span>, <span style='color:var(--gold-transcend);'>awaken</span>, <span style='color:var(--cyan-neon);'>clear</span>");
        break;
      case 'status':
        this.addLog("> SSHANET CORE [2026]: ONLINE<br>> TRANSCENDENCE QUOTIENT: <span style='color:var(--gold-transcend);'>99.8%</span><br>> ACTIVE NEURONS: 100,000,000,000");
        break;
      case 'ping':
        this.addLog("PING sshanet.cz (127.0.0.1): 56 bytes.<br><span style='color:#00ffcc;'>Reply from SSHAnet node #2026: time=0.42ms [SIGNAL STABLE]</span>");
        break;
      case 'awaken':
        this.addLog("<span style='color:var(--gold-transcend);'>[!] SYNAPSE SURGE INITIATED IN SSHANET...</span><br>[+] Awakening 2026 neural pathways in SSHAnet backbone...<br>[+] 100B Neurons synchronized.");
        break;
      case 'clear':
        this.state.terminalLogs = [];
        this.notify();
        break;
      default:
        this.addLog(`<span style="color:var(--pink-neon);">ERROR:</span> Unknown command '${cmd}'. Type 'help' for options.`);
    }
  }

  addLog(text) {
    this.state.terminalLogs.push({ text });
    this.notify();
  }
    }
