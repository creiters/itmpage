export class AppViewModel {
  constructor(model) {
    this.model = model;
    this.listeners = [];
    this.state = {
      system: {},
      ecosystem: [],
      terminalLogs: [
        { text: "<span style='color: var(--cyan-neon);'>SYSTEM:</span> Cyberpear Neural Interface Online [2026]." },
        { text: "<span style='color: var(--cyan-neon);'>SYSTEM:</span> Type <span style='color: var(--yellow-glow);'>'help'</span> for options." }
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
    this.state.system = data.system;
    this.state.ecosystem = data.ecosystem;
    this.notify();
  }

  awaken() {
    this.addLog("<span style='color:var(--gold-transcend);'>[!] SYNAPSE SURGE INITIATED IN SSHANET...</span>");
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
        this.addLog("Commands: <span style='color:var(--cyan-neon);'>status</span>, <span style='color:var(--gold-transcend);'>awaken</span>, <span style='color:var(--cyan-neon);'>clear</span>");
        break;
      case 'status':
        this.addLog("> SSHANET CORE: ONLINE | TRANSCENDENCE: <span style='color:var(--gold-transcend);'>99.8%</span>");
        break;
      case 'awaken':
        this.awaken();
        break;
      case 'clear':
        this.state.terminalLogs = [];
        this.notify();
        break;
      default:
        this.addLog(`<span style="color:var(--pink-neon);">ERROR:</span> Unknown command '${cmd}'`);
    }
  }

  addLog(text) {
    this.state.terminalLogs.push({ text });
    this.notify();
  }
    }
