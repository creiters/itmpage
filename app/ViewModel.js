export class AppViewModel {
  constructor(model) {
    this.model = model;
    this.listeners = [];
    this.state = {
      data: null,
      terminalLogs: [
        { type: 'system', text: 'SYSTEM: Cyberpear SSHAnet Neural Interface initialized [2026].' },
        { type: 'system', text: "SYSTEM: Type 'help' for available commands." }
      ]
    };
  }

  subscribe(listener) {
    this.listeners.push(listener);
  }

  notify() {
    this.listeners.forEach(listener => listener(this.state));
  }

  async init() {
    try {
      const data = await this.model.fetchNodeData();
      this.state.data = data;
      this.notify();
    } catch (err) {
      this.addLog('error', `ERROR: Failed to load node JSON data -> ${err.message}`);
    }
  }

  executeCommand(cmd) {
    const cleanCmd = cmd.trim().toLowerCase();
    if (!cleanCmd) return;

    this.addLog('user', `creiters@sshanet-2026:~$ ${cleanCmd}`);

    switch (cleanCmd) {
      case 'help':
        this.addLog('output', "Available commands: <span style='color:var(--cyan-neon);'>status</span>, <span style='color:var(--cyan-neon);'>ping</span>, <span style='color:var(--gold-transcend);'>awaken</span>, <span style='color:var(--cyan-neon);'>clear</span>");
        break;
      case 'status':
        this.addLog('output', "> SSHANET CORE [2026]: ONLINE<br>> TRANSCENDENCE QUOTIENT: <span style='color:var(--gold-transcend);'>99.8%</span><br>> ACTIVE NEURONS: 100,000,000,000");
        break;
      case 'ping':
        this.addLog('output', "PING sshanet.cz (127.0.0.1): 56 bytes.<br><span style='color:#00ffcc;'>Reply from SSHAnet node #2026: time=0.42ms [SIGNAL STABLE]</span>");
        break;
      case 'awaken':
        this.addLog('output', "<span style='color:var(--gold-transcend);'>[!] SYNAPSE SURGE INITIATED IN SSHANET...</span><br>[+] Awakening 2026 neural pathways in SSHAnet backbone...<br>[+] 100B Neurons synchronized.");
        break;
      case 'clear':
        this.state.terminalLogs = [];
        this.notify();
        break;
      default:
        this.addLog('error', `ERROR: Unknown command '${cleanCmd}'. Type 'help' for options.`);
    }
  }

  addLog(type, text) {
    this.state.terminalLogs.push({ type, text });
    this.notify();
  }
}

