export class KeyboardModel {
  constructor() {
    this.audioCtx = null;
  }

  async fetchConfig() {
    const res = await fetch('./data/keyboard.json');
    if (!res.ok) throw new Error('Cannot load dictionary configuration.');
    return await res.json();
  }

  playCarFeedback(type = 'tap') {
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      if (type === 'tap') {
        osc.frequency.setValueAtTime(880, this.audioCtx.currentTime);
        gain.gain.setValueAtTime(0.04, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + 0.05);
        osc.start();
        osc.stop(this.audioCtx.currentTime + 0.05);
      } else if (type === 'select') {
        osc.frequency.setValueAtTime(1320, this.audioCtx.currentTime);
        gain.gain.setValueAtTime(0.06, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + 0.08);
        osc.start();
        osc.stop(this.audioCtx.currentTime + 0.08);
      }

      if (navigator.vibrate) {
        navigator.vibrate(type === 'tap' ? 12 : 25);
      }
    } catch {
      // Ignore background audio limitations
    }
  }
}
