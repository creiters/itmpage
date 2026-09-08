export class KeyboardModel {
  async fetchConfig() {
    const res = await fetch('./data/keyboard.json');
    if (!res.ok) throw new Error('Failed to load keyboard configuration.');
    return await res.json();
  }
}
