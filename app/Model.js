export class AppModel {
  async fetchJSON(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load JSON asset at: ${path}`);
    return await response.json();
  }

  async loadInitialData() {
    return await this.fetchJSON('./data/content.json');
  }

  async loadTemplates() {
    const [hero, ecosystem, terminal] = await Promise.all([
      this.fetchJSON('./templates/hero.json'),
      this.fetchJSON('./templates/ecosystem.json'),
      this.fetchJSON('./templates/terminal.json')
    ]);
    return { hero, ecosystem, terminal };
  }
}
