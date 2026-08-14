export class AppModel {
  async fetchJSON(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load asset at: ${path}`);
    return await response.json();
  }

  async loadInitialData() {
    return await this.fetchJSON('./data/content.json');
  }

  async loadTemplates() {
    const [nav, hero, about, ecosystem, balance, terminal, contact, footer] = await Promise.all([
      this.fetchJSON('./templates/nav.json'),
      this.fetchJSON('./templates/hero.json'),
      this.fetchJSON('./templates/about.json'),
      this.fetchJSON('./templates/ecosystem.json'),
      this.fetchJSON('./templates/balance.json'),
      this.fetchJSON('./templates/terminal.json'),
      this.fetchJSON('./templates/contact.json'),
      this.fetchJSON('./templates/footer.json')
    ]);

    return { nav, hero, about, ecosystem, balance, terminal, contact, footer };
  }
}
