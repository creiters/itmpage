export class AppModel {
  async fetchNodeData() {
    const response = await fetch('./data/content.json');
    if (!response.ok) throw new Error('Network response failed loading node data.');
    return await response.json();
  }
}
