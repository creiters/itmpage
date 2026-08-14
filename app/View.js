import { JSONTemplateEngine } from './Framework.js';

export class AppView {
  constructor(viewModel, templates, rootContainer) {
    this.viewModel = viewModel;
    this.templates = templates;
    this.rootContainer = rootContainer;

    this.viewModel.subscribe((state) => this.render(state));
  }

  render(state) {
    this.rootContainer.innerHTML = '';

    // Transform JSON Templates + JSON State Data directly into DOM Node Trees
    const heroDOM = JSONTemplateEngine.render(this.templates.hero, state, this.viewModel);
    const ecosystemDOM = JSONTemplateEngine.render(this.templates.ecosystem, state, this.viewModel);
    const terminalDOM = JSONTemplateEngine.render(this.templates.terminal, state, this.viewModel);

    this.rootContainer.appendChild(heroDOM);
    this.rootContainer.appendChild(ecosystemDOM);
    this.rootContainer.appendChild(terminalDOM);

    // Auto-scroll terminal log to bottom
    const termBody = document.getElementById('term-body');
    if (termBody) termBody.scrollTop = termBody.scrollHeight;
  }
                                             }
