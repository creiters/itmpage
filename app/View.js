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

    // Render each template section with dynamic state binding
    const navDOM = JSONTemplateEngine.render(this.templates.nav, state, this.viewModel);
    const heroDOM = JSONTemplateEngine.render(this.templates.hero, state, this.viewModel);
    const aboutDOM = JSONTemplateEngine.render(this.templates.about, state, this.viewModel);
    const ecosystemDOM = JSONTemplateEngine.render(this.templates.ecosystem, state, this.viewModel);
    const balanceDOM = JSONTemplateEngine.render(this.templates.balance, state, this.viewModel);
    const terminalDOM = JSONTemplateEngine.render(this.templates.terminal, state, this.viewModel);
    const contactDOM = JSONTemplateEngine.render(this.templates.contact, state, this.viewModel);
    const footerDOM = JSONTemplateEngine.render(this.templates.footer, state.footer, this.viewModel);

    this.rootContainer.appendChild(navDOM);
    this.rootContainer.appendChild(heroDOM);
    this.rootContainer.appendChild(aboutDOM);
    this.rootContainer.appendChild(ecosystemDOM);
    this.rootContainer.appendChild(balanceDOM);
    this.rootContainer.appendChild(terminalDOM);
    this.rootContainer.appendChild(contactDOM);
    this.rootContainer.appendChild(footerDOM);

    // Scroll command line console to bottom on input
    const termBody = document.getElementById('term-body');
    if (termBody) termBody.scrollTop = termBody.scrollHeight;
  }
}
