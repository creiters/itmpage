export class JSONTemplateEngine {
  /**
   * Helper to evaluate dot-notation paths against data scope
   */
  static getValueByPath(obj, path) {
    if (!path || !obj) return '';
    return path.split('.').reduce((acc, key) => (acc && acc[key] !== undefined ? acc[key] : ''), obj);
  }

  /**
   * Parses JSON Schema / Template into real DOM nodes & binds ViewModel state/actions
   */
  static render(jsonTemplate, scopeData, viewModel) {
    if (typeof jsonTemplate === 'string') {
      return document.createTextNode(jsonTemplate);
    }

    const element = document.createElement(jsonTemplate.tag || 'div');

    // 1. Set static attributes
    if (jsonTemplate.attributes) {
      Object.entries(jsonTemplate.attributes).forEach(([attr, val]) => {
        element.setAttribute(attr, val);
      });
    }

    // 2. Process MVVM Data Bindings
    if (jsonTemplate.bindings) {
      Object.entries(jsonTemplate.bindings).forEach(([bindingType, targetKey]) => {
        const val = this.getValueByPath(scopeData, targetKey);

        switch (bindingType) {
          case 'text':
            element.textContent = val;
            break;
          case 'html':
            element.innerHTML = val;
            break;
          case 'value':
            element.value = val;
            break;
          case 'class':
            if (val) element.className += ` ${val}`;
            break;
          case 'click':
            element.addEventListener('click', () => viewModel[targetKey] && viewModel[targetKey]());
            break;
          case 'keyup':
            element.addEventListener('keyup', (e) => viewModel[targetKey] && viewModel[targetKey](e));
            break;
        }
      });
    }

    // 3. Process Loops/Repeats
    if (jsonTemplate.repeat) {
      const { dataKey, itemTemplate } = jsonTemplate.repeat;
      const listData = this.getValueByPath(scopeData, dataKey) || [];

      listData.forEach((itemScope) => {
        const childNode = this.render(itemTemplate, itemScope, viewModel);
        element.appendChild(childNode);
      });
    }

    // 4. Process Static/Nested Children
    if (jsonTemplate.children && Array.isArray(jsonTemplate.children)) {
      jsonTemplate.children.forEach((childJson) => {
        const childNode = this.render(childJson, scopeData, viewModel);
        element.appendChild(childNode);
      });
    }

    return element;
  }
                                                 }
    
