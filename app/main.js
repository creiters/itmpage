import { AppModel } from './Model.js';
import { AppViewModel } from './ViewModel.js';
import { AppView } from './View.js';

document.addEventListener('DOMContentLoaded', async () => {
  const rootContainer = document.getElementById('app-root');
  const model = new AppModel();
  
  // Load JSON templates and JSON data concurrently
  const templates = await model.loadTemplates();
  const viewModel = new AppViewModel(model);
  
  new AppView(viewModel, templates, rootContainer);
  
  // Initialize state
  await viewModel.init();
});
