import { AppModel } from './Model.js';
import { AppViewModel } from './ViewModel.js';
import { AppView } from './View.js';

document.addEventListener('DOMContentLoaded', async () => {
  const rootContainer = document.getElementById('app-root');
  const model = new AppModel();
  
  // Concurrently fetch all JSON templates and JSON content
  const templates = await model.loadTemplates();
  const viewModel = new AppViewModel(model);
  
  // Mount the View and subscribe to state changes
  new AppView(viewModel, templates, rootContainer);
  
  // Initialize state from content.json
  await viewModel.init();
});
