import { AppModel } from './Model.js';
import { AppViewModel } from './ViewModel.js';
import { AppView } from './View.js';

document.addEventListener('DOMContentLoaded', () => {
  const model = new AppModel();
  const viewModel = new AppViewModel(model);
  new AppView(viewModel);
  viewModel.init();
});

