// FOLO Studio: the app, mounted into #app.
import './firstScreen';
import { render } from 'fw';
import './lib/design/tokens.css';
import './lib/design/base.css';
import App from './App';
import { watchFailedStyles } from './lib/failedStyles';

// A stylesheet of a lazy chunk that failed to load is asked for again when the next chunk loads (Retry, the next
// screen), in its place. Before any network answer can arrive: this runs in the task that evaluated the app's modules.
watchFailedStyles();

const target = document.getElementById('app');
if (!target) throw new Error('#app missing');

export default render(() => <App />, target);
