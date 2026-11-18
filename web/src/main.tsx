// FOLO Studio: the app, mounted into #app.
import './firstScreen';
import { render } from 'fw';
import './lib/design/tokens.css';
import './lib/design/base.css';
import App from './App';
import { watchFailedStyles } from './lib/failedStyles';
import { installHeartbeat } from './lib/ui/heartbeat';

// A stylesheet of a lazy chunk that failed to load is loaded again before the next chunk (Retry, the next screen), never
// skipped. Before any network answer can arrive: this runs in the task that evaluated the app's modules.
watchFailedStyles();

const target = document.getElementById('app');
if (!target) throw new Error('#app missing');

// The live heartbeats hold still for most of each beat: paused then, so an idle page with a running run idles too.
installHeartbeat();

export default render(() => <App />, target);
