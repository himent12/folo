// FOLO Studio: the app, mounted into #app.
import './firstScreen';
import { render } from 'fw';
import './lib/design/tokens.css';
import './lib/design/base.css';
import App from './App';
import { installHeartbeat } from './lib/ui/heartbeat';

const target = document.getElementById('app');
if (!target) throw new Error('#app missing');

// The live heartbeats hold still for most of each beat: paused then, so an idle page with a running run idles too.
installHeartbeat();

export default render(() => <App />, target);
