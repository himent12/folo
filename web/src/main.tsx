// The fw app (next.html): FOLO Studio as it is ported from Svelte (port of main.ts).
import './firstScreen';
import { render } from 'fw';
import './lib/design/tokens.css';
import './lib/design/base.css';
import App from './App';

const target = document.getElementById('app');
if (!target) throw new Error('#app missing');

export default render(() => <App />, target);
