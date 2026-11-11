// FOLO Studio: the app, mounted into #app.
import './firstScreen';
import { render } from 'fw';
import './lib/design/tokens.css';
import './lib/design/base.css';
import App from './App';

const target = document.getElementById('app');
if (!target) throw new Error('#app missing');

export default render(() => <App />, target);
