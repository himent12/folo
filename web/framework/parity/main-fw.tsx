import { render } from 'fw';
import '$lib/design/tokens.css';
import '$lib/design/base.css';
import Demo from './Demo';

render(() => <Demo />, document.getElementById('app')!);
