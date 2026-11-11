import { mount } from 'svelte';
import '$lib/design/tokens.css';
import '$lib/design/base.css';
import Demo from './Demo.svelte';

mount(Demo, { target: document.getElementById('app')! });
