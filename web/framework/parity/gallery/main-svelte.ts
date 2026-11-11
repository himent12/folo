// The ui gallery, Svelte side: every `sections/<Name>.svelte` (the twin of `sections/<Name>.tsx`), as main-fw.tsx.
import { mount } from 'svelte';
import '$lib/design/tokens.css';
import '$lib/design/base.css';
import './gallery.css';
import Gallery from './Gallery.svelte';

mount(Gallery, { target: document.getElementById('app')! });
