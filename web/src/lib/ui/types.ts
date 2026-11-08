import type { Component } from 'svelte';
import type { LucideProps } from '@lucide/svelte';

/** An icon component (lucide or our own), rendered at the size the parent picks. */
export type IconType = Component<LucideProps>;
