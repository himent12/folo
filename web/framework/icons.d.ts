// `import X from '$icons/x'`: a lucide icon as an fw component (built by the fw Vite plugin from @lucide/svelte's data).
declare module '$icons/*' {
  const Icon: import('./runtime/icon').IconComponent;
  export default Icon;
}
