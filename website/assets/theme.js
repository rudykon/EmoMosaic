// Apply the saved preference before rendering to avoid a bright flash.
(() => {
  let theme;
  try { theme = localStorage.getItem('emomosaic-theme'); } catch (_) { /* Storage may be disabled. */ }
  document.documentElement.dataset.theme = theme === 'dark' || theme === 'light'
    ? theme : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
})();
