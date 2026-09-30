/* Copy the original 300-DPI PNG bytes; local galleries need no file fetch. */
(() => {
  'use strict';
  const assets = JSON.parse(document.getElementById('clipboard-assets').textContent);
  const cache = new Map();
  function originalPng(name) {
    if (!cache.has(name)) {
      const asset = assets[name];
      if (!asset) throw new Error('Missing image.');
      const bytes = Uint8Array.from(atob(asset.base64), ch => ch.charCodeAt(0));
      cache.set(name, new Blob([bytes], {type:'image/png'}));
    }
    return cache.get(name);
  }
  document.querySelectorAll('[data-copy-image]').forEach(button => {
    const label = button.textContent;
    const status = document.getElementById(button.getAttribute('aria-describedby'));
    let timer;
    button.addEventListener('click', async () => {
      clearTimeout(timer);
      button.disabled = true;
      button.textContent = 'Copying…';
      status.textContent = '';
      try {
        if (!navigator.clipboard || typeof navigator.clipboard.write !== 'function' || typeof ClipboardItem === 'undefined') {
          throw new Error('Clipboard unavailable.');
        }
        const name = button.dataset.copyImage;
        await navigator.clipboard.write([new ClipboardItem({'image/png':originalPng(name)})]);
        button.textContent = 'Copied ✓';
        status.textContent = `Copied ${assets[name].width} × ${assets[name].height} PNG. Paste into your slide.`;
        status.dataset.state = 'success';
        timer = setTimeout(() => {button.textContent = label;},2500);
      } catch (error) {
        button.textContent = label;
        status.textContent = 'Copying was blocked. Use Open PNG and right-click → Copy image, or download the PNG.';
        status.dataset.state = 'error';
      } finally {button.disabled = false;}
    });
  });
})();
