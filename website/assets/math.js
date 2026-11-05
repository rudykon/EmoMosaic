'use strict';

// Only explicitly marked mathematical content is rendered. Code, inputs and
// imported data remain plain text; localization skips the mathematical DOM.
(() => {
  const escape = value => String(value).replace(/[&<>"']/g, character =>
    ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[character]));
  function render(root = document) {
    const elements = [...(root.matches?.('[data-math]') ? [root] : []), ...root.querySelectorAll('[data-math]')];
    for (const element of elements) {
      const source = element.dataset.math;
      if (element.dataset.mathRendered === source) continue;
      element.setAttribute('translate', 'no');
      try {
        katex.render(source, element, {
          displayMode: element.classList.contains('math-display'),
          output: 'htmlAndMathml', throwOnError: true, strict: 'error', trust: false,
        });
        element.dataset.mathRendered = source;
      } catch (error) {
        element.dataset.mathError = 'true';
        console.error('Unable to typeset formula:', error.message);
      }
    }
  }
  function inline(source) {
    return `<span class="math-inline" data-math="${escape(source)}" translate="no">${escape(source)}</span>`;
  }
  window.EmoMath = {render, inline};
  render();
  // Sample selection and metric filters insert new formulas after load.
  const observer = new MutationObserver(changes => {
    for (const change of changes) {
      if (change.type === 'attributes') render(change.target);
      else for (const node of change.addedNodes) if (node.nodeType === Node.ELEMENT_NODE) render(node);
    }
  });
  observer.observe(document.querySelector('main'), {subtree:true, childList:true, attributes:true, attributeFilter:['data-math']});
})();
