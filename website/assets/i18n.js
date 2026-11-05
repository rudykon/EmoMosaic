/* Shared offline localization. Source text stays in Chinese; model data is untouched. */
(() => {
  'use strict';
  const dictionary = window.EmoMosaicEnglish;
  const escape = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(Object.keys(dictionary).sort((a, b) => b.length - a.length).map(escape).join('|'), 'g');
  const storageKey = 'emomosaic-language';
  let language = 'zh-CN';
  try { if (localStorage.getItem(storageKey) === 'en') language = 'en'; } catch (_) { /* Storage is optional. */ }
  // Web links may request English explicitly. Do not change the Android asset URL.
  const requested = new URLSearchParams(location.search).get('lang');
  if (requested === 'en' || requested === 'zh-CN') language = requested;
  function english(value) {
    if (!/[\u4e00-\u9fff]/.test(value)) return value;
    if (Object.hasOwn(dictionary, value.trim())) return value.replace(value.trim(), dictionary[value.trim()]);
    return value.replace(pattern, match => dictionary[match])
      .replace(/，/g, ', ').replace(/。/g, '. ').replace(/；/g, '; ').replace(/：/g, ': ')
      .replace(/、/g, ', ').replace(/「/g, '“').replace(/」/g, '”').replace(/（/g, ' (').replace(/）/g, ')');
  }
  const records = new WeakMap();
  const attributes = ['title', 'alt', 'placeholder', 'aria-label', 'aria-valuetext'];
  function localize(node, key, get, set) {
    let fields = records.get(node);
    if (!fields) { fields = new Map(); records.set(node, fields); }
    const current = get();
    let record = fields.get(key);
    // Application rendering can replace an existing text/attribute at any time.
    if (!record || current !== record.rendered) record = {source: current};
    const rendered = language === 'en' ? english(record.source) : record.source;
    record.rendered = rendered;
    fields.set(key, record);
    if (current !== rendered) set(rendered);
  }
  const excluded = element => element?.closest('script,style,code,pre,[translate="no"],[data-language-toggle]');
  function visit(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      if (!excluded(node.parentElement) && !node.parentElement?.closest('textarea')) localize(node, 'text', () => node.nodeValue, value => { node.nodeValue = value; });
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE || excluded(node)) return;
    for (const attr of attributes) if (node.hasAttribute(attr)) {
      localize(node, attr, () => node.getAttribute(attr), value => node.setAttribute(attr, value));
    }
    if (node.matches('meta[name="description"],meta[property="og:title"],meta[property="og:description"]')) {
      localize(node, 'content', () => node.content, value => { node.content = value; });
    }
    if (!node.matches('textarea')) for (const child of node.childNodes) visit(child);
  }
  let observer;
  const watch = () => observer.observe(document.documentElement, {subtree:true, childList:true, characterData:true, attributes:true, attributeFilter:[...attributes, 'content']});
  function refresh() {
    observer?.disconnect();
    document.documentElement.lang = language;
    visit(document.documentElement);
    document.querySelectorAll('[data-language-toggle]').forEach(button => {
      button.textContent = language === 'en' ? '中文' : 'English';
      button.lang = language === 'en' ? 'zh-CN' : 'en';
      button.setAttribute('aria-label', language === 'en' ? '切换为中文' : 'Switch to English');
    });
    if (observer) watch();
  }
  function setLanguage(next) {
    language = next === 'en' ? 'en' : 'zh-CN';
    try { localStorage.setItem(storageKey, language); } catch (_) { /* In-memory switching still works. */ }
    if (location.search.includes('lang=')) {
      const url = new URL(location.href); url.searchParams.set('lang', language);
      history.replaceState(null, '', url);
    }
    refresh();
    document.dispatchEvent(new CustomEvent('languagechange', {detail:language}));
  }
  window.EmoI18n = {t: value => language === 'en' ? english(String(value)) : String(value), english, setLanguage, refresh, get language() { return language; }};
  document.documentElement.lang = language;
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-language-toggle]').forEach(button => button.addEventListener('click', () => setLanguage(language === 'en' ? 'zh-CN' : 'en')));
    observer = new MutationObserver(changes => {
      observer.disconnect();
      const nodes = new Set();
      for (const change of changes) {
        if (change.type === 'childList') change.addedNodes.forEach(node => nodes.add(node));
        else nodes.add(change.target);
      }
      nodes.forEach(visit);
      watch();
    });
    refresh();
  });
})();
