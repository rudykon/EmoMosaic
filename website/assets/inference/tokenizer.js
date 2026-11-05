"use strict";
// BERT uncased BasicTokenizer + WordPiece, tested against the frozen Python
// tokenizer. Vocabulary, special tokens and truncation match training.
const BertTokenizer = (() => {
  function basic(text) {
    let clean = "";
    for (const character of text) {
      const n = character.codePointAt(0);
      if (n === 0 || n === 0xfffd) continue;
      if (/[\t\n\r\p{Zs}]/u.test(character)) clean += " ";
      else if (/[\p{Cc}\p{Cf}]/u.test(character)) continue;
      else if ((n >= 0x4e00 && n <= 0x9fff) || (n >= 0x3400 && n <= 0x4dbf) || (n >= 0x20000 && n <= 0x2fa1f) || (n >= 0xf900 && n <= 0xfaff)) clean += " " + character + " ";
      else clean += character;
    }
    const tokens = [];
    for (const word of clean.trim().split(/\s+/u)) {
      let token = "";
      for (const character of word.toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "")) {
        const code = character.codePointAt(0);
        const punctuation = (code >= 33 && code <= 47) || (code >= 58 && code <= 64) || (code >= 91 && code <= 96) || (code >= 123 && code <= 126) || /\p{P}/u.test(character);
        if (punctuation) { if (token) tokens.push(token); token = ""; tokens.push(character); }
        else token += character;
      }
      if (token) tokens.push(token);
    }
    return tokens;
  }
  function create(vocabulary) {
    const vocab = new Map(vocabulary.trimEnd().split(/\r?\n/).map((word, i) => [word, i]));
    return text => {
      const pieces = [];
      const tokens = text.split(/(\[UNK\]|\[SEP\]|\[PAD\]|\[CLS\]|\[MASK\])/).flatMap(part =>
        /^\[(UNK|SEP|PAD|CLS|MASK)\]$/.test(part) ? [part] : basic(part));
      for (const token of tokens) {
        if (/^\[(UNK|SEP|PAD|CLS|MASK)\]$/.test(token)) { pieces.push(vocab.get(token)); continue; }
        const chars = Array.from(token), word = [];
        if (chars.length > 100) { pieces.push(100); continue; }
        let start = 0, unknown = false;
        while (start < chars.length) {
          let end = chars.length, id;
          while (start < end) {
            id = vocab.get((start ? "##" : "") + chars.slice(start, end).join(""));
            if (id !== undefined) break;
            end--;
          }
          if (id === undefined) { unknown = true; break; }
          word.push(id); start = end;
        }
        pieces.push(...(unknown ? [100] : word));
      }
      const ids = [101, ...pieces.slice(0, 48), 102], length = ids.length;
      while (ids.length < 50) ids.push(0);
      return { ids, mask: ids.map((_, i) => i < length ? 1 : 0), length, truncated: pieces.length > 48 };
    };
  }
  return { create };
})();
if (typeof module !== "undefined") module.exports = BertTokenizer;
