/* =====================================================================
   TokenLab — tokenizer core
   ---------------------------------------------------------------------
   Pure functions, no DOM. Two real algorithms:

   1. "wordpiece"  Word + piece vocabulary (the classroom method, BERT-like)
      - keep the N most frequent whole words
      - add every single-character piece seen in training
      - encode with greedy longest-match-first; "##" marks a piece
        that continues a word.

   2. "bpe"        Byte-Pair-Encoding style (the GPT family's method)
      - start from single characters
      - repeatedly merge the most frequent adjacent pair
      - every merge becomes a new vocabulary token.

   Both share pre-tokenization, special tokens, encode and decode.
   ===================================================================== */
(function (root) {
  'use strict';

  const SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]'];
  const UNK = '[UNK]';
  const CONT = '##'; // marks a piece that continues the previous piece of the same word

  // A "word" is a run of letters / digits (any script), optionally with an
  // inner apostrophe (don't, it's). Everything else that isn't a space is a
  // single punctuation token.
  const PRE_TOKEN_RE = /[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}]+)*|[^\s\p{L}\p{M}\p{N}]/gu;
  const WORD_START_RE = /^[\p{L}\p{M}\p{N}]/u;

  /* ---------- Step 1: normalisation & pre-tokenization ---------- */

  function normalize(text, opts) {
    const t = String(text || '').normalize('NFC');
    return opts && opts.lowercase ? t.toLowerCase() : t;
  }

  /** Split text into words and punctuation, keeping character offsets. */
  function preTokenize(text) {
    const out = [];
    for (const m of text.matchAll(PRE_TOKEN_RE)) {
      const s = m[0];
      out.push({
        text: s,
        start: m.index,
        end: m.index + s.length,
        kind: WORD_START_RE.test(s) ? 'word' : 'punct',
      });
    }
    return out;
  }

  /** "zebra" -> ["z", "##e", "##b", "##r", "##a"] */
  function charPieces(word) {
    return Array.from(word).map((c, i) => (i === 0 ? c : CONT + c));
  }

  /** Join two pieces: ("c","##a") -> "ca", ("##a","##t") -> "##at" */
  function joinPieces(a, b) {
    return a + (b.startsWith(CONT) ? b.slice(CONT.length) : b);
  }

  /** Count every word / punctuation mark, remembering first appearance. */
  function countPreTokens(preTokens) {
    const map = new Map();
    preTokens.forEach((p, i) => {
      const e = map.get(p.text);
      if (e) e.count++;
      else map.set(p.text, { text: p.text, kind: p.kind, count: 1, first: i });
    });
    return [...map.values()].sort((a, b) => b.count - a.count || a.first - b.first);
  }

  /* ---------- Vocabulary: token <-> ID dictionary ---------- */

  class Vocab {
    constructor() {
      this.tokens = []; // id -> token string
      this.types = [];  // id -> 'special' | 'word' | 'subword' | 'char' | 'punct'
      this.index = new Map(); // token string -> id
    }
    add(token, type) {
      if (this.index.has(token)) return this.index.get(token);
      const id = this.tokens.length;
      this.tokens.push(token);
      this.types.push(type);
      this.index.set(token, id);
      return id;
    }
    has(token) { return this.index.has(token); }
    id(token) { return this.index.get(token); }
    get size() { return this.tokens.length; }
  }

  function addSpecials(vocab) {
    SPECIAL_TOKENS.forEach((t) => vocab.add(t, 'special'));
  }

  /**
   * Character pieces from all training words, most frequent first.
   * With opts.baseAlphabet, every a–z / 0–9 character is added too (as a
   * word-start piece and a "##" continuation piece), like the fixed base
   * alphabet real tokenizers ship with.
   */
  function collectCharPieces(wordEntries, opts) {
    const counts = new Map();
    wordEntries.forEach((w) => {
      charPieces(w.text).forEach((p) => counts.set(p, (counts.get(p) || 0) + w.count));
    });
    const pieces = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([p]) => p);
    if (opts && opts.baseAlphabet) {
      let alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789';
      if (!opts.lowercase) alphabet += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
      for (const c of alphabet) pieces.push(c, CONT + c);
    }
    return pieces;
  }

  /* ---------- Algorithm 1: word + piece (WordPiece-style) ---------- */

  function trainWordPiece(text, opts) {
    const norm = normalize(text, opts);
    const pre = preTokenize(norm);
    const entries = countPreTokens(pre);
    const words = entries.filter((e) => e.kind === 'word');
    const kept = new Set(words.slice(0, Math.max(0, opts.maxWords)).map((w) => w.text));

    const vocab = new Vocab();
    addSpecials(vocab);
    // Kept words and punctuation, most frequent first (IDs follow frequency).
    entries.forEach((e) => {
      if (e.kind === 'punct') vocab.add(e.text, 'punct');
      else if (kept.has(e.text)) vocab.add(e.text, 'word');
    });
    // Single-character pieces, so no word is ever completely unreadable.
    collectCharPieces(words, opts).forEach((p) => vocab.add(p, 'char'));

    return finish({
      algorithm: 'wordpiece',
      opts: { ...opts },
      vocab,
      merges: [],
      entries,
      keptWords: kept.size,
      uniqueWords: words.length,
      corpus: norm,
      preTokens: pre,
    });
  }

  /** Greedy longest-match-first, exactly like BERT's WordPiece encoder. */
  function wordPieceEncodeWord(tk, word) {
    const { vocab } = tk;
    if (vocab.has(word)) return [{ token: word, how: 'whole', source: word }];
    if (!tk.opts.subwordFallback) return [{ token: UNK, how: 'unk', source: word }];

    const cs = Array.from(word);
    const pieces = [];
    let start = 0;
    while (start < cs.length) {
      let end = cs.length;
      let found = null;
      while (end > start) {
        const sub = (start > 0 ? CONT : '') + cs.slice(start, end).join('');
        if (vocab.has(sub)) { found = sub; break; }
        end--;
      }
      if (found === null) {
        // Not even the single character is known: that character becomes [UNK].
        pieces.push({ token: UNK, how: 'unk', source: cs[start] });
        start += 1;
      } else {
        pieces.push({ token: found, how: 'piece', source: cs.slice(start, end).join('') });
        start = end;
      }
    }
    return pieces;
  }

  /* ---------- Algorithm 2: Byte-Pair Encoding ---------- */

  function trainBPE(text, opts) {
    const norm = normalize(text, opts);
    const pre = preTokenize(norm);
    const entries = countPreTokens(pre);
    const words = entries.filter((e) => e.kind === 'word');

    const vocab = new Vocab();
    addSpecials(vocab);
    entries.filter((e) => e.kind === 'punct').forEach((e) => vocab.add(e.text, 'punct'));
    collectCharPieces(words, opts).forEach((p) => vocab.add(p, 'char'));

    // Each unique word starts as a list of character pieces.
    const work = words.map((w) => ({ count: w.count, pieces: charPieces(w.text) }));
    const merges = [];
    const minPairCount = 2;
    let stopReason = 'limit';

    for (let step = 0; step < opts.numMerges; step++) {
      // 1. Count every adjacent pair, weighted by word frequency.
      const pairCounts = new Map();
      for (const w of work) {
        for (let i = 0; i < w.pieces.length - 1; i++) {
          const key = w.pieces[i] + '\u0000' + w.pieces[i + 1];
          pairCounts.set(key, (pairCounts.get(key) || 0) + w.count);
        }
      }
      // 2. Pick the most frequent pair (first seen wins ties -> deterministic).
      let bestKey = null;
      let bestCount = 0;
      for (const [k, c] of pairCounts) if (c > bestCount) { bestKey = k; bestCount = c; }
      if (!bestKey || bestCount < minPairCount) { stopReason = 'no-pairs'; break; }

      // 3. Merge that pair everywhere and record the rule.
      const [a, b] = bestKey.split('\u0000');
      const merged = joinPieces(a, b);
      for (const w of work) w.pieces = mergePair(w.pieces, a, b, merged);
      merges.push({ rank: merges.length, a, b, token: merged, count: bestCount });
      vocab.add(merged, 'subword');
    }

    // A merged token that spells a complete training word is a "word" token.
    const wordSet = new Set(words.map((w) => w.text));
    vocab.tokens.forEach((t, id) => {
      if (vocab.types[id] === 'subword' && wordSet.has(t)) vocab.types[id] = 'word';
    });

    const tk = finish({
      algorithm: 'bpe',
      opts: { ...opts },
      vocab,
      merges,
      stopReason,
      entries,
      uniqueWords: words.length,
      keptWords: vocab.types.filter((t) => t === 'word').length,
      corpus: norm,
      preTokens: pre,
    });
    return tk;
  }

  function mergePair(pieces, a, b, merged) {
    const out = [];
    let i = 0;
    while (i < pieces.length) {
      if (i < pieces.length - 1 && pieces[i] === a && pieces[i + 1] === b) {
        out.push(merged);
        i += 2;
      } else {
        out.push(pieces[i]);
        i += 1;
      }
    }
    return out;
  }

  /** Apply learned merges in the order they were learned (lowest rank first). */
  function bpeEncodeWord(tk, word) {
    let pieces = charPieces(word);
    while (pieces.length > 1) {
      let best = Infinity;
      let pair = null;
      for (let i = 0; i < pieces.length - 1; i++) {
        const r = tk.mergeRanks.get(pieces[i] + '\u0000' + pieces[i + 1]);
        if (r !== undefined && r < best) { best = r; pair = [pieces[i], pieces[i + 1]]; }
      }
      if (!pair) break;
      pieces = mergePair(pieces, pair[0], pair[1], joinPieces(pair[0], pair[1]));
    }
    const single = pieces.length === 1 && tk.vocab.has(pieces[0]);
    return pieces.map((p) => {
      const source = p.startsWith(CONT) ? p.slice(CONT.length) : p;
      if (!tk.vocab.has(p)) return { token: UNK, how: 'unk', source };
      return { token: p, how: single ? 'whole' : 'piece', source };
    });
  }

  /* ---------- Shared: finishing, encoding, decoding ---------- */

  /** Index merges, then count how often every token is used on the training text. */
  function finish(tk) {
    tk.mergeRanks = new Map(tk.merges.map((m) => [m.a + '\u0000' + m.b, m.rank]));
    tk.cache = new Map();
    const counts = new Array(tk.vocab.size).fill(0);
    let total = 0;
    for (const p of tk.preTokens) {
      for (const part of encodeWord(tk, p.text)) {
        counts[tk.vocab.id(part.token)]++;
        total++;
      }
    }
    tk.counts = counts;
    tk.totalTokens = total;
    tk.totalPreTokens = tk.preTokens.length;
    tk.totalChars = Array.from(tk.corpus.replace(/\s+/g, '')).length;
    delete tk.preTokens; // not needed after counting
    return tk;
  }

  function encodeWord(tk, word) {
    if (tk.cache.has(word)) return tk.cache.get(word);
    const parts = tk.algorithm === 'bpe' ? bpeEncodeWord(tk, word) : wordPieceEncodeWord(tk, word);
    tk.cache.set(word, parts);
    return parts;
  }

  function train(text, opts) {
    return opts.algorithm === 'bpe' ? trainBPE(text, opts) : trainWordPiece(text, opts);
  }

  /**
   * Encode new text with a frozen vocabulary.
   * Returns every step so the UI can show the whole pipeline.
   */
  function encode(tk, text, options) {
    const o = options || {};
    const normalized = normalize(text, tk.opts);
    const preTokens = preTokenize(normalized);
    const items = [];
    const special = (t) => ({
      token: t, id: tk.vocab.id(t), type: 'special', how: 'special', pre: -1, piece: 0, pieces: 1, source: '',
    });

    if (o.addBos) items.push(special('[BOS]'));
    preTokens.forEach((p, pi) => {
      const parts = encodeWord(tk, p.text);
      parts.forEach((part, k) => {
        const id = tk.vocab.id(part.token);
        items.push({
          token: part.token,
          id,
          type: tk.vocab.types[id],
          how: part.how,
          pre: pi,
          piece: k,
          pieces: parts.length,
          source: part.source,
        });
      });
    });
    if (o.addEos) items.push(special('[EOS]'));
    const contentLength = items.length;
    if (o.padTo && o.padTo > items.length) {
      while (items.length < o.padTo) items.push(special('[PAD]'));
    }
    return {
      normalized,
      preTokens,
      items,
      ids: items.map((i) => i.id),
      contentLength,
      overflow: o.padTo ? Math.max(0, contentLength - o.padTo) : 0,
    };
  }

  const NO_SPACE_BEFORE = /^[.,!?;:%)\]}…»”’]$/;
  const NO_SPACE_AFTER = /[(\[{«“‘$#@]$/;

  /** Turn IDs back into text. Spaces are re-inserted between words. */
  function decode(tk, ids, options) {
    const o = options || {};
    let out = '';
    const unknownIds = [];
    for (const raw of ids) {
      const id = Number(raw);
      if (!Number.isInteger(id) || id < 0 || id >= tk.vocab.size) {
        unknownIds.push(raw);
        continue;
      }
      const tok = tk.vocab.tokens[id];
      const type = tk.vocab.types[id];
      if (type === 'special' && o.skipSpecial && tok !== UNK) continue;
      if (tok.startsWith(CONT)) { out += tok.slice(CONT.length); continue; }
      const glue = out === '' || NO_SPACE_BEFORE.test(tok) || NO_SPACE_AFTER.test(out);
      out += (glue ? '' : ' ') + tok;
    }
    return { text: out, unknownIds };
  }

  /** Parse "[2, 14, 7]" or "2 14 7" into an array of strings/numbers. */
  function parseIds(str) {
    return String(str || '')
      .replace(/[\[\]]/g, ' ')
      .split(/[\s,]+/)
      .filter(Boolean)
      .map((s) => (/^-?\d+$/.test(s) ? Number(s) : s));
  }

  const api = {
    SPECIAL_TOKENS, UNK, CONT,
    normalize, preTokenize, charPieces, joinPieces,
    train, encode, decode, parseIds, encodeWord,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Tokenizer = api;
})(typeof window !== 'undefined' ? window : globalThis);
