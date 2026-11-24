/* =====================================================================
   TokenLab — UI wiring
   ---------------------------------------------------------------------
   Reads the controls, calls Tokenizer (js/tokenizer.js), and renders:
   hero pipeline, tokenization trace, vocabulary, token IDs, chart.
   ===================================================================== */
(function () {
  'use strict';

  const T = window.Tokenizer;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (n) => n.toLocaleString('en-IN');
  const pct = (x) => (x * 100 >= 10 ? (x * 100).toFixed(0) : (x * 100).toFixed(1)) + '%';
  const plural = (n, one, many) => `${fmt(n)} ${n === 1 ? one : many || one + 's'}`;
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  const TYPE_LABEL = {
    special: 'Special token',
    word: 'Whole word',
    subword: 'Merged piece',
    char: 'Single character',
    punct: 'Punctuation',
  };

  const state = {
    algorithm: 'wordpiece',
    tk: null,
    enc: null,
    tokenView: 'test',
    vocabTab: 'table',
    chartView: 'compare',
    testCounts: new Map(),
    showAllRows: false,
    stale: false,
    animate: true,
  };

  /* ---------------- options ---------------- */

  function readOpts() {
    return {
      algorithm: state.algorithm,
      lowercase: $('lowercase').checked,
      baseAlphabet: $('alphabet').checked,
      subwordFallback: $('fallback').checked,
      maxWords: Number($('maxwords').value),
      numMerges: Number($('merges').value),
    };
  }

  function encodeOpts() {
    return {
      addBos: $('add-bos').checked,
      addEos: $('add-eos').checked,
      padTo: Math.max(0, Math.min(256, Number($('pad-to').value) || 0)),
    };
  }

  /* ---------------- build & encode ---------------- */

  function build() {
    const t0 = performance.now();
    state.tk = T.train($('corpus').value, readOpts());
    const ms = performance.now() - t0;
    state.stale = false;
    const tk = state.tk;
    const extra = tk.algorithm === 'bpe'
      ? ` It learned ${plural(tk.merges.length, 'merge')}${tk.stopReason === 'no-pairs' && tk.merges.length < tk.opts.numMerges ? ' and stopped early because no pair of pieces appears twice any more' : ''}.`
      : '';
    $('build-status').className = 'build-status ok';
    $('build-status').textContent = `Built ${plural(tk.vocab.size, 'token')} in ${ms < 1 ? '<1' : Math.round(ms)} ms.${extra}`;
    updateMaxWordsRange();
    renderVocabStats();
    encodeTest();
    startEmbedding();
  }

  function encodeTest() {
    if (!state.tk) return;
    state.enc = T.encode(state.tk, $('test').value, encodeOpts());
    state.testCounts = new Map();
    state.enc.items.forEach((it) => state.testCounts.set(it.id, (state.testCounts.get(it.id) || 0) + 1));
    renderPipeline();
    renderTokens();
    renderIds();
    renderVocabTable();
    renderMerges();
    renderChart();
    $('decode-in').value = state.enc.ids.join(', ');
    runDecode();
    if (emb.trainer) { updateEmbedView(); renderNeighbours(); }
  }

  /* ---------------- chips ---------------- */

  function tokenText(tok) {
    if (tok.startsWith(T.CONT)) return `<span class="cont">##</span>${esc(tok.slice(2))}`;
    return esc(tok);
  }

  function chipClass(item) {
    if (item.token === T.UNK) return 'chip chip-unk';
    if (item.type === 'special') return 'chip chip-special';
    return `chip h${item.id % 8}${item.how === 'piece' ? ' chip-piece' : ''}`;
  }

  function chip(item, withId) {
    return `<span class="${chipClass(item)}" data-id="${item.id}" data-how="${item.how}" tabindex="0">` +
      `<span class="chip-t">${tokenText(item.token)}</span>` +
      (withId ? `<small class="chip-id">${item.id}</small>` : '') + '</span>';
  }

  /** Group encoded items by the word they came from; split words get a bracket. */
  function groupedChips(enc, withId, limit) {
    const out = [];
    let i = 0;
    const items = limit ? enc.items.slice(0, limit) : enc.items;
    while (i < items.length) {
      const it = items[i];
      if (it.pre < 0) { out.push(chip(it, withId)); i++; continue; }
      const group = [];
      while (i < items.length && items[i].pre === it.pre) group.push(items[i++]);
      const word = enc.preTokens[it.pre].text;
      const isSplit = group.length > 1;
      const wholeUnk = group.length === 1 && group[0].token === T.UNK;
      if (isSplit || wholeUnk) {
        const cap = wholeUnk
          ? `“${esc(word)}” unknown`
          : `“${esc(word)}” ${group.length} pieces`;
        out.push(`<span class="grp"><span class="grp-chips">${group.map((g) => chip(g, withId)).join('')}</span><span class="grp-cap">${cap}</span></span>`);
      } else {
        out.push(chip(group[0], withId));
      }
    }
    return out.join('');
  }

  /* ---------------- hero: Tiktokenizer-style live view ---------------- */

  /** Colour each token inside the user's own text, like Tiktokenizer does. */
  function highlightHTML(enc, raw) {
    // Keep the user's original casing when lowercasing didn't change the length.
    const text = Array.from(raw.normalize('NFC')).length === Array.from(enc.normalized).length ? raw.normalize('NFC') : enc.normalized;
    const gap = (str) => esc(str).replace(/\n/g, '<span class="nl">↵</span><br>');
    const span = (it, str, i) => {
      const cls = it.token === T.UNK ? 'hl-unk' : `h${it.id % 8}`;
      return `<span class="hl-t ${cls}${it.how === 'piece' ? ' hl-piece' : ''}" data-id="${it.id}" style="--i:${i}">${esc(str)}</span>`;
    };
    let html = '';
    let pos = 0;
    let k = 0;
    let lastPre = -1;
    let offset = 0;
    for (const it of enc.items) {
      if (it.pre < 0) {
        if (it.token !== '[PAD]') html += `<span class="hl-special" data-id="${it.id}" style="--i:${k++}">${esc(it.token)}</span>`;
        continue;
      }
      if (it.pre !== lastPre) { offset = enc.preTokens[it.pre].start; lastPre = it.pre; }
      const len = it.source.length;
      if (offset > pos) html += `<span class="hl-gap">${gap(text.slice(pos, offset))}</span>`;
      html += span(it, text.slice(offset, offset + len), k++);
      offset += len;
      pos = offset;
    }
    if (pos < text.length) html += `<span class="hl-gap">${gap(text.slice(pos))}</span>`;
    return html;
  }

  function renderPipeline() {
    const enc = state.enc;
    const raw = $('test').value;
    const content = enc.items.filter((it) => it.token !== '[PAD]');
    const words = enc.items.filter((it) => it.pre >= 0).length;
    $('hero-chars').textContent = `${plural(Array.from(raw).length, 'character')}`;
    $('hero-count').textContent = fmt(words);
    $('hero-idcount').textContent = fmt(content.length);
    $('hero-hl').innerHTML = highlightHTML(enc, raw) || '<span class="na">Type something on the left.</span>';
    $('pl-ids').innerHTML = content.map((it, i) =>
      `<span class="hid ${it.token === T.UNK ? 'hl-unk' : it.type === 'special' ? 'hid-special' : 'h' + (it.id % 8)}" data-id="${it.id}" style="--i:${i}">${it.id}</span>`).join('');
    document.getElementById('hero').classList.toggle('popping', state.animate);
    if (state.animate) {
      // restart the CSS pop-in animation for an explicit Tokenize
      const h = $('hero');
      h.classList.remove('popping');
      void h.offsetWidth;
      h.classList.add('popping');
      state.animate = false;
    }
  }

  /* ---------------- 2. tokenization ---------------- */

  function renderTokens() {
    const tk = state.tk;
    const isTest = state.tokenView === 'test';
    const enc = isTest ? state.enc : T.encode(tk, $('corpus').value, {});
    const LIMIT = 700;

    $('trace-norm').textContent = enc.normalized.length > 600 ? enc.normalized.slice(0, 600) + '…' : enc.normalized;
    const pre = enc.preTokens.slice(0, 300);
    $('trace-pre').innerHTML = pre.map((p) => `<span class="pre pre-${p.kind}">${esc(p.text)}</span>`).join('') +
      (enc.preTokens.length > 300 ? `<span class="more">and ${fmt(enc.preTokens.length - 300)} more</span>` : '');
    $('token-view').innerHTML = groupedChips(enc, true, LIMIT) +
      (enc.items.length > LIMIT ? `<span class="more">and ${fmt(enc.items.length - LIMIT)} more tokens</span>` : '');

    // summary sentence
    const content = enc.items.filter((it) => it.type !== 'special' || it.token === T.UNK);
    const words = enc.preTokens.filter((p) => p.kind === 'word').length;
    const puncts = enc.preTokens.length - words;
    const splitWords = new Set(content.filter((it) => it.pieces > 1).map((it) => it.pre)).size;
    const unk = content.filter((it) => it.token === T.UNK).length;
    const chars = Array.from(enc.normalized.replace(/\s+/g, '')).length;
    const who = isTest ? 'Your test sentence' : 'The training corpus';
    $('token-summary').innerHTML = enc.preTokens.length
      ? `${who} has ${plural(words, 'word')} and ${plural(puncts, 'punctuation mark')}. They became <b>${plural(content.length, 'token')}</b>` +
        ` (${content.length ? (chars / content.length).toFixed(1) : 0} characters per token). ` +
        `${splitWords ? `${plural(splitWords, 'word')} had to be split into pieces. ` : 'No word had to be split. '}` +
        `${unk ? `<span class="unk-inline">${plural(unk, '[UNK] token')}</span>: those characters are lost.` : 'Nothing became [UNK].'}`
      : 'Type something to tokenize.';
  }

  /* ---------------- 3. vocabulary ---------------- */

  function vocabComposition(tk) {
    const c = { special: 0, word: 0, piece: 0, punct: 0 };
    tk.vocab.types.forEach((t) => { c[t === 'subword' || t === 'char' ? 'piece' : t]++; });
    return c;
  }

  function renderVocabStats() {
    const tk = state.tk;
    const comp = vocabComposition(tk);
    // share of word occurrences in training encoded as one whole token
    let wordOcc = 0, wholeOcc = 0;
    tk.entries.forEach((e) => {
      if (e.kind !== 'word') return;
      wordOcc += e.count;
      const parts = T.encodeWord(tk, e.text);
      if (parts.length === 1 && parts[0].how === 'whole') wholeOcc += e.count;
    });
    const stat = (value, label, note) => `<div class="vstat"><b>${value}</b><span>${label}</span>${note ? `<em>${note}</em>` : ''}</div>`;
    $('vocab-stats').innerHTML =
      stat(fmt(tk.vocab.size), 'Vocabulary size', `${comp.special} special, ${comp.word} words, ${comp.piece} pieces, ${comp.punct} punctuation`) +
      stat(fmt(tk.totalTokens), 'Tokens in training text', `from ${plural(tk.totalPreTokens, 'word')} and punctuation marks`) +
      stat(`${fmt(tk.keptWords)} <small>of ${fmt(tk.uniqueWords)}</small>`, 'Unique words kept whole', tk.algorithm === 'bpe' ? 'built up by merges' : 'set by the slider') +
      stat(wordOcc ? pct(wholeOcc / wordOcc) : '–', 'Word coverage', 'training words encoded as a single token') +
      stat(tk.totalTokens ? (tk.totalChars / tk.totalTokens).toFixed(2) : '–', 'Characters per token', 'higher means better compression');

    const total = tk.vocab.size;
    const seg = (k, label) => comp[k]
      ? `<span class="cseg c-${k}" style="flex-grow:${comp[k]}" title="${label}: ${comp[k]}"><b>${comp[k]}</b> ${label}</span>`
      : '';
    $('vocab-composition').innerHTML = `<div class="cbar">${seg('special', 'special')}${seg('word', 'words')}${seg('piece', 'pieces')}${seg('punct', 'punctuation')}</div>` +
      `<span class="ctotal">${fmt(total)} IDs, numbered 0 to ${fmt(total - 1)}</span>`;
  }

  function vocabRows() {
    const tk = state.tk;
    const q = $('vsearch').value.trim().toLowerCase();
    const filter = $('vfilter').value;
    const sort = $('vsort').value;
    let rows = tk.vocab.tokens.map((token, id) => ({ id, token, type: tk.vocab.types[id], count: tk.counts[id] }));
    if (q) rows = rows.filter((r) => r.token.toLowerCase().includes(q) || String(r.id) === q);
    if (filter === 'unused') rows = rows.filter((r) => r.count === 0);
    else if (filter === 'test') rows = rows.filter((r) => state.testCounts.has(r.id));
    else if (filter !== 'all') rows = rows.filter((r) => r.type === filter);
    const cmp = {
      id: (a, b) => a.id - b.id,
      'count-desc': (a, b) => b.count - a.count || a.id - b.id,
      'count-asc': (a, b) => a.count - b.count || a.id - b.id,
      alpha: (a, b) => a.token.replace(/^##/, '').localeCompare(b.token.replace(/^##/, '')) || a.id - b.id,
      length: (a, b) => Array.from(b.token.replace(/^##/, '')).length - Array.from(a.token.replace(/^##/, '')).length || a.id - b.id,
    }[sort];
    return rows.sort(cmp);
  }

  function renderVocabTable() {
    const tk = state.tk;
    const rows = vocabRows();
    const max = Math.max(1, ...tk.counts);
    const LIMIT = 250;
    const shown = state.showAllRows ? rows : rows.slice(0, LIMIT);
    $('vcount').textContent = `${plural(rows.length, 'token')} shown`;
    $('vbody').innerHTML = shown.map((r) => {
      const inTest = state.testCounts.get(r.id);
      return `<tr data-id="${r.id}" class="${inTest ? 'in-test' : ''}">` +
        `<td class="mono">${r.id}</td>` +
        `<td><span class="vtok t-${r.type}">${tokenText(r.token)}</span>${inTest ? `<span class="badge-test">in test ×${inTest}</span>` : ''}</td>` +
        `<td class="num">${r.type === 'special' ? '<span class="na">reserved</span>' : `<span class="mini"><i style="width:${(r.count / max) * 100}%"></i></span>${fmt(r.count)}`}</td>` +
        `<td class="num">${r.type === 'special' || !tk.totalTokens ? '' : pct(r.count / tk.totalTokens)}</td>` +
        `<td class="type">${TYPE_LABEL[r.type]}</td></tr>`;
    }).join('') + (rows.length > shown.length
      ? `<tr class="more-row"><td colspan="5"><button type="button" class="btn btn-quiet" id="show-all">Show all ${fmt(rows.length)} tokens</button></td></tr>`
      : '') + (rows.length ? '' : '<tr class="more-row"><td colspan="5">No tokens match. Clear the search or pick another type.</td></tr>');
    const btn = $('show-all');
    if (btn) btn.addEventListener('click', () => { state.showAllRows = true; renderVocabTable(); });
  }

  function renderMerges() {
    const tk = state.tk;
    if (tk.algorithm !== 'bpe') {
      $('merge-help').textContent = 'Merge rules only exist for BPE. Switch the algorithm to "BPE merges" in step 1 to see them.';
      $('mbody').innerHTML = '';
      return;
    }
    $('merge-help').textContent = tk.merges.length
      ? `BPE started from single characters and learned ${plural(tk.merges.length, 'merge')}. At each step it counted every neighbouring pair of pieces and glued the most frequent pair together. New text is encoded by replaying these rules in the same order.`
      : 'No merges yet. Raise "Merges to learn", or add more training text so some pairs appear at least twice.';
    $('mbody').innerHTML = tk.merges.map((m) => {
      const id = tk.vocab.id(m.token);
      const used = state.testCounts.has(id);
      return `<tr data-id="${id}" class="${used ? 'in-test' : ''}"><td class="mono">${m.rank + 1}</td>` +
        `<td><span class="vtok">${tokenText(m.a)}</span> + <span class="vtok">${tokenText(m.b)}</span></td>` +
        `<td><span class="vtok t-${tk.vocab.types[id]}">${tokenText(m.token)}</span>${used ? '<span class="badge-test">in test</span>' : ''}</td>` +
        `<td class="mono">${id}</td><td class="num">${fmt(m.count)}×</td></tr>`;
    }).join('');
  }

  /* ---------------- 4. token IDs ---------------- */

  const HOW = {
    whole: 'Found whole in the vocabulary',
    piece: 'Longest matching piece',
    bpe: 'Built by replaying merge rules',
    unk: 'Nothing matched, so [UNK]',
    special: 'Added by the encoder',
  };

  function howText(item) {
    if (item.how === 'piece' && state.tk.algorithm === 'bpe') return HOW.bpe;
    return HOW[item.how];
  }

  function renderIds() {
    const enc = state.enc;
    $('id-len').textContent = `(${plural(enc.ids.length, 'ID')})`;
    $('id-array').innerHTML = '[' + enc.items.map((it) =>
      `<span data-id="${it.id}" class="${it.token === T.UNK ? 'id-unk' : it.type === 'special' ? 'id-special' : ''}">${it.id}</span>`).join(', ') + ']';

    const o = encodeOpts();
    const padded = enc.items.length - enc.contentLength;
    $('pad-note').textContent = enc.overflow
      ? `This sequence needs ${enc.contentLength} tokens, ${enc.overflow} more than the pad length of ${o.padTo}. A real model would cut off (truncate) the extra tokens.`
      : padded
        ? `Added ${plural(padded, '[PAD] token')} so the sequence is exactly ${o.padTo} long. Models process sentences in batches, and every sequence in a batch must be the same length.`
        : '';

    $('map-body').innerHTML = enc.items.map((it, i) => {
      let from = '<span class="na">none</span>';
      if (it.pre >= 0) {
        const word = enc.preTokens[it.pre].text;
        from = it.pieces > 1 || it.token === T.UNK && word !== it.source
          ? `<span class="mono">${esc(it.source)}</span> <span class="na">of “${esc(word)}”</span>`
          : `<span class="mono">${esc(word)}</span>`;
      }
      return `<tr data-id="${it.id}"><td class="mono">${i}</td><td>${from}</td><td>${chip(it, false)}</td>` +
        `<td class="mono id-cell">${it.id}</td><td class="type">${howText(it)}</td></tr>`;
    }).join('');
  }

  function runDecode() {
    if (!state.tk) return;
    const ids = T.parseIds($('decode-in').value);
    const res = T.decode(state.tk, ids, { skipSpecial: $('skip-special').checked });
    const original = $('test').value;
    let html = `<div class="dec-text">${res.text ? esc(res.text) : '<span class="na">(nothing to show)</span>'}</div>`;
    if (res.unknownIds.length) {
      html += `<p class="dec-warn">Not in this vocabulary, skipped: ${res.unknownIds.map(esc).join(', ')}. Valid IDs run from 0 to ${state.tk.vocab.size - 1}.</p>`;
    }
    const matchesEncoding = ids.join(',') === state.enc.ids.join(',');
    if (matchesEncoding) {
      if (res.text === original) {
        html += '<p class="dec-ok">Round trip is exact: decoding gave back exactly what you typed.</p>';
      } else {
        const reasons = [];
        if (state.tk.opts.lowercase && original !== original.toLowerCase()) reasons.push('capital letters were lowercased');
        if (res.text.includes(T.UNK)) reasons.push('[UNK] replaced characters the vocabulary did not have');
        if (!reasons.length) reasons.push('spacing around punctuation was rebuilt by rule, not remembered');
        html += `<p class="dec-diff">Not an exact round trip, because ${reasons.join(', and ')}.</p>` +
          `<dl class="dec-compare"><dt>You typed</dt><dd>${esc(original)}</dd><dt>Decoded</dt><dd>${esc(res.text)}</dd></dl>`;
      }
    }
    $('decode-out').innerHTML = html;
  }

  /* ---------------- 5. chart ---------------- */

  function chartRows() {
    const tk = state.tk;
    const withPunct = $('chart-punct').checked;
    const withUnused = $('chart-unused').checked;
    const rows = [];
    tk.vocab.tokens.forEach((token, id) => {
      const type = tk.vocab.types[id];
      const count = tk.counts[id];
      if (type === 'special') return;
      if (type === 'punct' && !withPunct) return;
      if (count === 0 && !withUnused) return;
      rows.push({ id, token, type, count, inTest: state.testCounts.has(id) });
    });
    rows.sort((a, b) => b.count - a.count || a.id - b.id);
    rows.forEach((r, i) => { r.rank = i + 1; r.share = tk.totalTokens ? r.count / tk.totalTokens : 0; });
    return rows;
  }

  function describe(id, extra) {
    const tk = state.tk;
    const token = tk.vocab.tokens[id];
    const type = tk.vocab.types[id];
    const count = tk.counts[id];
    const inTest = state.testCounts.get(id);
    const line = (k, v) => `<div><span>${k}</span><b>${v}</b></div>`;
    return `<div class="tip-token">${tokenText(token)}</div>` +
      line('ID', id) +
      line('Type', TYPE_LABEL[type]) +
      (type === 'special'
        ? line('Purpose', { '[PAD]': 'Fills empty slots', '[UNK]': 'Stands in for unknown text', '[BOS]': 'Marks the start', '[EOS]': 'Marks the end' }[token])
        : line('Used in training', count === 0
          ? (type === 'char' ? 'never (from the base alphabet)' : 'never')
          : `${plural(count, 'time')} (${pct(count / tk.totalTokens)})`)) +
      (extra && extra.rank ? line('Rank', `${extra.rank} of ${extra.of}`) : '') +
      (extra && extra.how ? line('How found', howText(extra)) : '') +
      (extra && extra.neighbours && extra.neighbours.length
        ? line('Closest', extra.neighbours.map((n) => esc(tk.vocab.tokens[n])).join(', '))
        : '') +
      line('In test sentence', inTest ? `yes, ${inTest}×` : 'no');
  }

  function renderChart() {
    const rows = chartRows();
    const view = state.chartView;
    $('f-topn').hidden = view !== 'compare';
    $('topn-out').textContent = $('topn').value;
    FreqChart.render($('chart'), rows, {
      view,
      n: Number($('topn').value),
      describe: (r) => describe(r.id, { rank: r.rank, of: rows.length }),
      onHover: (id) => highlight(id),
    });
    renderInsight(rows);
  }

  function renderInsight(rows) {
    const tk = state.tk;
    const used = rows.filter((r) => r.count > 0);
    if (!used.length || !tk.totalTokens) { $('insight').textContent = ''; return; }
    const total = used.reduce((s, r) => s + r.count, 0);
    const k = Math.max(1, Math.round(used.length * 0.1));
    const topShare = used.slice(0, k).reduce((s, r) => s + r.count, 0) / total;
    const once = used.filter((r) => r.count === 1).length;
    const first = used[0];
    $('insight').innerHTML =
      `<b>“${esc(first.token)}”</b> alone makes up ${pct(first.count / total)} of the ${fmt(total)} tokens shown. ` +
      `The top ${plural(k, 'token')} (10% of those used) cover <b>${pct(topShare)}</b> of all uses, ` +
      `while ${plural(once, 'token')} ${once === 1 ? 'appears' : 'appear'} only once.`;
  }

  /* ---------------- 6. embeddings (word2vec, trained live) ---------------- */

  const emb = { trainer: null, raf: 0, paused: false, seed: 7, perFrame: 0, shown: [], view: null, galaxy: null, frame: 0 };

  function startEmbedding() {
    cancelAnimationFrame(emb.raf);
    emb.raf = 0;
    emb.trainer = new Embedding.Trainer(state.tk, $('corpus').value, { dim: 16, window: 2, negatives: 4, seed: emb.seed });
    emb.perFrame = Math.max(50, Math.ceil(emb.trainer.total / 200)); // about 3 seconds of animation
    emb.paused = false;
    $('emb-pause').textContent = 'Pause';
    emb.trainer.step(0);
    updateEmbedView();
    renderNeighbours();
    animateEmbedding();
  }

  function animateEmbedding() {
    if (emb.raf || emb.paused || !emb.trainer || emb.trainer.finished) return;
    const tick = () => {
      emb.raf = 0;
      const tr = emb.trainer;
      if (!tr || emb.paused) return;
      tr.step(emb.perFrame);
      updateEmbedView();
      if (++emb.frame % 15 === 0 || tr.finished) renderNeighbours();
      if (!tr.finished) emb.raf = requestAnimationFrame(tick);
    };
    emb.raf = requestAnimationFrame(tick);
  }

  function updateEmbedView() {
    const tr = emb.trainer;
    if (!tr) return;
    const tk = state.tk;
    const withPunct = $('emb-punct').checked;
    const ids = tr.used.filter((id) => withPunct || tk.vocab.types[id] !== 'punct');
    emb.shown = ids;
    const pos = tr.project3(ids);
    const top = new Set(ids.slice().sort((a, b) => tr.counts[b] - tr.counts[a]).slice(0, ids.length <= 40 ? ids.length : 18));
    const rows = ids.map((id) => ({
      id,
      token: tk.vocab.tokens[id],
      type: tk.vocab.types[id],
      count: tk.counts[id],
      inTest: state.testCounts.has(id),
      label: top.has(id) || state.testCounts.has(id),
      pos: pos.get(id),
    }));
    const path = $('emb-path').checked && state.enc ? state.enc.items.filter((it) => it.type !== 'special').map((it) => it.id) : null;
    if (!emb.view) {
      emb.view = new Chart3D($('embed3d'), {
        describe: (r) => describe(r.id, { neighbours: neighbourIds(r.id, 4) }),
        onHover: (id) => highlight(id),
        neighboursOf: (id) => neighbourIds(id, 5),
      });
      $('c3-auto').checked = emb.view.auto;
    }
    emb.view.setData(rows, { mode: 'embed', path });
    if (!emb.galaxy) emb.galaxy = new Chart3D($('hero'), { minimal: true });
    emb.galaxy.setData(rows, { mode: 'embed' });

    const pctDone = Math.round(tr.progress * 100);
    $('emb-fill').style.width = pctDone + '%';
    $('emb-status').textContent = !tr.total
      ? 'Not enough text to train on. Add a few more sentences.'
      : tr.finished
        ? `Trained: ${fmt(tr.epochs)} passes over ${plural(tr.pairsPerEpoch, 'neighbour pair')}. Final loss ${tr.lastLoss.toFixed(2)}.`
        : `Training: pass ${fmt(tr.epoch)} of ${fmt(tr.epochs)}${tr.lastLoss ? `, loss ${tr.lastLoss.toFixed(2)}` : ''}`;
  }

  function neighbourIds(id, k) {
    if (!emb.trainer || !emb.trainer.counts[id]) return [];
    return emb.trainer.neighbours(id, k, emb.shown).map((n) => n.id);
  }

  function renderNeighbours() {
    const tr = emb.trainer;
    if (!tr || !state.enc) return;
    const tk = state.tk;
    const seen = new Set();
    const rows = [];
    state.enc.items.forEach((it) => {
      if (it.type === 'special' && it.token !== T.UNK) return;
      if (seen.has(it.id)) return;
      seen.add(it.id);
      let cell;
      if (it.token === T.UNK) cell = '<span class="na">[UNK] stands for anything unknown, so its embedding means nothing in particular.</span>';
      else if (!tr.counts[it.id]) cell = '<span class="na">Never appeared in the training text, so its embedding was never trained.</span>';
      else {
        cell = tr.neighbours(it.id, 4, emb.shown).map((n) =>
          `<span class="nb"><span class="vtok t-${tk.vocab.types[n.id]}" data-id="${n.id}">${tokenText(tk.vocab.tokens[n.id])}</span><em>${n.sim.toFixed(2)}</em></span>`).join('');
      }
      rows.push(`<tr data-id="${it.id}"><td><span class="vtok t-${it.type}">${tokenText(it.token)}</span></td><td class="mono">${it.id}</td><td>${cell}</td></tr>`);
    });
    $('emb-body').innerHTML = rows.join('') || '<tr class="more-row"><td colspan="3">Type a test sentence to compare its tokens.</td></tr>';
  }

  /* ---------------- hover linking ---------------- */

  let hot = null;
  function highlight(id) {
    if (hot !== null) document.querySelectorAll(`[data-id="${hot}"]`).forEach((el) => el.classList.remove('is-hot'));
    hot = id;
    if (id !== null && id !== undefined) document.querySelectorAll(`[data-id="${id}"]`).forEach((el) => el.classList.add('is-hot'));
  }

  function setupTooltips() {
    const tip = $('tip');
    const findItem = (el) => {
      // chips in the test views map back to their encoded item for "how found"
      if (!state.enc) return null;
      const id = Number(el.dataset.id);
      return state.enc.items.find((it) => it.id === id && it.how === el.dataset.how) || null;
    };
    const show = (el) => {
      if (!state.tk) return;
      const id = Number(el.dataset.id);
      const item = el.classList.contains('chip') ? findItem(el) : null;
      tip.innerHTML = describe(id, item);
      tip.hidden = false;
      const b = el.getBoundingClientRect();
      const left = Math.min(window.innerWidth - tip.offsetWidth - 8, Math.max(8, b.left + b.width / 2 - tip.offsetWidth / 2));
      let top = b.top - tip.offsetHeight - 8;
      if (top < 8) top = b.bottom + 8;
      tip.style.left = left + 'px';
      tip.style.top = top + 'px';
      highlight(id);
    };
    const hide = () => { tip.hidden = true; highlight(null); };
    const target = (e) => e.target.closest && e.target.closest('.chip[data-id], .id-array [data-id], .pl-ids [data-id], .hl [data-id], .vtable tr[data-id]');
    document.addEventListener('mouseover', (e) => { const el = target(e); if (el) show(el); });
    document.addEventListener('mouseout', (e) => { const el = target(e); if (el && !el.contains(e.relatedTarget)) hide(); });
    document.addEventListener('focusin', (e) => { const el = target(e); if (el) show(el); });
    document.addEventListener('focusout', (e) => { if (target(e)) hide(); });
    window.addEventListener('scroll', () => { if (!tip.hidden) hide(); }, { passive: true });
  }

  /* ---------------- controls ---------------- */

  function setAlgorithm(algo) {
    state.algorithm = algo;
    document.querySelectorAll('#algo button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.algo === algo)));
    $('f-maxwords').hidden = algo !== 'wordpiece';
    $('f-fallback').hidden = algo !== 'wordpiece';
    $('f-merges').hidden = algo !== 'bpe';
    $('algo-help').textContent = algo === 'bpe'
      ? 'Starts from single characters and keeps gluing together the pair of pieces that appears most often. Every glued pair becomes a new token. This is how GPT-style tokenizers are trained.'
      : 'Keeps the most frequent whole words, plus single-character pieces. A new word is matched against the longest piece the vocabulary knows. This is how BERT\'s WordPiece works, and what your class demo shows.';
  }

  function updateMaxWordsRange() {
    const tk = state.tk;
    if (!tk || tk.algorithm !== 'wordpiece') return;
    const input = $('maxwords');
    input.max = String(Math.max(1, tk.uniqueWords));
    $('maxwords-out').textContent = `${fmt(Math.min(Number(input.value), tk.uniqueWords))} of ${fmt(tk.uniqueWords)}`;
  }

  function corpusMeta() {
    const text = $('corpus').value;
    const lines = text.split('\n').filter((l) => l.trim()).length;
    const pre = T.preTokenize(text).length;
    $('corpus-meta').textContent = `${plural(lines, 'line')}, ${plural(text.length, 'character')}, ${plural(pre, 'word')} and punctuation marks`;
  }

  function presetButtons(containerId, list, onPick) {
    const box = $(containerId);
    box.innerHTML = list.map((p, i) => `<button type="button" data-i="${i}">${esc(p.name)}</button>`).join('');
    box.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      box.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
      onPick(list[Number(b.dataset.i)]);
    });
    return box;
  }

  function tabs(containerId, onPick) {
    const box = $(containerId);
    box.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
      onPick(b.dataset.view);
    });
  }

  function init() {
    const P = window.Presets;

    presetButtons('corpus-presets', P.corpora, (p) => { $('corpus').value = p.text; corpusMeta(); build(); });
    const pickTest = (p) => {
      $('test').value = p.text;
      $('hero-test').value = p.text;
      ['test-presets', 'hero-presets'].forEach((id) => $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x.textContent === p.name)));
      state.animate = true;
      encodeTest();
    };
    presetButtons('test-presets', P.tests, pickTest);
    presetButtons('hero-presets', P.tests, pickTest);

    $('corpus').value = P.corpora[0].text;
    $('test').value = P.tests[0].text;
    $('corpus-presets').querySelector('button').classList.add('on');
    $('test-presets').querySelector('button').classList.add('on');
    $('hero-presets').querySelector('button').classList.add('on');
    $('hero-test').value = $('test').value;

    document.querySelectorAll('#algo button').forEach((b) => b.addEventListener('click', () => { setAlgorithm(b.dataset.algo); build(); }));
    setAlgorithm('wordpiece');

    const liveBuild = debounce(build, 250);
    $('corpus').addEventListener('input', () => {
      corpusMeta();
      if ($('live').checked) liveBuild();
      else {
        state.stale = true;
        $('build-status').className = 'build-status stale';
        $('build-status').textContent = 'The corpus changed. Build the vocabulary again to use the new text.';
      }
    });

    $('maxwords').addEventListener('input', () => { updateMaxWordsRange(); liveBuild(); });
    $('merges').addEventListener('input', () => { $('merges-out').textContent = $('merges').value; liveBuild(); });
    $('merges-out').textContent = $('merges').value;
    ['lowercase', 'alphabet', 'fallback'].forEach((id) => $(id).addEventListener('change', build));
    $('build').addEventListener('click', build);

    const liveEncode = debounce(encodeTest, 120);
    const onTestInput = (src, dst) => () => {
      $(dst).value = $(src).value;
      ['test-presets', 'hero-presets'].forEach((id) => $(id).querySelectorAll('button').forEach((x) => x.classList.remove('on')));
      liveEncode();
    };
    $('test').addEventListener('input', onTestInput('test', 'hero-test'));
    $('hero-test').addEventListener('input', onTestInput('hero-test', 'test'));
    const tokenizeNow = () => { state.animate = true; encodeTest(); };
    $('test').addEventListener('keydown', (e) => { if (e.key === 'Enter') tokenizeNow(); });
    $('tokenize').addEventListener('click', tokenizeNow);

    tabs('token-tabs', (v) => { state.tokenView = v; renderTokens(); });
    tabs('vocab-tabs', (v) => {
      state.vocabTab = v;
      $('vocab-table-wrap').hidden = v !== 'table';
      $('merges-wrap').hidden = v !== 'merges';
    });

    const vRerender = () => { state.showAllRows = false; renderVocabTable(); };
    $('vsearch').addEventListener('input', debounce(vRerender, 100));
    $('vfilter').addEventListener('change', vRerender);
    $('vsort').addEventListener('change', vRerender);

    ['add-bos', 'add-eos'].forEach((id) => $(id).addEventListener('change', encodeTest));
    $('pad-to').addEventListener('input', debounce(encodeTest, 150));
    $('copy-ids').addEventListener('click', () => {
      const text = '[' + state.enc.ids.join(', ') + ']';
      const done = () => { $('copy-ids').textContent = 'Copied'; setTimeout(() => { $('copy-ids').textContent = 'Copy IDs'; }, 1400); };
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, () => {});
      else {
        const ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (_) { /* ignore */ }
        ta.remove();
      }
    });

    $('decode').addEventListener('click', runDecode);
    $('use-encoded').addEventListener('click', () => { $('decode-in').value = state.enc.ids.join(', '); runDecode(); });
    $('skip-special').addEventListener('change', runDecode);

    document.querySelectorAll('#chart-view button').forEach((b) => b.addEventListener('click', () => {
      state.chartView = b.dataset.view;
      document.querySelectorAll('#chart-view button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
      renderChart();
    }));
    document.querySelector('#chart-view button').setAttribute('aria-checked', 'true');
    $('topn').addEventListener('input', renderChart);
    $('emb-train').addEventListener('click', () => { emb.seed += 1; startEmbedding(); });
    $('emb-pause').addEventListener('click', () => {
      if (!emb.trainer || emb.trainer.finished) return;
      emb.paused = !emb.paused;
      $('emb-pause').textContent = emb.paused ? 'Resume' : 'Pause';
      if (!emb.paused) animateEmbedding();
    });
    ['emb-path', 'emb-punct'].forEach((id) => $(id).addEventListener('change', () => { updateEmbedView(); renderNeighbours(); }));
    $('c3-auto').addEventListener('change', () => emb.view && emb.view.setAuto($('c3-auto').checked));
    $('c3-zoom-in').addEventListener('click', () => emb.view && emb.view.zoomBy(1.15));
    $('c3-zoom-out').addEventListener('click', () => emb.view && emb.view.zoomBy(1 / 1.15));
    $('c3-reset').addEventListener('click', () => emb.view && emb.view.reset());
    ['chart-punct', 'chart-unused'].forEach((id) => $(id).addEventListener('change', renderChart));
    window.addEventListener('resize', debounce(() => state.tk && renderChart(), 150));

    setupTooltips();
    corpusMeta();
    build();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
