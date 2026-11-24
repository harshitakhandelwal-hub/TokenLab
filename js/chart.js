/* =====================================================================
   TokenLab — frequency charts (plain SVG, no libraries)
   ---------------------------------------------------------------------
   FreqChart.render(container, rows, options)
     rows:    [{ id, token, type, count, rank, share, inTest }]
     options: { view: 'compare' | 'rank', n, onHover(id|null), describe(row) }
   ===================================================================== */
(function (root) {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const TYPE_GROUP = { word: 'word', subword: 'piece', char: 'piece', punct: 'punct' };

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const clip = (s, n) => (Array.from(s).length > n ? Array.from(s).slice(0, n - 1).join('') + '…' : s);

  /* ---------- tooltip shared by both views ---------- */
  function tooltip(container) {
    let tip = container.querySelector('.chart-tip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'chart-tip';
      tip.setAttribute('role', 'status');
      tip.hidden = true;
      container.appendChild(tip);
    }
    return {
      show(html, x, y) {
        tip.innerHTML = html;
        tip.hidden = false;
        const cw = container.clientWidth;
        const tw = tip.offsetWidth;
        const left = Math.min(Math.max(8, x + 14), cw - tw - 8);
        tip.style.left = left + 'px';
        tip.style.top = Math.max(4, y - tip.offsetHeight - 12) + 'px';
      },
      hide() { tip.hidden = true; },
    };
  }

  function niceMax(v) {
    if (v <= 5) return Math.max(1, v);
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
    return v;
  }

  function linearTicks(max) {
    const step = max <= 5 ? 1 : max / 5;
    const out = [];
    for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
    return out;
  }

  function logTicks(max) {
    const out = [];
    for (let p = 1; p <= max * 1.0001; p *= 10) {
      for (const m of [1, 2, 5]) if (m * p <= max * 1.0001) out.push(m * p);
    }
    return out;
  }

  /* ---------- View 1: most used vs least used ---------- */
  function renderCompare(container, rows, opt) {
    const W = container.clientWidth || 720;
    const stacked = W < 640;
    const gap = 36;
    const panelW = stacked ? W : (W - gap) / 2;
    const n = Math.min(opt.n, rows.length);

    const top = rows.slice(0, n);
    const bottom = rows.slice().reverse().slice(0, n); // least used first
    const max = niceMax(Math.max(1, ...rows.map((r) => r.count)));

    const rowH = 24;
    const headH = 34;
    const axisH = 26;
    const panelH = headH + n * rowH + axisH;
    const H = stacked ? panelH * 2 + 20 : panelH;
    const gutter = 92; // token labels
    const valueRoom = 34;

    let svg = `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img"
      aria-label="Bar charts of the ${n} most used and ${n} least used tokens">`;

    const panel = (list, title, ox, oy) => {
      const plotW = panelW - gutter - valueRoom;
      const x = (v) => ox + gutter + (v / max) * plotW;
      let g = `<text class="ch-title" x="${ox}" y="${oy + 16}">${esc(title)}</text>`;
      // recessive grid + ticks
      linearTicks(max).forEach((t) => {
        const tx = x(t);
        g += `<line class="ch-grid" x1="${tx}" x2="${tx}" y1="${oy + headH - 6}" y2="${oy + headH + n * rowH}"/>`;
        g += `<text class="ch-tick" x="${tx}" y="${oy + headH + n * rowH + 16}" text-anchor="middle">${t}</text>`;
      });
      g += `<line class="ch-base" x1="${x(0)}" x2="${x(0)}" y1="${oy + headH - 6}" y2="${oy + headH + n * rowH}"/>`;
      list.forEach((r, i) => {
        const y = oy + headH + i * rowH;
        const bw = Math.max(0, x(r.count) - x(0));
        const bh = 14;
        const by = y + (rowH - bh) / 2;
        const grp = TYPE_GROUP[r.type] || 'word';
        // bar with 4px rounded data-end, square at the baseline
        const rr = Math.min(4, bw);
        const path = bw > 0
          ? `M${x(0)},${by} H${x(0) + bw - rr} Q${x(0) + bw},${by} ${x(0) + bw},${by + rr} V${by + bh - rr} Q${x(0) + bw},${by + bh} ${x(0) + bw - rr},${by + bh} H${x(0)} Z`
          : '';
        g += `<g class="ch-row" data-i="${rows.indexOf(r)}" tabindex="0">
          <rect class="ch-hit" x="${ox}" y="${y}" width="${panelW}" height="${rowH}"/>
          <text class="ch-label${r.inTest ? ' in-test' : ''}" x="${ox + gutter - 10}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(clip(r.token, 11))}</text>
          ${path ? `<path class="ch-bar g-${grp}" d="${path}"/>` : ''}
          <text class="ch-value" x="${x(r.count) + 6}" y="${y + rowH / 2 + 4}">${r.count}</text>
        </g>`;
      });
      return g;
    };

    svg += panel(top, `Most used (top ${n})`, 0, 0);
    svg += stacked
      ? panel(bottom, `Least used (bottom ${n})`, 0, panelH + 20)
      : panel(bottom, `Least used (bottom ${n})`, panelW + gap, 0);
    svg += '</svg>';
    return svg;
  }

  /* ---------- View 2: rank curve on log–log axes (Zipf) ---------- */
  function renderRank(container, rows) {
    const W = container.clientWidth || 720;
    const H = 340;
    const m = { l: 56, r: 20, t: 18, b: 46 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;
    const pts = rows.filter((r) => r.count > 0);
    const nMax = Math.max(2, pts.length) * 1.08;
    const cMax = Math.max(2, ...pts.map((r) => r.count)) * 1.35; // headroom above the top point
    const lx = (rank) => m.l + (Math.log(rank) / Math.log(nMax)) * pw;
    const ly = (c) => m.t + ph - (Math.log(c) / Math.log(cMax)) * ph;

    let svg = `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img"
      aria-label="Rank versus frequency of every used token on logarithmic axes">`;
    logTicks(nMax).forEach((t) => {
      svg += `<line class="ch-grid" x1="${lx(t)}" x2="${lx(t)}" y1="${m.t}" y2="${m.t + ph}"/>`;
      svg += `<text class="ch-tick" x="${lx(t)}" y="${m.t + ph + 16}" text-anchor="middle">${t}</text>`;
    });
    logTicks(cMax).forEach((t) => {
      svg += `<line class="ch-grid" x1="${m.l}" x2="${m.l + pw}" y1="${ly(t)}" y2="${ly(t)}"/>`;
      svg += `<text class="ch-tick" x="${m.l - 8}" y="${ly(t) + 4}" text-anchor="end">${t}</text>`;
    });
    svg += `<line class="ch-base" x1="${m.l}" x2="${m.l + pw}" y1="${m.t + ph}" y2="${m.t + ph}"/>`;
    svg += `<text class="ch-axis" x="${m.l + pw / 2}" y="${H - 6}" text-anchor="middle">Rank (1 = most used), log scale</text>`;
    svg += `<text class="ch-axis" transform="translate(14 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">Times used, log scale</text>`;

    if (pts.length) {
      const d = pts.map((r, i) => `${i ? 'L' : 'M'}${lx(i + 1).toFixed(1)},${ly(r.count).toFixed(1)}`).join(' ');
      svg += `<path class="ch-line" d="${d}"/>`;
      pts.forEach((r, i) => {
        const grp = TYPE_GROUP[r.type] || 'word';
        svg += `<circle class="ch-dot g-${grp}" cx="${lx(i + 1)}" cy="${ly(r.count)}" r="4"/>`;
      });
      // direct labels for the top 3 only
      pts.slice(0, 3).forEach((r, i) => {
        svg += `<text class="ch-direct" x="${lx(i + 1) + 8}" y="${ly(r.count) - 8}">${esc(clip(r.token, 10))}</text>`;
      });
      svg += `<line class="ch-cross" x1="0" x2="0" y1="${m.t}" y2="${m.t + ph}" visibility="hidden"/>`;
      svg += `<circle class="ch-focus" r="6" visibility="hidden"/>`;
      svg += `<rect class="ch-overlay" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}"/>`;
    } else {
      svg += `<text class="ch-axis" x="${W / 2}" y="${H / 2}" text-anchor="middle">No tokens used yet. Add some training text.</text>`;
    }
    svg += '</svg>';
    return { svg, pts, lx, ly, nMax };
  }

  /* ---------- public entry ---------- */
  function render(container, rows, opt) {
    const plot = container.querySelector('.chart-plot') || container;
    const tip = tooltip(container);
    tip.hide();
    const onHover = opt.onHover || (() => {});
    const describe = opt.describe || ((r) => esc(r.token));

    if (!rows.length) {
      plot.innerHTML = '<p class="chart-empty">Build a vocabulary to see how often each token is used.</p>';
      return;
    }

    const rel = (ev) => {
      const b = container.getBoundingClientRect();
      return [ev.clientX - b.left, ev.clientY - b.top];
    };

    if (opt.view === 'rank') {
      const r = renderRank(plot, rows);
      plot.innerHTML = r.svg;
      const svgEl = plot.querySelector('svg');
      const overlay = svgEl.querySelector('.ch-overlay');
      if (!overlay) return;
      const cross = svgEl.querySelector('.ch-cross');
      const focus = svgEl.querySelector('.ch-focus');
      overlay.addEventListener('mousemove', (ev) => {
        const pt = svgEl.createSVGPoint();
        pt.x = ev.clientX; pt.y = ev.clientY;
        const p = pt.matrixTransform(svgEl.getScreenCTM().inverse());
        // invert the log x scale to find the nearest rank
        const frac = (p.x - r.lx(1)) / (r.lx(r.nMax) - r.lx(1));
        let rank = Math.round(Math.exp(frac * Math.log(r.nMax)));
        rank = Math.min(r.pts.length, Math.max(1, rank));
        const row = r.pts[rank - 1];
        const cx = r.lx(rank);
        const cy = r.ly(row.count);
        cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
        focus.setAttribute('cx', cx); focus.setAttribute('cy', cy); focus.setAttribute('visibility', 'visible');
        const [mx, my] = rel(ev);
        tip.show(describe(row), mx, my);
        onHover(row.id);
      });
      overlay.addEventListener('mouseleave', () => {
        cross.setAttribute('visibility', 'hidden');
        focus.setAttribute('visibility', 'hidden');
        tip.hide();
        onHover(null);
      });
      return;
    }

    plot.innerHTML = renderCompare(plot, rows, opt);
    plot.querySelectorAll('.ch-row').forEach((g) => {
      const row = rows[Number(g.dataset.i)];
      const show = (x, y) => { g.classList.add('is-hot'); tip.show(describe(row), x, y); onHover(row.id); };
      const hide = () => { g.classList.remove('is-hot'); tip.hide(); onHover(null); };
      g.addEventListener('mousemove', (ev) => { const [x, y] = rel(ev); show(x, y); });
      g.addEventListener('mouseleave', hide);
      g.addEventListener('focus', () => {
        const b = g.getBoundingClientRect();
        const c = container.getBoundingClientRect();
        show(b.left - c.left + 120, b.top - c.top);
      });
      g.addEventListener('blur', hide);
    });
  }

  root.FreqChart = { render, TYPE_GROUP };
})(window);
