/* =====================================================================
   TokenLab — 3D charts (canvas 2D + a tiny hand-written 3D projection)
   ---------------------------------------------------------------------
   No libraries. Points are rotated (yaw, pitch), projected with simple
   perspective, and drawn back-to-front (painter's algorithm).

   Two modes, both from real tokenizer data:
     "scatter"  Token space: x = token ID, y = times used (log),
                z = token length in characters. One dot per token.
     "bars"     Frequency bars: one row per token kind (words, pieces,
                punctuation), bars sorted by use, height = times used.

   const c3 = new Chart3D(containerEl, { describe(row), onHover(id|null) });
   c3.setData(rows, { mode: 'scatter' | 'bars', n: 10 });
   ===================================================================== */
(function (root) {
  'use strict';

  const GROUP = { word: 'word', subword: 'piece', char: 'piece', punct: 'punct' };
  const GROUP_LABEL = { word: 'Whole words', piece: 'Pieces', punct: 'Punctuation' };
  // Theme colours, read from CSS custom properties on the container.
  let COLOR = { word: '#3987e5', piece: '#d95926', punct: '#199e70' };
  let INK = '#f3f0ff', INK2 = '#c4bdf0', INK3 = '#8d84c4', PROBE = '#ff6bc1';
  let FLOOR = '#221b52', WALL = '#1f1849', WALL2 = '#1c1645', GRID = 'rgba(196,189,240,0.12)', EDGE = 'rgba(196,189,240,0.35)', SURF = '#1c1645';
  function readTheme(el) {
    const cs = getComputedStyle(el);
    const v = (name, fb) => (cs.getPropertyValue(name).trim() || fb);
    COLOR = { word: v('--c-word', COLOR.word), piece: v('--c-piece', COLOR.piece), punct: v('--c-punct', COLOR.punct) };
    INK = v('--ink', INK); INK2 = v('--ink-2', INK2); INK3 = v('--ink-3', INK3); PROBE = v('--probe', PROBE);
    FLOOR = v('--c3-floor', FLOOR); WALL = v('--c3-wall', WALL); WALL2 = v('--c3-wall-2', WALL2);
    GRID = v('--c3-grid', GRID); EDGE = v('--c3-edge', EDGE); SURF = v('--panel', SURF);
  }
  const SANS = '"Instrument Sans", system-ui, sans-serif';
  const MONO = '"JetBrains Mono", ui-monospace, Menlo, monospace';

  const D = 7;      // camera distance from the scene centre
  const H = 1.3;    // scene height (y runs 0..H)
  const ZR = 0.75;  // scene depth half-size (z runs -ZR..ZR)
  const LIGHT = norm([-0.35, 0.8, -0.5]); // light direction in view space

  const reducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const tokenLength = (t) => Math.max(1, Array.from(t.replace(/^##/, '')).length);

  function norm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  /** Lighten (k > 0) or darken (k < 0) a hex colour. */
  function tone(hex, k) {
    const n = parseInt(hex.slice(1), 16);
    return 'rgb(' + [n >> 16 & 255, n >> 8 & 255, n & 255]
      .map((c) => Math.round(k < 0 ? c * (1 + k) : c + (255 - c) * k)).join(',') + ')';
  }

  function linearTicks(max) {
    if (max <= 5) { const top = Math.max(1, Math.ceil(max)); return { top, ticks: [...Array(top + 1).keys()] }; }
    const raw = max / 5;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const top = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = 0; v <= top + 1e-9; v += step) ticks.push(Math.round(v));
    return { top, ticks };
  }

  function logTicks(max) {
    const out = [0];
    for (let p = 1; p <= max; p *= 10) for (const m of [1, 2, 5]) if (m * p <= max) out.push(m * p);
    return out;
  }

  function inPoly(x, y, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i];
      const [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  class Chart3D {
    constructor(container, opts) {
      this.el = container;
      this.opts = opts || {};
      this.canvas = container.querySelector('canvas');
      this.ctx = this.canvas.getContext('2d');
      this.tip = document.createElement('div');
      this.tip.className = 'chart-tip';
      this.tip.hidden = true;
      container.appendChild(this.tip);

      this.rows = [];
      this.mode = 'scatter';
      this.n = 10;
      this.auto = !reducedMotion;
      this.hover = null;
      this.pointerInside = false;
      this.drag = null;
      this.raf = 0;
      this.hits = [];
      this.reset(false);
      this.bindEvents();

      if ('IntersectionObserver' in window) {
        new IntersectionObserver((entries) => {
          this.onScreen = entries[0].isIntersecting;
          if (this.onScreen) this.loop();
        }).observe(this.canvas);
      } else {
        this.onScreen = true;
      }
    }

    /* ---------- public ---------- */

    setData(rows, o) {
      readTheme(this.el);
      this.rows = rows;
      this.mode = o.mode || 'scatter';
      this.n = o.n || 10;
      this.path = o.path || null;
      // keep the hovered token while the data animates (e.g. during training)
      this.hover = (this.hover && rows.find((r) => r.id === this.hover.id)) || null;
      this.build();
      this.draw();
      this.loop();
    }

    reset(redraw = true) {
      this.yaw = -0.62;
      this.pitch = 0.42;
      this.zoom = 1;
      if (redraw) this.draw();
    }

    zoomBy(f) { this.zoom = clamp(this.zoom * f, 0.6, 2.4); this.draw(); }
    setAuto(on) { this.auto = on; this.loop(); }
    stop() { cancelAnimationFrame(this.raf); this.raf = 0; }

    /* ---------- scene building (world coordinates) ---------- */

    build() {
      const s = { points: [], bars: [], xTicks: [], yTicks: [], zTicks: [], labels: [] };
      this.scene = s;
      const rows = this.rows;
      if (!rows.length) return;

      if (this.mode === 'embed') {
        // positions come in already projected to -1..1 on every axis
        rows.forEach((r) => {
          s.points.push({ row: r, color: COLOR[GROUP[r.type]] || INK3, p: [r.pos[0], ((r.pos[1] + 1) / 2) * H, r.pos[2] * ZR] });
        });
        s.labels = s.points.filter((pt) => pt.row.label).map((pt) => ({ row: pt.row, p: pt.p }));
        [-1, -0.5, 0, 0.5, 1].forEach((v) => {
          s.xTicks.push({ x: v, label: '' });
          s.zTicks.push({ z: v * ZR, label: '' });
          s.yTicks.push({ y: ((v + 1) / 2) * H, label: '' });
        });
        s.titles = { x: 'Direction 1', y: 'Direction 2', z: 'Direction 3' };
        s.byId = new Map(s.points.map((pt) => [pt.row.id, pt]));
        return;
      }

      if (this.mode === 'bars') {
        const groups = ['word', 'piece', 'punct'].filter((g) => rows.some((r) => GROUP[r.type] === g && r.count > 0));
        const per = groups.map((g) => rows.filter((r) => GROUP[r.type] === g && r.count > 0).slice(0, this.n));
        const n = Math.max(1, ...per.map((p) => p.length));
        const { top, ticks } = linearTicks(Math.max(1, ...rows.map((r) => r.count)));
        const zRow = (i) => (groups.length === 1 ? 0 : -ZR + (2 * ZR * i) / (groups.length - 1));
        const slot = 2 / n;
        const bw = slot * 0.32;
        const bd = groups.length === 1 ? 0.3 : Math.min(0.24, ZR / (groups.length - 1) * 0.42);
        per.forEach((list, gi) => {
          list.forEach((r, k) => {
            const xc = -1 + (k + 0.5) * slot;
            const z = zRow(gi);
            s.bars.push({ row: r, k, color: COLOR[groups[gi]], x0: xc - bw, x1: xc + bw, z0: z - bd, z1: z + bd, h: (r.count / top) * H });
          });
        });
        const step = Math.ceil(n / 12);
        for (let k = 0; k < n; k += step) s.xTicks.push({ x: -1 + (k + 0.5) * slot, label: String(k + 1) });
        s.zTicks = groups.map((g, i) => ({ z: zRow(i), label: GROUP_LABEL[g] }));
        s.yTicks = ticks.map((t) => ({ y: (t / top) * H, label: String(t) }));
        s.titles = { x: 'Rank in its row (1 = most used)', y: 'Times used', z: '' };
        s.showFloorX = true;
        return;
      }

      // scatter: ID × frequency × length
      const ids = rows.map((r) => r.id);
      const minId = Math.min(...ids);
      const maxId = Math.max(...ids);
      const maxLen = Math.max(...rows.map((r) => tokenLength(r.token)));
      const maxCount = Math.max(1, ...rows.map((r) => r.count));
      const X = (id) => (maxId === minId ? 0 : -1 + (2 * (id - minId)) / (maxId - minId));
      const Z = (len) => (maxLen === 1 ? 0 : -ZR + (2 * ZR * (len - 1)) / (maxLen - 1));
      const Y = (c) => (Math.log1p(c) / Math.log1p(maxCount)) * H;

      rows.forEach((r) => {
        s.points.push({ row: r, color: COLOR[GROUP[r.type]] || INK3, p: [X(r.id), Y(r.count), Z(tokenLength(r.token))] });
      });
      rows.slice().sort((a, b) => b.count - a.count).slice(0, 5).forEach((r) => {
        s.labels.push({ row: r, p: [X(r.id), Y(r.count), Z(tokenLength(r.token))] });
      });
      linearTicks(maxId - minId).ticks.forEach((t) => {
        if (minId + t <= maxId) s.xTicks.push({ x: X(minId + t), label: String(minId + t) });
      });
      const zStep = Math.ceil(maxLen / 8);
      for (let L = 1; L <= maxLen; L += zStep) s.zTicks.push({ z: Z(L), label: String(L) });
      s.yTicks = logTicks(maxCount).map((t) => ({ y: Y(t), label: String(t) }));
      s.titles = { x: 'Token ID', y: 'Times used (log)', z: 'Length in characters' };
    }

    /* ---------- 3D math ---------- */

    /** Rotate a world point (scene centred on y = H/2) into view space. */
    rot(p) {
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const x = p[0], y = p[1] - H / 2, z = p[2];
      const x1 = x * cy - z * sy;
      const z1 = x * sy + z * cy;
      return [x1, y * cp + z1 * sp, -y * sp + z1 * cp];
    }

    rotVec(v) {
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const x1 = v[0] * cy - v[2] * sy;
      const z1 = v[0] * sy + v[2] * cy;
      return [x1, v[1] * cp + z1 * sp, -v[1] * sp + z1 * cp];
    }

    /** World point -> [screenX, screenY, depth, perspectiveScale]. */
    proj(p) {
      const r = this.rot(p);
      const k = D / (D + r[2]);
      return [this.cx + r[0] * this.S * k, this.cy - r[1] * this.S * k, r[2], k];
    }

    /* ---------- drawing ---------- */

    resize() {
      const w = this.canvas.clientWidth || 700;
      const h = this.canvas.clientHeight || 460;
      const dpr = window.devicePixelRatio || 1;
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
      }
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = w;
      this.h = h;
      if (this.opts.minimal) {
        // background "galaxy" in the hero: sits to the right on wide screens
        const wide = w > 900;
        this.cx = wide ? w * 0.75 : w * 0.5;
        this.cy = wide ? h * 0.3 : h * 0.17;
        this.S = Math.min(w * (wide ? 0.22 : 0.4), h * 0.3);
      } else {
        this.cx = w / 2;
        this.cy = h * 0.56;
        this.S = Math.min(w * 0.3, h * 0.44) * this.zoom;
      }
    }

    draw() {
      if (!this.el.offsetParent) return; // hidden
      this.resize();
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      this.hits = [];
      if (this.opts.minimal) { if (this.scene) this.drawScatter(); return; }
      if (!this.rows.length || !this.scene) {
        ctx.fillStyle = INK3;
        ctx.font = `14px ${SANS}`;
        ctx.textAlign = 'center';
        ctx.fillText('No tokens to show yet. Add some training text.', this.cx, this.h / 2);
        return;
      }
      this.drawFrame();
      if (this.mode === 'bars') this.drawBars();
      else this.drawScatter();
    }

    line(a, b, style, width) {
      const ctx = this.ctx;
      const p = this.proj(a);
      const q = this.proj(b);
      ctx.strokeStyle = style;
      ctx.lineWidth = width || 1;
      ctx.beginPath();
      ctx.moveTo(p[0], p[1]);
      ctx.lineTo(q[0], q[1]);
      ctx.stroke();
    }

    poly(pts, fill) {
      const ctx = this.ctx;
      ctx.beginPath();
      pts.forEach((p, i) => { const s = this.proj(p); i ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]); });
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }

    text(p, str, o) {
      const ctx = this.ctx;
      const s = this.proj(p);
      ctx.font = o.font || `11px ${SANS}`;
      ctx.fillStyle = o.color || INK3;
      let align = o.align || (s[0] < this.cx - 4 ? 'right' : s[0] > this.cx + 4 ? 'left' : 'center');
      const tw = ctx.measureText(str).width;
      let x = s[0];
      const left = align === 'right' ? x - tw : align === 'center' ? x - tw / 2 : x;
      if (left < 6) { x = 6; align = 'left'; }
      else if (left + tw > this.w - 6) { x = this.w - 6; align = 'right'; }
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.fillText(str, x, Math.max(8, Math.min(this.h - 8, s[1])));
    }

    /** Floor, the two walls farthest from the camera, grid lines, ticks and titles. */
    drawFrame() {
      const sc = this.scene;
      const depth = (p) => this.rot(p)[2];
      const xWall = depth([1, H / 2, 0]) > depth([-1, H / 2, 0]) ? 1 : -1;
      const zWall = depth([0, H / 2, ZR]) > depth([0, H / 2, -ZR]) ? ZR : -ZR;
      const xNear = -xWall;
      const zNear = -zWall;

      // surfaces
      this.poly([[-1, 0, -ZR], [1, 0, -ZR], [1, 0, ZR], [-1, 0, ZR]], FLOOR);
      this.poly([[-1, 0, zWall], [1, 0, zWall], [1, H, zWall], [-1, H, zWall]], WALL);
      this.poly([[xWall, 0, -ZR], [xWall, 0, ZR], [xWall, H, ZR], [xWall, H, -ZR]], WALL2);

      // grid
      sc.xTicks.forEach((t) => {
        this.line([t.x, 0, -ZR], [t.x, 0, ZR], GRID);
        this.line([t.x, 0, zWall], [t.x, H, zWall], GRID);
      });
      sc.zTicks.forEach((t) => {
        this.line([-1, 0, t.z], [1, 0, t.z], GRID);
        this.line([xWall, 0, t.z], [xWall, H, t.z], GRID);
      });
      sc.yTicks.forEach((t) => {
        this.line([-1, t.y, zWall], [1, t.y, zWall], GRID);
        this.line([xWall, t.y, -ZR], [xWall, t.y, ZR], GRID);
      });

      // edges
      this.line([-1, 0, zNear], [1, 0, zNear], EDGE);
      this.line([xNear, 0, -ZR], [xNear, 0, ZR], EDGE);
      this.line([xWall, 0, zNear], [xWall, H, zNear], EDGE);

      // tick labels
      const zOut = zNear + Math.sign(zNear) * 0.1;
      const xOut = xNear + Math.sign(xNear) * 0.1;
      sc.xTicks.forEach((t) => this.text([t.x, 0, zOut], t.label, { align: 'center' }));
      sc.zTicks.forEach((t) => this.text([xOut, 0, t.z], t.label, {
        font: this.mode === 'bars' ? `600 12px ${SANS}` : `11px ${SANS}`,
        color: this.mode === 'bars' ? INK2 : INK3,
      }));
      sc.yTicks.forEach((t) => this.text([xWall, t.y, zNear + Math.sign(zNear) * 0.06], t.label, {}));

      // axis titles
      const title = { font: `600 12px ${SANS}`, color: INK2 };
      this.text([0, 0, zNear + Math.sign(zNear) * 0.28], sc.titles.x, { ...title, align: 'center' });
      if (sc.titles.z) this.text([xNear + Math.sign(xNear) * 0.3, 0, 0], sc.titles.z, title);
      this.text([xWall, H + 0.12, zNear], sc.titles.y, title);
    }

    drawBars() {
      const ctx = this.ctx;
      const cam = [0, 0, -D];
      const bars = this.scene.bars.map((b) => {
        const c = this.rot([(b.x0 + b.x1) / 2, b.h / 2, (b.z0 + b.z1) / 2]);
        return { b, depth: c[2] };
      }).sort((a, b) => b.depth - a.depth);

      for (const { b } of bars) {
        const { x0, x1, z0, z1, h } = b;
        const faces = [
          { n: [0, 1, 0], c: [(x0 + x1) / 2, h, (z0 + z1) / 2], v: [[x0, h, z0], [x1, h, z0], [x1, h, z1], [x0, h, z1]] },
          { n: [0, 0, -1], c: [(x0 + x1) / 2, h / 2, z0], v: [[x0, 0, z0], [x1, 0, z0], [x1, h, z0], [x0, h, z0]] },
          { n: [0, 0, 1], c: [(x0 + x1) / 2, h / 2, z1], v: [[x0, 0, z1], [x1, 0, z1], [x1, h, z1], [x0, h, z1]] },
          { n: [-1, 0, 0], c: [x0, h / 2, (z0 + z1) / 2], v: [[x0, 0, z0], [x0, 0, z1], [x0, h, z1], [x0, h, z0]] },
          { n: [1, 0, 0], c: [x1, h / 2, (z0 + z1) / 2], v: [[x1, 0, z0], [x1, 0, z1], [x1, h, z1], [x1, h, z0]] },
        ];
        const hot = this.hover && this.hover.id === b.row.id;
        const polys = [];
        for (const f of faces) {
          const rn = this.rotVec(f.n);
          const rc = this.rot(f.c);
          const toCam = [cam[0] - rc[0], cam[1] - rc[1], cam[2] - rc[2]];
          if (dot(rn, toCam) <= 0) continue; // facing away
          const lit = Math.max(0, dot(rn, LIGHT));
          const pts = f.v.map((p) => this.proj(p));
          ctx.beginPath();
          pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
          ctx.closePath();
          ctx.fillStyle = tone(b.color, (lit - 0.62) * 0.75);
          ctx.fill();
          ctx.strokeStyle = hot ? INK : 'rgba(17,13,43,0.55)';
          ctx.lineWidth = hot ? 1.6 : 1;
          ctx.stroke();
          polys.push(pts.map((p) => [p[0], p[1]]));
        }
        this.hits.push({ row: b.row, polys });
      }

      // token labels on top of bars (all when few, otherwise the top 3 per row)
      const many = this.scene.bars.length > 36;
      this.scene.bars.forEach((b) => {
        if (many && b.k >= 3) return;
        const tok = b.row.token.length > 9 ? b.row.token.slice(0, 8) + '…' : b.row.token;
        this.text([(b.x0 + b.x1) / 2, b.h + 0.06, (b.z0 + b.z1) / 2], tok, {
          font: `${b.row.inTest ? 700 : 500} 11px ${MONO}`,
          color: b.row.inTest ? INK : INK2,
          align: 'center',
        });
      });
    }

    drawScatter() {
      const ctx = this.ctx;
      const minimal = !!this.opts.minimal;
      const pts = this.scene.points.map((pt) => ({ pt, s: this.proj(pt.p), f: this.proj([pt.p[0], 0, pt.p[2]]) }))
        .sort((a, b) => b.s[2] - a.s[2]);
      const zf = Math.sqrt(this.zoom);

      // your sentence as a path through embedding space, in reading order
      if (!minimal && this.path && this.scene.byId) {
        const stops = this.path.map((id) => this.scene.byId.get(id)).filter(Boolean).map((pt) => this.proj(pt.p));
        if (stops.length > 1) {
          ctx.save();
          ctx.strokeStyle = PROBE;
          ctx.globalAlpha = 0.75;
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 4]);
          ctx.beginPath();
          stops.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
          ctx.stroke();
          ctx.restore();
        }
      }

      for (const { pt, s, f } of pts) {
        const hot = this.hover && this.hover.id === pt.row.id;
        const inTest = pt.row.inTest;
        if (minimal) {
          // galaxy: faint stars, glowing stars for tokens in the test text
          const r = (inTest ? 5.5 : 2.6) * s[3];
          ctx.save();
          ctx.globalAlpha = inTest ? 1 : 0.55;
          if (inTest) { ctx.shadowColor = PROBE; ctx.shadowBlur = 16; }
          ctx.beginPath();
          ctx.arc(s[0], s[1], r, 0, Math.PI * 2);
          ctx.fillStyle = inTest ? PROBE : pt.color;
          ctx.fill();
          ctx.restore();
          continue;
        }
        // drop line to the floor helps judge height and depth
        ctx.strokeStyle = hot ? INK2 : GRID;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(s[0], s[1]);
        ctx.lineTo(f[0], f[1]);
        ctx.stroke();

        const r = 4.6 * s[3] * zf * (hot ? 1.4 : 1);
        if (inTest) {
          ctx.beginPath();
          ctx.arc(s[0], s[1], r + 3.5, 0, Math.PI * 2);
          ctx.strokeStyle = PROBE;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(s[0], s[1], r, 0, Math.PI * 2);
        ctx.fillStyle = pt.color;
        ctx.fill();
        ctx.strokeStyle = hot ? INK : SURF;
        ctx.lineWidth = hot ? 2 : 1.5;
        ctx.stroke();
        this.hits.push({ row: pt.row, circle: [s[0], s[1], r + 3] });
      }

      // hovering a token draws lines to its nearest neighbours in embedding space
      let near = [];
      if (!minimal && this.hover && this.opts.neighboursOf && this.scene.byId) {
        const from = this.scene.byId.get(this.hover.id);
        near = this.opts.neighboursOf(this.hover.id).map((id) => this.scene.byId.get(id)).filter(Boolean);
        if (from) {
          const a = this.proj(from.p);
          ctx.save();
          ctx.strokeStyle = INK;
          ctx.lineWidth = 1.5;
          near.forEach((pt) => { const b = this.proj(pt.p); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); });
          ctx.restore();
        }
      }

      // labels: the 5 most used tokens (chart), every test token (galaxy), plus neighbours on hover
      let labels = minimal
        ? (this.w > 900 ? this.scene.points.filter((pt) => pt.row.inTest) : [])
        : this.scene.labels;
      if (near.length) labels = labels.concat(near.map((pt) => ({ row: pt.row, p: pt.p })));
      if (this.hover && this.scene.byId && this.scene.byId.get(this.hover.id)) labels = labels.concat([this.scene.byId.get(this.hover.id)]);
      labels.forEach((l) => {
        const s = this.proj(l.p);
        ctx.font = `600 ${minimal ? 13 : 12}px ${MONO}`;
        ctx.fillStyle = minimal || l.row.inTest ? PROBE : INK;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(l.row.token, s[0] + 9, s[1] - 9);
      });
    }

    /* ---------- interaction ---------- */

    pick(clientX, clientY) {
      const r = this.canvas.getBoundingClientRect();
      const x = clientX - r.left;
      const y = clientY - r.top;
      let found = null;
      for (let i = this.hits.length - 1; i >= 0; i--) {
        const h = this.hits[i];
        if (h.circle ? Math.hypot(x - h.circle[0], y - h.circle[1]) <= h.circle[2] : h.polys.some((p) => inPoly(x, y, p))) {
          found = h.row;
          break;
        }
      }
      const changed = (found && found.id) !== (this.hover && this.hover.id);
      this.hover = found;
      if (changed) this.draw();
      if (found) {
        const c = this.el.getBoundingClientRect();
        this.showTip(this.opts.describe ? this.opts.describe(found) : found.token, clientX - c.left, clientY - c.top);
      } else {
        this.tip.hidden = true;
      }
      if (changed && this.opts.onHover) this.opts.onHover(found ? found.id : null);
    }

    showTip(html, x, y) {
      const tip = this.tip;
      tip.innerHTML = html;
      tip.hidden = false;
      const left = Math.min(Math.max(8, x + 14), this.el.clientWidth - tip.offsetWidth - 8);
      tip.style.left = left + 'px';
      tip.style.top = Math.max(4, y - tip.offsetHeight - 12) + 'px';
    }

    clearHover() {
      if (!this.hover) return;
      this.hover = null;
      this.tip.hidden = true;
      this.draw();
      if (this.opts.onHover) this.opts.onHover(null);
    }

    bindEvents() {
      const c = this.canvas;
      c.addEventListener('pointerdown', (e) => {
        this.drag = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch };
        c.setPointerCapture(e.pointerId);
        c.classList.add('grabbing');
        this.clearHover();
      });
      c.addEventListener('pointermove', (e) => {
        if (this.drag) {
          this.yaw = this.drag.yaw + (e.clientX - this.drag.x) * 0.008;
          this.pitch = clamp(this.drag.pitch + (e.clientY - this.drag.y) * 0.006, 0.05, 1.35);
          this.draw();
          return;
        }
        this.pick(e.clientX, e.clientY);
      });
      const end = () => { this.drag = null; c.classList.remove('grabbing'); this.loop(); };
      c.addEventListener('pointerup', end);
      c.addEventListener('pointercancel', end);
      c.addEventListener('pointerenter', () => { this.pointerInside = true; });
      c.addEventListener('pointerleave', () => { this.pointerInside = false; this.clearHover(); this.loop(); });
      // Scroll zooms only after the chart is clicked/focused, so page scrolling is never hijacked.
      c.addEventListener('wheel', (e) => {
        if (document.activeElement !== c) return;
        e.preventDefault();
        this.zoomBy(Math.exp(-e.deltaY * 0.0012));
      }, { passive: false });
      c.addEventListener('keydown', (e) => {
        const k = e.key;
        if (k === 'ArrowLeft') this.yaw -= 0.12;
        else if (k === 'ArrowRight') this.yaw += 0.12;
        else if (k === 'ArrowUp') this.pitch = clamp(this.pitch - 0.08, 0.05, 1.35);
        else if (k === 'ArrowDown') this.pitch = clamp(this.pitch + 0.08, 0.05, 1.35);
        else if (k === '+' || k === '=') this.zoom = clamp(this.zoom * 1.15, 0.6, 2.4);
        else if (k === '-') this.zoom = clamp(this.zoom / 1.15, 0.6, 2.4);
        else return;
        e.preventDefault();
        this.draw();
      });
    }

    /** Slow auto-rotation; pauses while hovering, dragging, hidden or off-screen. */
    loop() {
      if (this.raf) return;
      const step = () => {
        this.raf = 0;
        if (!this.auto || this.pointerInside || this.drag || !this.onScreen || !this.el.offsetParent) return;
        this.yaw += 0.0035;
        this.draw();
        this.raf = requestAnimationFrame(step);
      };
      this.raf = requestAnimationFrame(step);
    }
  }

  root.Chart3D = Chart3D;
})(window);
