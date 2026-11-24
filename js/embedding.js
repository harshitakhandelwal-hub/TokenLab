/* =====================================================================
   TokenLab — token embeddings, trained in the browser
   ---------------------------------------------------------------------
   word2vec skip-gram with negative sampling (Mikolov et al., 2013):
     - every token gets a vector of DIM numbers, starting random
     - for each token in the training text, look at its neighbours
       within WINDOW positions
     - nudge the vectors so real neighbours score high and random
       tokens ("negative samples") score low
   Tokens that appear in similar places end up with similar vectors.

   For display, PCA squashes the DIM-dimensional vectors to 3D, the
   same trick people use to picture real LLM embeddings.
   ===================================================================== */
(function (root) {
  'use strict';

  const T = root.Tokenizer;

  /** Small seeded random generator, so training is repeatable. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class Trainer {
    /**
     * @param tk      a trained tokenizer (js/tokenizer.js)
     * @param corpus  training text
     * @param opts    { dim, window, negatives, seed }
     */
    constructor(tk, corpus, opts) {
      const o = opts || {};
      this.tk = tk;
      this.dim = o.dim || 16;
      this.window = o.window || 2;
      this.neg = o.negatives || 4;
      this.rng = mulberry32(o.seed || 7);
      const V = tk.vocab.size;
      this.V = V;

      // 1. Turn the corpus into ID sequences, one per line (windows don't cross lines).
      const seqs = String(corpus || '').split(/\n+/).map((line) => T.encode(tk, line, {}).ids).filter((s) => s.length > 1);
      this.counts = new Float64Array(V);
      seqs.forEach((s) => s.forEach((id) => { this.counts[id]++; }));
      this.used = [];
      for (let id = 0; id < V; id++) if (this.counts[id] > 0) this.used.push(id);

      // 2. Every (centre, context) pair inside the window.
      const centres = [];
      const contexts = [];
      for (const s of seqs) {
        for (let i = 0; i < s.length; i++) {
          for (let j = Math.max(0, i - this.window); j <= Math.min(s.length - 1, i + this.window); j++) {
            if (j !== i) { centres.push(s[i]); contexts.push(s[j]); }
          }
        }
      }
      this.centres = Int32Array.from(centres);
      this.contexts = Int32Array.from(contexts);
      this.pairsPerEpoch = centres.length;
      this.order = new Int32Array(this.pairsPerEpoch).map((_, i) => i);

      // 3. Negative samples are drawn in proportion to count^0.75.
      this.cum = new Float64Array(this.used.length);
      let acc = 0;
      this.used.forEach((id, i) => { acc += Math.pow(this.counts[id], 0.75); this.cum[i] = acc; });
      this.cumTotal = acc;

      // 4. Two weight tables: W = the embeddings, C = "context" vectors.
      this.W = new Float32Array(V * this.dim).map(() => (this.rng() - 0.5) / this.dim);
      this.C = new Float32Array(V * this.dim);
      this.neu = new Float32Array(this.dim);

      this.epochs = this.pairsPerEpoch ? Math.min(3000, Math.max(30, Math.ceil(160000 / this.pairsPerEpoch))) : 0;
      this.total = this.epochs * this.pairsPerEpoch;
      this.done = 0;
      this.epoch = 0;
      this.p = this.pairsPerEpoch; // forces a shuffle on the first step
      this.lr0 = 0.05;
      this.loss = 0;
      this.lossN = 0;
      this.lastLoss = null;
      this.prevAxes = null;
    }

    get finished() { return this.done >= this.total; }
    get progress() { return this.total ? this.done / this.total : 1; }

    sampleNegative() {
      const r = this.rng() * this.cumTotal;
      let lo = 0, hi = this.cum.length - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (this.cum[m] < r) lo = m + 1; else hi = m; }
      return this.used[lo];
    }

    shuffle() {
      const a = this.order;
      for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(this.rng() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
    }

    /** Train on the next n pairs. */
    step(n) {
      const d = this.dim, W = this.W, C = this.C, neu = this.neu;
      for (let k = 0; k < n && this.done < this.total; k++) {
        if (this.p >= this.pairsPerEpoch) { this.shuffle(); this.p = 0; this.epoch++; }
        const idx = this.order[this.p++];
        const c = this.centres[idx];
        const o = this.contexts[idx];
        const lr = Math.max(this.lr0 * 0.0001, this.lr0 * (1 - this.done / this.total));
        const wOff = c * d;
        neu.fill(0);
        for (let j = 0; j <= this.neg; j++) {
          const target = j === 0 ? o : this.sampleNegative();
          if (j > 0 && target === o) continue;
          const label = j === 0 ? 1 : 0;
          const cOff = target * d;
          let f = 0;
          for (let i = 0; i < d; i++) f += W[wOff + i] * C[cOff + i];
          const sig = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, f))));
          const g = (label - sig) * lr;
          this.loss += -Math.log(Math.max(1e-7, label ? sig : 1 - sig));
          for (let i = 0; i < d; i++) { neu[i] += g * C[cOff + i]; C[cOff + i] += g * W[wOff + i]; }
        }
        this.lossN++;
        for (let i = 0; i < d; i++) W[wOff + i] += neu[i];
        this.done++;
      }
      if (this.lossN) { this.lastLoss = this.loss / this.lossN; this.loss = 0; this.lossN = 0; }
    }

    /** Cosine similarity between two token embeddings. */
    similarity(a, b) {
      const d = this.dim, W = this.W;
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < d; i++) { const x = W[a * d + i], y = W[b * d + i]; dot += x * y; na += x * x; nb += y * y; }
      return dot / (Math.sqrt(na * nb) || 1);
    }

    /** The k tokens whose embeddings point the most in the same direction. */
    neighbours(id, k, allowed) {
      const pool = allowed || this.used;
      return pool.filter((j) => j !== id)
        .map((j) => ({ id: j, sim: this.similarity(id, j) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, k);
    }

    /**
     * PCA: project the embeddings of `ids` onto their 3 main directions.
     * Returns a Map id -> [x, y, z], scaled to fit inside -1..1.
     */
    project3(ids) {
      const d = this.dim, W = this.W, n = ids.length;
      const out = new Map();
      if (n < 2) { ids.forEach((id) => out.set(id, [0, 0, 0])); return out; }
      const mean = new Float64Array(d);
      ids.forEach((id) => { for (let i = 0; i < d; i++) mean[i] += W[id * d + i] / n; });
      // covariance matrix (d × d)
      const cov = new Float64Array(d * d);
      const row = new Float64Array(d);
      ids.forEach((id) => {
        for (let i = 0; i < d; i++) row[i] = W[id * d + i] - mean[i];
        for (let i = 0; i < d; i++) for (let j = i; j < d; j++) cov[i * d + j] += row[i] * row[j];
      });
      for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) cov[i * d + j] = cov[j * d + i];
      // top-3 eigenvectors by power iteration + deflation
      const axes = [];
      for (let a = 0; a < 3; a++) {
        let v = new Float64Array(d).map((_, i) => Math.sin(i * 12.9898 + a * 78.233) + 0.01);
        for (let it = 0; it < 60; it++) {
          const nv = new Float64Array(d);
          for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += cov[i * d + j] * v[j]; nv[i] = s; }
          const len = Math.hypot(...nv) || 1;
          v = nv.map((x) => x / len);
        }
        // keep the same orientation as last time so the picture doesn't flip
        if (this.prevAxes) { let s = 0; for (let i = 0; i < d; i++) s += v[i] * this.prevAxes[a][i]; if (s < 0) v = v.map((x) => -x); }
        let lambda = 0;
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += cov[i * d + j] * v[j]; lambda += v[i] * s; }
        for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i * d + j] -= lambda * v[i] * v[j];
        axes.push(v);
      }
      this.prevAxes = axes;
      let maxAbs = 1e-9;
      ids.forEach((id) => {
        const p = axes.map((ax) => { let s = 0; for (let i = 0; i < d; i++) s += (W[id * d + i] - mean[i]) * ax[i]; return s; });
        p.forEach((x) => { maxAbs = Math.max(maxAbs, Math.abs(x)); });
        out.set(id, p);
      });
      out.forEach((p, id) => out.set(id, p.map((x) => x / maxAbs)));
      return out;
    }
  }

  root.Embedding = { Trainer };
})(window);
