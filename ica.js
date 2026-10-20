/* Unmixing School Sounds: a small FastICA project. Vanilla JS, no dependencies.
   The math part (top) also runs in Node for testing: node test/ica-test.js */

// ===================== Settings you can change =====================
// Students' own recordings (Bocuide School summer project), shown as cards with a player and a
// spectrogram strip. Entries with demo: true (exactly 3) can be used as the full demo's sources via
// the toggle; the demo uses the first 9 seconds of each. Older example format:
//   { label: 'Kitchen (Class 4B)', place: 'kitchen', src: 'audio/kids/kitchen-4b.mp3' },
//   { label: 'Hall (Class 4B)',    place: 'hall',    src: 'audio/kids/hall-4b.mp3' },
//   { label: 'Classroom (Class 4B)', place: 'classroom', src: 'audio/kids/classroom-4b.mp3' }
// place is one of: 'kitchen', 'hall', 'classroom'. Any audio format the browser can play works;
// each recording is cut or padded to 9 seconds. With 3 or more recordings, the full demo uses the
// first three of them instead of the examples. All recordings also show up in the visualizer's lists.
var CLASSROOM_CLIPS = [
  { label: 'Student recording 1', short: 'Rec 1', src: 'audio/students/student-1.mp3', strip: 'audio/students/student-1-spec.png', seconds: 8,  color: '#a78bfa' },
  { label: 'Student recording 2', short: 'Rec 2', src: 'audio/students/student-2.mp3', strip: 'audio/students/student-2-spec.png', seconds: 30, color: '#7ee081', demo: true },
  { label: 'Student recording 3', short: 'Rec 3', src: 'audio/students/student-3.mp3', strip: 'audio/students/student-3-spec.png', seconds: 30, color: '#6cb6ff', demo: true },
  { label: 'Student recording 4', short: 'Rec 4', src: 'audio/students/student-4.mp3', strip: 'audio/students/student-4-spec.png', seconds: 30, color: '#f0e36b' },
  { label: 'Student recording 5', short: 'Rec 5', src: 'audio/students/student-5.mp3', strip: 'audio/students/student-5-spec.png', seconds: 22, color: '#ff8a65' },
  { label: 'Student recording 6', short: 'Rec 6', src: 'audio/students/student-6.mp3', strip: 'audio/students/student-6-spec.png', seconds: 30, color: '#f48fb1', demo: true }
];

// Screenshots or photos of students' submitted work, shown as image cards in
// "Student recordings & submissions". When empty, placeholder cards are shown. Example entry:
//   { title: 'Sound map (Student A)', src: 'submissions/sound-map-a.png', caption: 'A short caption' }
var STUDENT_SUBMISSIONS = [
  { title: 'Student editing their recordings in a DAW', src: 'images/activity/daw-projector.jpg', caption: 'Students reviewing and editing their recordings.' }
];

// Example sounds: openly licensed recordings standing in until the classroom recordings are added.
// Full credits are in audio/credits.json.
var EXAMPLE_CLIPS = [
  { id: 'kitchen',   label: 'Kitchen',   place: 'kitchen',   src: 'audio/kitchen.mp3' },
  { id: 'hall',      label: 'Hall',      place: 'hall',      src: 'audio/hall.mp3' },
  { id: 'classroom', label: 'Classroom', place: 'classroom', src: 'audio/classroom.mp3' }
];

(function (root) {
  'use strict';

  var SR = 16000;          // sample rate (Hz)
  var DUR = 9;             // seconds per clip
  var N = SR * DUR;
  var GAUSS_LOGCOSH = 0.3745672;   // average of log(cosh(v)) for bell-curve (Gaussian) noise

  // ---------- small helpers ----------
  function rng(seed) {     // mulberry32, deterministic
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normalizePeak(x, peak) {
    var m = 0, i;
    for (i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
    var y = new Float32Array(x.length);
    if (m === 0) return y;
    var k = (peak || 0.9) / m;
    for (i = 0; i < x.length; i++) y[i] = x[i] * k;
    return y;
  }

  function fitLength(x, n) {   // cut or zero-pad to exactly n samples
    var y = new Float64Array(n);
    for (var i = 0; i < n && i < x.length; i++) y[i] = x[i];
    return y;
  }

  function logcosh(u) { var a = Math.abs(u); return a + Math.log1p(Math.exp(-2 * a)) - Math.LN2; }

  // How far a signal is from bell-curve noise (negentropy, logcosh version). Bigger = more "one clean sound".
  // y should have mean 0 and spread (std) 1.
  function nonGaussianity(y) {
    var s = 0;
    for (var t = 0; t < y.length; t++) s += logcosh(y[t]);
    var d = s / y.length - GAUSS_LOGCOSH;
    return d * d;
  }

  // ---------- mixing ----------
  function mix(A, S) {
    var n = S[0].length, X = [], i, j, t;
    for (i = 0; i < A.length; i++) {
      var x = new Float64Array(n);
      for (j = 0; j < S.length; j++) {
        var a = A[i][j];
        if (a === 0) continue;
        for (t = 0; t < n; t++) x[t] += a * S[j][t];
      }
      X.push(x);
    }
    return X;
  }

  function det(A) {
    if (A.length === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  }

  // ---------- small dense linear algebra ----------
  function matMul(A, B) {
    var n = A.length, m = B[0].length, k = B.length, C = [], i, j, l;
    for (i = 0; i < n; i++) {
      C.push(new Array(m).fill(0));
      for (j = 0; j < m; j++) for (l = 0; l < k; l++) C[i][j] += A[i][l] * B[l][j];
    }
    return C;
  }
  function transpose(A) {
    return A[0].map(function (_, j) { return A.map(function (row) { return row[j]; }); });
  }

  // Jacobi eigen-decomposition of a symmetric matrix: A = V diag(values) V^T (eigenvectors in columns of V).
  function jacobiEig(S) {
    var n = S.length, A = S.map(function (r) { return r.slice(); }), V = [], i, j, p, q, sweep;
    for (i = 0; i < n; i++) { V.push(new Array(n).fill(0)); V[i][i] = 1; }
    for (sweep = 0; sweep < 100; sweep++) {
      var off = 0;
      for (p = 0; p < n; p++) for (q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-20) break;
      for (p = 0; p < n; p++) {
        for (q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < 1e-30) continue;
          var theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
          var tt = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
          for (i = 0; i < n; i++) { var aip = A[i][p], aiq = A[i][q]; A[i][p] = c * aip - s * aiq; A[i][q] = s * aip + c * aiq; }
          for (j = 0; j < n; j++) { var apj = A[p][j], aqj = A[q][j]; A[p][j] = c * apj - s * aqj; A[q][j] = s * apj + c * aqj; }
          for (i = 0; i < n; i++) { var vip = V[i][p], viq = V[i][q]; V[i][p] = c * vip - s * viq; V[i][q] = s * vip + c * viq; }
        }
      }
    }
    var values = [];
    for (i = 0; i < n; i++) values.push(A[i][i]);
    return { values: values, vectors: V };
  }

  // Symmetric decorrelation: W <- (W W^T)^(-1/2) W  (keeps the rows at right angles to each other)
  function symDecorrelate(W) {
    var e = jacobiEig(matMul(W, transpose(W)));
    var n = W.length, M = [], i, j, k;
    for (i = 0; i < n; i++) {
      M.push(new Array(n).fill(0));
      for (j = 0; j < n; j++)
        for (k = 0; k < n; k++)
          M[i][j] += e.vectors[i][k] * e.vectors[j][k] / Math.sqrt(Math.max(e.values[k], 1e-12));
    }
    return matMul(M, W);
  }

  function applyMatrix(W, Z) {   // rows of W times signals Z
    var n = Z[0].length, Y = [], i, j, t;
    for (i = 0; i < W.length; i++) {
      var y = new Float64Array(n);
      for (j = 0; j < Z.length; j++) { var w = W[i][j]; for (t = 0; t < n; t++) y[t] += w * Z[j][t]; }
      Y.push(y);
    }
    return Y;
  }

  // Step 1: move to the middle (subtract each signal's average).
  function center(X) {
    var means = [], Xc = [];
    X.forEach(function (x) {
      var m = 0, t, n = x.length;
      for (t = 0; t < n; t++) m += x[t];
      m /= n;
      var xc = new Float64Array(n);
      for (t = 0; t < n; t++) xc[t] = x[t] - m;
      means.push(m); Xc.push(xc);
    });
    return { Xc: Xc, means: means };
  }

  // Step 2: make it round (whitening). C = E D E^T, K = D^(-1/2) E^T, Z = K Xc.
  function whiten(Xc) {
    var m = Xc.length, n = Xc[0].length, C = [], i, j, t;
    for (i = 0; i < m; i++) {
      C.push(new Array(m).fill(0));
      for (j = 0; j < m; j++) {
        var sum = 0;
        for (t = 0; t < n; t++) sum += Xc[i][t] * Xc[j][t];
        C[i][j] = sum / n;
      }
    }
    var eig = jacobiEig(C), K = [];
    for (i = 0; i < m; i++) {
      K.push(new Array(m).fill(0));
      var d = 1 / Math.sqrt(Math.max(eig.values[i], 1e-12));
      for (j = 0; j < m; j++) K[i][j] = d * eig.vectors[j][i];
    }
    return { Z: applyMatrix(K, Xc), K: K };
  }

  // ---------- FastICA (symmetric, logcosh / tanh) ----------
  // X: array of m signals. opts: { maxIter, tol, seed, W0, onRound(round, W, change) }.
  function fastICA(X, opts) {
    opts = opts || {};
    var maxIter = opts.maxIter || 200, tol = opts.tol || 1e-6;
    var c = center(X), wh = whiten(c.Xc), Z = wh.Z;
    var m = Z.length, n = Z[0].length, i, j, t, W;
    if (opts.W0) W = opts.W0.map(function (r) { return r.slice(); });
    else {
      var rand = rng(opts.seed == null ? 1 : opts.seed);
      W = [];
      for (i = 0; i < m; i++) { W.push([]); for (j = 0; j < m; j++) W[i].push(rand() * 2 - 1); }
    }
    W = symDecorrelate(W);
    if (opts.onRound) opts.onRound(0, W, null);

    var it, converged = false, wz = new Float64Array(n);
    for (it = 1; it <= maxIter; it++) {
      var Wn = [];
      for (i = 0; i < m; i++) {
        for (t = 0; t < n; t++) {                       // wz = this output, right now
          for (var v = 0, j = 0; j < m; j++) v += W[i][j] * Z[j][t];
          wz[t] = v;
        }
        var row = new Array(m).fill(0), gp = 0;
        for (t = 0; t < n; t++) {                       // push w toward a less bell-shaped output
          var g = Math.tanh(wz[t]);
          gp += 1 - g * g;
          for (j = 0; j < m; j++) row[j] += Z[j][t] * g;
        }
        for (j = 0; j < m; j++) row[j] = row[j] / n - (gp / n) * W[i][j];
        Wn.push(row);
      }
      Wn = symDecorrelate(Wn);                          // keep outputs at right angles
      var change = 0;                                   // how much did the dial move this round?
      for (i = 0; i < m; i++) {
        for (var dot = 0, j = 0; j < m; j++) dot += Wn[i][j] * W[i][j];
        change = Math.max(change, Math.abs(Math.abs(dot) - 1));
      }
      W = Wn;
      if (opts.onRound) opts.onRound(it, W, change);
      if (change < tol) { converged = true; break; }
    }
    return { Y: applyMatrix(W, Z), W: W, K: wh.K, Z: Z, means: c.means,
             iterations: Math.min(it, maxIter), converged: converged };
  }

  // ---------- matching recovered tracks to sources ----------
  function corr(a, b) {
    var n = a.length, ma = 0, mb = 0, t;
    for (t = 0; t < n; t++) { ma += a[t]; mb += b[t]; }
    ma /= n; mb /= n;
    var sab = 0, saa = 0, sbb = 0;
    for (t = 0; t < n; t++) { var da = a[t] - ma, db = b[t] - mb; sab += da * db; saa += da * da; sbb += db * db; }
    return (saa > 0 && sbb > 0) ? sab / Math.sqrt(saa * sbb) : 0;
  }

  // For each recovered track, the source it best matches (one-to-one, by |corr|).
  function matchComponents(Y, S) {
    var m = Y.length, R = [], i, j, best = null, bestScore = -1;
    for (i = 0; i < m; i++) { R.push([]); for (j = 0; j < S.length; j++) R[i].push(corr(Y[i], S[j])); }
    (function permute(arr, k) {
      if (k === arr.length) {
        var sc = 0;
        for (var q = 0; q < arr.length; q++) sc += Math.abs(R[q][arr[q]]);
        if (sc > bestScore) { bestScore = sc; best = arr.slice(); }
        return;
      }
      for (var p = k; p < arr.length; p++) {
        var tmp = arr[k]; arr[k] = arr[p]; arr[p] = tmp;
        permute(arr, k + 1);
        tmp = arr[k]; arr[k] = arr[p]; arr[p] = tmp;
      }
    })(S.map(function (_, idx) { return idx; }), 0);
    return Y.map(function (_, i) { return { source: best[i], corr: R[i][best[i]], abs: Math.abs(R[i][best[i]]) }; });
  }

  var DEFAULT_MIX = [[1.0, 0.45, 0.6], [0.5, 1.0, 0.35], [0.4, 0.55, 0.9]];   // the page's default 3x3 mix
  var DEFAULT_MIX2 = [[1.0, 0.6], [0.45, 0.9]];                               // the visualizer's default 2x2 mix

  var ICA = {
    SR: SR, N: N, DUR: DUR, rng: rng, mix: mix, det: det, matMul: matMul, transpose: transpose,
    jacobiEig: jacobiEig, symDecorrelate: symDecorrelate, applyMatrix: applyMatrix, center: center, whiten: whiten,
    fastICA: fastICA, corr: corr, matchComponents: matchComponents, normalizePeak: normalizePeak,
    fitLength: fitLength, logcosh: logcosh, nonGaussianity: nonGaussianity,
    DEFAULT_MIX: DEFAULT_MIX, DEFAULT_MIX2: DEFAULT_MIX2,
    EXAMPLE_CLIPS: EXAMPLE_CLIPS, CLASSROOM_CLIPS: CLASSROOM_CLIPS, STUDENT_SUBMISSIONS: STUDENT_SUBMISSIONS
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = ICA; return; }
  root.ICA = ICA;

  // ============================ Browser UI ============================
  // "Sound lab" look: every source keeps one signature color everywhere on the page.
  var COLORS = { kitchen: '#ffb347', hall: '#3fd0c9', classroom: '#ff6fae', mix: '#c9d1dc' };
  var PLOT_BG = '#080c13';
  var PLACE_NAMES = { kitchen: 'the kitchen', hall: 'the hall', classroom: 'a classroom' };
  var MICS = ['Mic by the door', 'Mic by the window', 'Mic at the back'];
  var reduceMotion = !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var params = root.location ? root.location.search : '';
  var clips = [];          // all loaded clips: { id, label, place, src, kind: 'example' | 'kids', x: Float64Array }
  var demoClips = [];      // the three used in the full demo
  var sources = null, A = DEFAULT_MIX.map(function (r) { return r.slice(); }), mixtures = null;
  var audioCtx = null, playing = null, playingBtn = null, live = null, lastOutputs = null;

  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // ---------- colors ----------
  function colorOf(clip) { return (clip && (clip.color || COLORS[clip.place])) || COLORS.mix; }
  function hexRgb(h) { var n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function blend(h1, h2, u) {
    var a = hexRgb(h1), b = hexRgb(h2);
    return '#' + [0, 1, 2].map(function (k) { var v = Math.round(a[k] + (b[k] - a[k]) * u); return (v < 16 ? '0' : '') + v.toString(16); }).join('');
  }
  function rgba(h, al) { var a = hexRgb(h); return 'rgba(' + a[0] + ',' + a[1] + ',' + a[2] + ',' + al + ')'; }
  function isHex(s) { return /^#[0-9a-f]{6}$/i.test(s); }

  // ---------- loading sounds ----------
  function resample(x, from, to) {
    if (from === to) return x;
    var n = Math.floor(x.length * to / from), y = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var p = i * from / to, k = Math.floor(p), f = p - k;
      y[i] = (x[k] || 0) * (1 - f) + (x[k + 1] || 0) * f;
    }
    return y;
  }
  function decode(buf) {
    var OAC = root.OfflineAudioContext || root.webkitOfflineAudioContext;
    function viaOffline() {
      return new Promise(function (res, rej) {
        var ctx = new OAC(1, N, SR);
        var p = ctx.decodeAudioData(buf.slice(0), res, rej);
        if (p && p.then) p.then(res, rej);
      });
    }
    function viaRealtime() {
      var AC = root.AudioContext || root.webkitAudioContext;
      var ctx = new AC();
      return new Promise(function (res, rej) {
        var p = ctx.decodeAudioData(buf.slice(0), res, rej);
        if (p && p.then) p.then(res, rej);
      });
    }
    return viaOffline().catch(viaRealtime).then(function (ab) {
      var ch = ab.getChannelData(0);
      if (ab.numberOfChannels > 1) {        // average to mono
        var mono = new Float32Array(ch.length), c, i;
        for (c = 0; c < ab.numberOfChannels; c++) { var d = ab.getChannelData(c); for (i = 0; i < d.length; i++) mono[i] += d[i] / ab.numberOfChannels; }
        ch = mono;
      }
      return fitLength(resample(ch, ab.sampleRate, SR), N);
    });
  }
  function loadClip(c) {
    return fetch(c.src).then(function (r) {
      if (!r.ok) throw new Error(c.src + ': HTTP ' + r.status);
      return r.arrayBuffer();
    }).then(decode).then(function (x) { c.x = x; return c; });
  }

  // ---------- drawing ----------
  function setupCanvas(canvas, hDefault) {
    var dpr = root.devicePixelRatio || 1;
    var w = canvas.clientWidth || 600, h = canvas.clientHeight || hDefault || 64;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    var c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c: c, w: w, h: h, dpr: dpr };
  }

  // Oscilloscope graticule: faint divisions, a brighter center line with small ticks.
  function graticule(c, x0, y0, w, h, nx, ny, both) {
    var i, x, y;
    c.save(); c.lineWidth = 1;
    c.strokeStyle = 'rgba(120,150,190,0.08)'; c.beginPath();
    for (i = 1; i < nx; i++) { x = Math.round(x0 + w * i / nx) + 0.5; c.moveTo(x, y0); c.lineTo(x, y0 + h); }
    for (i = 1; i < ny; i++) { y = Math.round(y0 + h * i / ny) + 0.5; c.moveTo(x0, y); c.lineTo(x0 + w, y); }
    c.stroke();
    c.strokeStyle = 'rgba(150,175,210,0.20)'; c.beginPath();
    var cy = Math.round(y0 + h / 2) + 0.5, cx = Math.round(x0 + w / 2) + 0.5, tk = w / (nx * 5);
    c.moveTo(x0, cy); c.lineTo(x0 + w, cy);
    for (x = x0; x <= x0 + w; x += tk) { c.moveTo(Math.round(x) + 0.5, cy - 2); c.lineTo(Math.round(x) + 0.5, cy + 2); }
    if (both) {
      var tv = h / (ny * 5);
      c.moveTo(cx, y0); c.lineTo(cx, y0 + h);
      for (y = y0; y <= y0 + h; y += tv) { c.moveTo(cx - 2, Math.round(y) + 0.5); c.lineTo(cx + 2, Math.round(y) + 0.5); }
    }
    c.stroke(); c.restore();
  }
  function glowStroke(c, color, width, blur) {
    c.save();
    c.strokeStyle = rgba(color, 0.22); c.lineWidth = (width || 1.4) * 3.2; c.lineJoin = 'round'; c.stroke();
    c.shadowColor = color; c.shadowBlur = blur == null ? 10 : blur;
    c.strokeStyle = color; c.lineWidth = width || 1.4; c.stroke();
    c.restore();
  }

  function drawWave(canvas, x, color) {
    color = isHex(color || '') ? color : COLORS.mix;
    var g = setupCanvas(canvas, 72), c = g.c, w = g.w, h = g.h;
    c.fillStyle = PLOT_BG; c.fillRect(0, 0, w, h);
    graticule(c, 0, 0, w, h, 18, 4);
    if (x) {
      var peak = 0, i;
      for (i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
      if (peak > 0) {
        var scale = (h / 2 - 4) / peak, per = x.length / w, hi = [], lo = [], px;
        for (px = 0; px < w; px++) {
          var a = Math.floor(px * per), b = Math.min(x.length, Math.floor((px + 1) * per));
          var l = Infinity, u = -Infinity;
          for (i = a; i < b; i++) { if (x[i] < l) l = x[i]; if (x[i] > u) u = x[i]; }
          if (l === Infinity) { l = 0; u = 0; }
          hi.push(h / 2 - u * scale); lo.push(h / 2 - l * scale + 0.6);
        }
        var grad = c.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, rgba(color, 0.55)); grad.addColorStop(0.5, rgba(color, 0.95)); grad.addColorStop(1, rgba(color, 0.55));
        c.save(); c.shadowColor = color; c.shadowBlur = 9; c.fillStyle = grad;
        c.beginPath(); c.moveTo(0, hi[0]);
        for (px = 1; px < w; px++) c.lineTo(px + 0.5, hi[px]);
        for (px = w - 1; px >= 0; px--) c.lineTo(px + 0.5, lo[px]);
        c.closePath(); c.fill(); c.restore();
      }
    }
    // keep a copy so the playhead can be drawn on top without redrawing the waveform
    var base = canvas._base || document.createElement('canvas');
    base.width = canvas.width; base.height = canvas.height;
    base.getContext('2d').drawImage(canvas, 0, 0);
    canvas._base = base; canvas._color = color; canvas._x = x;
  }
  function drawPlayhead(canvas, frac) {
    if (!canvas || !canvas._base) return;
    var c = canvas.getContext('2d'), dpr = canvas.width / (canvas.clientWidth || canvas.width);
    c.save(); c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, canvas.width, canvas.height);
    c.drawImage(canvas._base, 0, 0);
    if (frac != null) {
      var X = Math.max(0, Math.min(1, frac)) * canvas.width, col = canvas._color || COLORS.mix;
      c.fillStyle = 'rgba(8,12,19,0.45)'; c.fillRect(X, 0, canvas.width - X, canvas.height);   // not yet played = dimmer
      c.fillStyle = rgba(col, 0.10); c.fillRect(0, 0, X, canvas.height);
      c.shadowColor = '#ffffff'; c.shadowBlur = 10 * dpr;
      c.fillStyle = '#ffffff'; c.fillRect(Math.round(X) - dpr, 0, 2 * dpr, canvas.height);
    }
    c.restore();
  }

  // ---------- playback, with a live oscilloscope + spectrum ----------
  function stopLive() {
    if (!live) return;
    if (live.raf) cancelAnimationFrame(live.raf);
    if (live.el.parentNode) live.el.parentNode.removeChild(live.el);
    if (live.wave) drawPlayhead(live.wave, null);
    live = null;
  }
  function stopPlaying() {
    if (playing) { try { playing.stop(); } catch (e) {} playing = null; }
    if (playingBtn) { playingBtn.textContent = playingBtn.dataset.label || 'Play'; playingBtn.classList.remove('is-playing'); playingBtn = null; }
    stopLive();
  }
  function startLive(btn, an, dur) {
    var color = (root.getComputedStyle(btn).getPropertyValue('--c') || '').trim();
    if (!isHex(color)) color = COLORS.mix;
    var el = document.createElement('div');
    el.className = 'live'; el.style.setProperty('--c', color); el.setAttribute('aria-hidden', 'true');
    el.innerHTML = '<span class="live-tag">LIVE <span class="live-time">0.0 s</span></span>' +
      '<canvas class="live-scope"></canvas><canvas class="live-spec"></canvas>';
    var track = btn.closest('.track'), fig = btn.closest('figure'), ctl = btn.closest('.controls');
    if (track) track.appendChild(el);
    else if (fig) fig.appendChild(el);
    else if (ctl) ctl.parentNode.insertBefore(el, ctl.nextSibling);
    else btn.parentNode.appendChild(el);
    live = { el: el, an: an, color: color, dur: dur, t0: audioCtx.currentTime, wave: track ? track.querySelector('canvas.wave') : null,
             td: new Uint8Array(an.fftSize), fd: new Uint8Array(an.frequencyBinCount), time: el.querySelector('.live-time') };
    liveSize();
    live.raf = requestAnimationFrame(liveFrame);
  }
  function liveSize() {
    if (!live) return;
    live.sc = setupCanvas(live.el.querySelector('.live-scope'), 64);
    live.sp = setupCanvas(live.el.querySelector('.live-spec'), 64);
  }
  function liveFrame() {
    if (!live) return;
    var L = live, i, k, c, w, h;
    // oscilloscope (triggered on a rising zero crossing so the trace holds still)
    L.an.getByteTimeDomainData(L.td);
    c = L.sc.c; w = L.sc.w; h = L.sc.h;
    c.fillStyle = PLOT_BG; c.fillRect(0, 0, w, h);
    graticule(c, 0, 0, w, h, 10, 4);
    var start = 0, half = L.td.length >> 1;
    for (i = 1; i < half; i++) if (L.td[i - 1] < 128 && L.td[i] >= 128) { start = i; break; }
    var span = Math.min(1024, L.td.length - start), pk = 0;
    for (k = 0; k < span; k++) pk = Math.max(pk, Math.abs(L.td[start + k] - 128));
    var gain = Math.min(5, 0.85 / Math.max(pk / 128, 0.02));   // auto-scale, like a scope's auto mode
    c.beginPath();
    for (k = 0; k < span; k++) {
      var v = (L.td[start + k] - 128) / 128 * gain;
      v = Math.max(-1, Math.min(1, v));
      var X = k / (span - 1) * w, Y = h / 2 - v * (h / 2 - 3);
      if (k) c.lineTo(X, Y); else c.moveTo(X, Y);
    }
    glowStroke(c, L.color, 1.4, 8);
    // spectrum: 22 log-spaced bars, 50 Hz to 8 kHz
    L.an.getByteFrequencyData(L.fd);
    c = L.sp.c; w = L.sp.w; h = L.sp.h;
    c.fillStyle = PLOT_BG; c.fillRect(0, 0, w, h);
    var bars = 22, nyq = audioCtx.sampleRate / 2, bw = w / bars;
    c.save(); c.shadowColor = L.color; c.shadowBlur = 6;
    for (k = 0; k < bars; k++) {
      var f0 = 50 * Math.pow(160, k / bars), f1 = 50 * Math.pow(160, (k + 1) / bars);
      var b0 = Math.floor(f0 / nyq * L.fd.length), b1 = Math.max(b0 + 1, Math.floor(f1 / nyq * L.fd.length)), m = 0;
      for (i = b0; i < b1 && i < L.fd.length; i++) m = Math.max(m, L.fd[i]);
      var lv = Math.pow(m / 255, 1.6), bh = Math.max(1, (h - 4) * lv);
      c.fillStyle = rgba(L.color, 0.35 + 0.6 * lv);
      c.fillRect(k * bw + 1, h - bh, Math.max(1, bw - 2), bh);
    }
    c.restore();
    var t = audioCtx.currentTime - L.t0;
    if (L.wave) drawPlayhead(L.wave, t / L.dur);
    if (L.time) L.time.textContent = Math.min(t, L.dur).toFixed(1) + ' s / ' + L.dur.toFixed(1) + ' s';
    L.raf = requestAnimationFrame(liveFrame);
  }
  function play(x, btn) {
    if (btn && playingBtn === btn) { stopPlaying(); return; }   // second click = stop
    if (!audioCtx) {
      var AC = root.AudioContext || root.webkitAudioContext;
      if (!AC) { alert('Sorry, this browser cannot play sounds made on the page.'); return; }
      audioCtx = new AC();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
    stopPlaying();
    var y = normalizePeak(x, 0.8);
    var buf = audioCtx.createBuffer(1, y.length, SR);
    if (buf.copyToChannel) buf.copyToChannel(y, 0); else buf.getChannelData(0).set(y);
    var src = audioCtx.createBufferSource();
    var an = audioCtx.createAnalyser();
    an.fftSize = 2048; an.smoothingTimeConstant = 0.7; an.minDecibels = -95; an.maxDecibels = -12;
    src.buffer = buf; src.connect(an); an.connect(audioCtx.destination); src.start();
    playing = src;
    if (btn) {
      btn.dataset.label = btn.dataset.label || btn.textContent;
      btn.textContent = 'Stop'; btn.classList.add('is-playing'); playingBtn = btn;
      startLive(btn, an, y.length / SR);
    }
    src.onended = function () { if (playing === src) stopPlaying(); };
  }
  function playButton(btn, getSignal) {
    btn.addEventListener('click', function () { var x = getSignal(); if (x) play(x, btn); });
  }

  function trackRow(label, id, note, color) {
    var div = document.createElement('div');
    div.className = 'track';
    if (color) div.style.setProperty('--c', color);
    div.innerHTML =
      '<div class="track-head"><span class="track-label">' + esc(label) + '</span>' +
      '<button type="button" class="play" id="' + id + '-play">Play</button></div>' +
      '<canvas id="' + id + '-wave" class="wave" aria-label="Waveform: ' + esc(label) + '"></canvas>' +
      (note ? '<div class="match" id="' + id + '-note">' + note + '</div>' : '');
    return div;
  }

  // ===================== The ICA visualizer (2 sounds, 2 mics) =====================
  var V = {
    step: 0, A: DEFAULT_MIX2.map(function (r) { return r.slice(); }),
    OFFSET: [1.2, -0.8],          // a small steady offset per mic (in "spread" units) so centering is visible
    a: 0, b: 2, angle: 0, idx: null, S: null, X: null, Xc: null, Z: null, K: null,
    sub: null, frames: [], cur: null, anim: null, running: false, maxJ: 1, table: null
  };
  var STEPS = [
    { title: 'The sounds as dots',
      text: 'Each dot is one tiny moment of time. Left–right is how loud sound A is at that moment; up–down is sound B. Most of the time one sound is quiet while the other is busy, so the dots make a plus-shaped cloud, with its arms along the axes.' },
    { title: 'Mixed up',
      text: 'Now both mics hear a bit of both sounds. Left–right is mic 1, up–down is mic 2. The cross gets slanted and squashed. Move the sliders to change how loud each sound is at each mic.' },
    { title: 'Move to the middle',
      text: 'First the computer slides the whole cloud so its center sits on the middle of the plot.' },
    { title: 'Make it round',
      text: 'Next it stretches and squashes the cloud until it is spread out the same amount in every direction, so each microphone counts fairly. The cross is still there, but its two arms are now at a right angle.' },
    { title: 'Turn the dial',
      text: 'Now only one thing is left: which way to turn. Drag the dial. The two lines are the two outputs. When the lines run along the arms of the cross, the independence meter is highest and each output is one clean sound. Listen to the difference.' }
  ];
  var DOT_BINS = 9;

  function sampleIdx(n, k, seed) {   // k random moments (fixed seed so the picture is stable)
    var r = rng(seed), idx = new Uint32Array(k);
    for (var i = 0; i < k; i++) idx[i] = Math.floor(r() * n);
    return idx;
  }
  function std(x) { var m = 0, s = 0, t; for (t = 0; t < x.length; t++) m += x[t]; m /= x.length; for (t = 0; t < x.length; t++) s += (x[t] - m) * (x[t] - m); return Math.sqrt(s / x.length) || 1; }
  function rot(theta) { var c = Math.cos(theta), s = Math.sin(theta); return [[c, s], [-s, c]]; }

  function vizColors() {
    V.colA = colorOf(V.clipList[V.a]); V.colB = colorOf(V.clipList[V.b]);
    if (V.colB === V.colA) V.colB = blend(V.colA, '#ffffff', 0.55);   // two clips from the same place
    V.binColors = [];
    for (var b = 0; b < DOT_BINS; b++) V.binColors.push(blend(V.colA, V.colB, b / (DOT_BINS - 1)));
  }

  function vizCompute() {
    var sa = V.clipList[V.a].x, sb = V.clipList[V.b].x;
    var ka = 1 / std(sa), kb = 1 / std(sb), t;
    var s1 = new Float64Array(N), s2 = new Float64Array(N);
    for (t = 0; t < N; t++) { s1[t] = sa[t] * ka; s2[t] = sb[t] * kb; }    // each sound with spread 1
    V.S = [s1, s2];
    V.X = mix(V.A, V.S);
    for (var i = 0; i < 2; i++) for (t = 0; t < N; t++) V.X[i][t] += V.OFFSET[i];
    var c = center(V.X); V.Xc = c.Xc; V.means = c.means;
    var w = whiten(V.Xc); V.Z = w.Z; V.K = w.K;
    if (!V.idx) { V.idx = sampleIdx(N, 3000, 11); V.sub = sampleIdx(N, 16000, 23); }
    function pts(P) { var out = new Float32Array(V.idx.length * 2); for (var k = 0; k < V.idx.length; k++) { out[2 * k] = P[0][V.idx[k]]; out[2 * k + 1] = P[1][V.idx[k]]; } return out; }
    V.frames = [pts(V.S), pts(V.X), pts(V.Xc), pts(V.Z), pts(V.Z)];
    V.views = V.frames.map(function (f, k) { return viewFor(k === 1 ? [V.frames[1], V.frames[2]] : [f]); });
    V.views[2] = V.views[1];   // keep the same zoom while sliding, so the move is visible
    // color each dot by which sound is louder at that moment (blend of the two source colors)
    vizColors();
    V.binIdx = [];
    for (var b = 0; b < DOT_BINS; b++) V.binIdx.push([]);
    for (var k = 0; k < V.idx.length; k++) {
      var ea = Math.abs(s1[V.idx[k]]), eb = Math.abs(s2[V.idx[k]]);
      V.binIdx[Math.round(eb / (ea + eb || 1) * (DOT_BINS - 1))].push(k);
    }
    // independence meter for every angle (0.5 degree steps) on 16,000 moments
    var z1 = new Float64Array(V.sub.length), z2 = new Float64Array(V.sub.length);
    var r1 = new Float64Array(V.sub.length), r2 = new Float64Array(V.sub.length);
    for (t = 0; t < V.sub.length; t++) { z1[t] = V.Z[0][V.sub[t]]; z2[t] = V.Z[1][V.sub[t]]; r1[t] = s1[V.sub[t]]; r2[t] = s2[V.sub[t]]; }
    V.z1 = z1; V.z2 = z2; V.s1sub = r1; V.s2sub = r2;
    V.table = []; V.maxJ = 0;
    for (var d = 0; d <= 360; d++) { var j = meterAt(d / 2); V.table.push(j); if (j > V.maxJ) { V.maxJ = j; V.bestAngle = d / 2; } }
  }
  function viewFor(frames) {   // half-width of the plot so ~98% of dots fit (zoomed in so the busy middle is easy to see)
    var r = [];
    frames.forEach(function (f) { for (var k = 0; k < f.length; k += 2) r.push(Math.max(Math.abs(f[k]), Math.abs(f[k + 1]))); });
    r.sort(function (p, q) { return p - q; });
    return r[Math.floor(r.length * 0.98)] * 1.15 || 1;
  }
  function project(deg) {
    var th = deg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th), n = V.z1.length;
    var y1 = new Float64Array(n), y2 = new Float64Array(n);
    for (var t = 0; t < n; t++) { y1[t] = c * V.z1[t] + s * V.z2[t]; y2[t] = -s * V.z1[t] + c * V.z2[t]; }
    return [y1, y2];
  }
  function meterAt(deg) { var y = project(deg); return nonGaussianity(y[0]) + nonGaussianity(y[1]); }
  function outputsAt(deg) { return applyMatrix(rot(deg * Math.PI / 180), V.Z); }
  // Color of each output at this dial angle: the color of the sound it matches, blended while it is still a mix.
  function outColors(deg) {
    var y = project(deg);
    return y.map(function (yi) {
      var ra = Math.abs(corr(yi, V.s1sub)), rb = Math.abs(corr(yi, V.s2sub)), u = rb / (ra + rb || 1);
      u = Math.max(0, Math.min(1, (u - 0.5) * 1.6 + 0.5));
      return blend(V.colA, V.colB, u);
    });
  }

  function drawScatter(pts, view, angle) {
    var cv = $('viz-canvas'), g = setupCanvas(cv, 420), c = g.c, w = g.w, h = g.h;
    var half = Math.min(w, h) / 2 - 18, cx = w / 2, cy = h / 2, k = half / view, i, b;
    c.fillStyle = PLOT_BG; c.fillRect(0, 0, w, h);
    graticule(c, cx - half, cy - half, 2 * half, 2 * half, 8, 8, true);
    c.strokeStyle = 'rgba(150,175,210,0.28)'; c.lineWidth = 1;
    c.strokeRect(Math.round(cx - half) + 0.5, Math.round(cy - half) + 0.5, Math.round(2 * half), Math.round(2 * half));
    c.globalCompositeOperation = 'lighter';
    for (b = 0; b < DOT_BINS; b++) {
      var list = V.binIdx[b];
      c.fillStyle = rgba(V.binColors[b], 0.5);
      for (i = 0; i < list.length; i++) { var q = 2 * list[i]; c.fillRect(cx + pts[q] * k - 1, cy - pts[q + 1] * k - 1, 2, 2); }
    }
    c.globalCompositeOperation = 'source-over';
    var lbl = V.step === 0 ? ['Sound A', 'Sound B'] : (V.step >= 3 ? ['', ''] : ['Mic 1', 'Mic 2']);
    var lc = V.step === 0 ? [V.colA, V.colB] : [COLORS.mix, COLORS.mix];
    c.font = '600 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    c.textAlign = 'right'; c.fillStyle = lc[0]; c.fillText(lbl[0], cx + half - 4, cy - 7);
    c.textAlign = 'left'; c.fillStyle = lc[1]; c.fillText(lbl[1], cx + 7, cy - half + 14);
    if (angle != null) {
      var th = angle * Math.PI / 180, L = half + 6, oc = outColors(angle);
      [[Math.cos(th), Math.sin(th), '1'], [-Math.sin(th), Math.cos(th), '2']].forEach(function (d, n) {
        c.beginPath(); c.moveTo(cx - d[0] * L, cy + d[1] * L); c.lineTo(cx + d[0] * L, cy - d[1] * L);
        glowStroke(c, oc[n], 2, 14);
        c.save(); c.font = 'bold 13px system-ui, -apple-system, sans-serif'; c.textAlign = 'center';
        c.shadowColor = '#000'; c.shadowBlur = 6; c.fillStyle = oc[n];
        c.fillText('Output ' + d[2], cx + d[0] * (L - 26), cy - d[1] * (L - 26) - 6); c.restore();
      });
    }
  }

  function drawHist(canvas, y, color) {
    var g = setupCanvas(canvas, 74), c = g.c, w = g.w, h = g.h, bins = 41, lo = -4, hi = 4, cnt = new Array(bins).fill(0), i, mx = 0;
    for (i = 0; i < y.length; i++) { var b = Math.floor((y[i] - lo) / (hi - lo) * bins); if (b >= 0 && b < bins) cnt[b]++; }
    for (i = 0; i < bins; i++) mx = Math.max(mx, cnt[i]);
    c.fillStyle = PLOT_BG; c.fillRect(0, 0, w, h);
    graticule(c, 0, 0, w, h, 8, 3);
    var bw = w / bins, grad = c.createLinearGradient(0, h, 0, 0);
    grad.addColorStop(0, rgba(color, 0.25)); grad.addColorStop(1, rgba(color, 0.95));
    c.save(); c.shadowColor = color; c.shadowBlur = 8; c.fillStyle = grad;
    for (i = 0; i < bins; i++) { var bh = (h - 4) * cnt[i] / mx; c.fillRect(i * bw + 0.5, h - bh, Math.max(1, bw - 1.5), bh); }
    c.restore();
  }

  function vizBtnColors() {
    $('viz-pa').style.setProperty('--c', V.step === 0 ? V.colA : COLORS.mix);
    $('viz-pb').style.setProperty('--c', V.step === 0 ? V.colB : COLORS.mix);
  }
  function vizRender() {
    var s = V.step;
    vizBtnColors();
    if (V.anim) return;
    drawScatter(V.frames[s], V.views[s], s === 4 ? V.angle : null);
    if (s === 4) vizDial();
  }

  function vizDial() {
    var j = meterAt(V.angle), pct = Math.round(100 * j / V.maxJ);
    $('viz-angle-out').textContent = Math.round(V.angle) + '°';
    var meter = $('viz-meter-bar');
    meter.style.setProperty('--meter-w', (meter.parentNode.clientWidth || 300) + 'px');
    meter.style.width = Math.max(1, pct) + '%';
    $('viz-meter-out').textContent = pct + '% of the best';
    var y = project(V.angle), oc = outColors(V.angle);
    drawHist($('viz-hist1'), y[0], oc[0]); drawHist($('viz-hist2'), y[1], oc[1]);
    $('viz-hist1').parentNode.style.setProperty('--c', oc[0]);
    $('viz-hist2').parentNode.style.setProperty('--c', oc[1]);
    if ($('viz-strip-wrap').open) drawStrip();
  }

  function drawStrip() {
    var cv = $('viz-strip'), g = setupCanvas(cv, 64), c = g.c, w = g.w, h = g.h;
    c.fillStyle = PLOT_BG; c.fillRect(0, 0, w, h);
    graticule(c, 0, 0, w, h, 12, 3);
    c.beginPath();
    V.table.forEach(function (j, d) { var x = d / 360 * w, y = h - 4 - (h - 10) * j / V.maxJ; if (d) c.lineTo(x, y); else c.moveTo(x, y); });
    glowStroke(c, '#e6ebf2', 1.4, 8);
    var xa = V.angle / 180 * w;
    c.beginPath(); c.moveTo(xa, 0); c.lineTo(xa, h);
    glowStroke(c, '#ff5a5a', 1.5, 10);
  }

  function ease(u) { return u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; }
  function vizGo(step) {
    step = Math.max(0, Math.min(4, step));
    var from = V.step;
    V.step = step;
    stopPlaying();
    $('viz-title').textContent = 'Step ' + step + ' · ' + STEPS[step].title;
    $('viz-text').textContent = STEPS[step].text;
    $('viz-back').disabled = step === 0; $('viz-next').disabled = step === 4;
    [].forEach.call(document.querySelectorAll('.viz-steps button'), function (b, k) {
      b.setAttribute('aria-current', k === step ? 'step' : 'false');
    });
    $('viz-mixer').hidden = step !== 1;
    $('viz-dial').hidden = step !== 4;
    $('viz-play-row').hidden = step === 4;
    $('viz-pa').textContent = step === 0 ? 'Play sound A' : 'Play mic 1';
    $('viz-pb').textContent = step === 0 ? 'Play sound B' : 'Play mic 2';
    if (V.anim) { cancelAnimationFrame(V.anim); V.anim = null; }
    if (reduceMotion || Math.abs(step - from) !== 1 || step === 4 || from === 4) { vizRender(); return; }
    var P = V.frames[from], Q = V.frames[step], vp = V.views[from], vq = V.views[step];
    var tmp = new Float32Array(P.length), t0 = null, DURA = 900;
    vizBtnColors();
    function frame(ts) {
      if (t0 == null) t0 = ts;
      var u = ease(Math.min(1, (ts - t0) / DURA));
      for (var i = 0; i < P.length; i++) tmp[i] = P[i] + (Q[i] - P[i]) * u;
      drawScatter(tmp, vp + (vq - vp) * u, null);
      if (u < 1) V.anim = requestAnimationFrame(frame); else { V.anim = null; vizRender(); }
    }
    V.anim = requestAnimationFrame(frame);
  }

  function vizRebuild() {
    vizCompute();
    $('viz-done').textContent = ''; $('viz-rounds').textContent = ''; $('viz-round-counter').innerHTML = '';
    vizRender();
  }

  function vizMixer() {
    var box = $('viz-mixer'), html = '';
    var la = V.clipList[V.a].label, lb = V.clipList[V.b].label;
    vizColors();
    for (var i = 0; i < 2; i++) {
      html += '<div class="mixrow"><span class="mixmic">Mic ' + (i + 1) + ' hears</span>';
      for (var j = 0; j < 2; j++) {
        html += '<label class="slider" style="--c:' + (j ? V.colB : V.colA) + '">' + esc(j ? 'B: ' + lb : 'A: ' + la) +
          ' <input type="range" min="0" max="1" step="0.05" id="v' + i + j + '" value="' + V.A[i][j] + '">' +
          ' <output id="v' + i + j + '-out">' + V.A[i][j].toFixed(2) + '</output></label>';
      }
      html += '</div>';
    }
    html += '<p id="viz-warn" class="warn" role="status"></p>';
    box.innerHTML = html;
    for (i = 0; i < 2; i++) for (j = 0; j < 2; j++) (function (i, j) {
      $('v' + i + j).addEventListener('input', function () {
        V.A[i][j] = parseFloat(this.value);
        $('v' + i + j + '-out').textContent = V.A[i][j].toFixed(2);
        var bad = Math.abs(det(V.A)) < 0.05;
        $('viz-warn').textContent = bad ? 'Now both mics hear almost the same blend, so the sounds can’t be pulled apart. Move a slider a bit more.' : '';
        if (!bad) vizRebuild();
      });
    })(i, j);
  }

  function letComputerTurn() {
    if (V.running) return;
    V.running = true;
    $('viz-auto').disabled = true; $('viz-angle').disabled = true;
    $('viz-done').textContent = '';
    var hist = [];
    fastICA(V.X, { W0: rot(V.angle * Math.PI / 180), maxIter: 60, tol: 1e-6,
      onRound: function (r, W, change) { hist.push({ r: r, W: W.map(function (x) { return x.slice(); }), change: change }); } });
    var k = 1, last = hist[hist.length - 1];
    function angleOf(W) { var a = Math.atan2(W[0][1], W[0][0]) * 180 / Math.PI; if (a < 0) a += 180; if (a >= 180) a -= 180; return a; }
    function show(h) {
      var deg = Math.round(angleOf(h.W)), pct = Math.round(100 * meterAt(angleOf(h.W)) / V.maxJ);
      $('viz-rounds').textContent = 'Round ' + h.r + ' · dial at ' + deg + '° · meter ' +
        pct + '%' + (h.change != null ? ' · moved ' + h.change.toExponential(1) : '');
      $('viz-round-counter').innerHTML = '<span class="rc-label">ROUND</span><span class="rc-num tick">' +
        (h.r < 10 ? '0' : '') + h.r + '</span><span class="rc-label">' + deg + '° · ' + pct + '%</span>';
    }
    function finish() {
      V.running = false;
      $('viz-auto').disabled = false; $('viz-angle').disabled = false;
      $('viz-done').textContent = 'Finished after ' + last.r + ' round' + (last.r === 1 ? '' : 's') +
        ' (it stops when the dial barely moves). The computer did what you just did by hand, only faster.';
    }
    if (reduceMotion) {
      V.angle = angleOf(last.W); $('viz-angle').value = V.angle; show(last); vizRender(); finish(); return;
    }
    function nextRound() {
      if (k >= hist.length) { finish(); return; }
      var h = hist[k++], a0 = V.angle, a1 = angleOf(h.W);
      if (a1 - a0 > 90) a1 -= 180; else if (a0 - a1 > 90) a1 += 180;   // turn the short way
      var t0 = null;
      function frame(ts) {
        if (t0 == null) t0 = ts;
        var u = ease(Math.min(1, (ts - t0) / 450));
        V.angle = ((a0 + (a1 - a0) * u) % 180 + 180) % 180;
        $('viz-angle').value = V.angle;
        drawScatter(V.frames[4], V.views[4], V.angle); vizDial();
        if (u < 1) requestAnimationFrame(frame);
        else { show(h); setTimeout(nextRound, 500); }
      }
      requestAnimationFrame(frame);
    }
    show(hist[0]);
    setTimeout(nextRound, 400);
  }

  function vizInit() {
    V.clipList = clips;
    V.a = Math.max(0, clips.findIndex(function (c) { return c.id === 'kitchen'; }));
    V.b = Math.max(0, clips.findIndex(function (c) { return c.id === 'classroom'; }));
    var opts = clips.map(function (c, i) { return '<option value="' + i + '">' + esc(c.label) + '</option>'; }).join('');
    $('viz-sa').innerHTML = opts; $('viz-sb').innerHTML = opts;
    $('viz-sa').value = V.a; $('viz-sb').value = V.b;
    function pick() {
      var a = +$('viz-sa').value, b = +$('viz-sb').value;
      $('viz-pick-warn').textContent = a === b ? 'Pick two different sounds.' : '';
      if (a === b) return;
      V.a = a; V.b = b; vizMixer(); vizRebuild();
    }
    $('viz-sa').addEventListener('change', pick); $('viz-sb').addEventListener('change', pick);
    STEPS.forEach(function (s, k) {
      var b = document.createElement('button'); b.type = 'button'; b.textContent = k + ' · ' + s.title;
      b.addEventListener('click', function () { vizGo(k); });
      $('viz-steps').appendChild(b);
    });
    var rc = document.createElement('div');                 // big monospace round readout
    rc.className = 'round-counter'; rc.id = 'viz-round-counter'; rc.setAttribute('aria-hidden', 'true');
    $('viz-rounds').parentNode.insertBefore(rc, $('viz-rounds'));
    $('viz-back').addEventListener('click', function () { vizGo(V.step - 1); });
    $('viz-next').addEventListener('click', function () { vizGo(V.step + 1); });
    playButton($('viz-pa'), function () { return V.step === 0 ? V.S[0] : V.X[0]; });
    playButton($('viz-pb'), function () { return V.step === 0 ? V.S[1] : V.X[1]; });
    playButton($('viz-po1'), function () { return outputsAt(V.angle)[0]; });
    playButton($('viz-po2'), function () { return outputsAt(V.angle)[1]; });
    $('viz-angle').addEventListener('input', function () {
      V.angle = parseFloat(this.value); $('viz-done').textContent = ''; stopPlaying(); vizRender();
    });
    $('viz-auto').addEventListener('click', letComputerTurn);
    $('viz-strip-wrap').addEventListener('toggle', function () { if (this.open) drawStrip(); });
    vizMixer();
    vizCompute();
    var m = /[?&]vizstep=(\d)/.exec(params);
    vizGo(m ? +m[1] : 0);
    var ma = /[?&]vizangle=([\d.]+)/.exec(params);
    if (ma && V.step === 4) { V.angle = Math.min(180, +ma[1]); $('viz-angle').value = V.angle; vizRender(); }
    $('viz').classList.remove('loading');
  }

  // ===================== The full 3-sound demo =====================
  function buildMatrix() {
    var tbl = $('mix-matrix'), html = '<tr><th></th>', i, j;
    for (j = 0; j < 3; j++) html += '<th scope="col" style="--c:' + colorOf(demoClips[j]) + '">' + esc(demoClips[j].label) + '</th>';
    html += '</tr>';
    for (i = 0; i < 3; i++) {
      html += '<tr><th scope="row">' + MICS[i] + '</th>';
      for (j = 0; j < 3; j++)
        html += '<td style="--c:' + colorOf(demoClips[j]) + '"><input type="number" min="0" max="1" step="0.05" id="a' + i + j + '" value="' +
          A[i][j].toFixed(2) + '" aria-label="How loud ' + esc(demoClips[j].label) + ' is at the ' + MICS[i].toLowerCase() + '"></td>';
      html += '</tr>';
    }
    tbl.innerHTML = html;
    for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) (function (i, j) {
      $('a' + i + j).addEventListener('change', function () {
        var v = parseFloat(this.value);
        if (isNaN(v)) v = 0;
        v = Math.max(0, Math.min(1, v));
        this.value = v.toFixed(2);
        A[i][j] = v;
        updateMixtures();
      });
    })(i, j);
  }

  function updateMixtures() {
    stopPlaying();
    mixtures = mix(A, sources);
    for (var i = 0; i < 3; i++) drawWave($('mic' + i + '-wave'), mixtures[i], COLORS.mix);
    $('mix-warn').textContent = Math.abs(det(A)) < 1e-3
      ? 'With these numbers, the mics hear (almost) the same blend, so the sounds can’t be pulled apart. Try changing a number.' : '';
    $('ica-out').innerHTML = '<p class="muted">Press “Separate” first.</p>';
    $('ica-status').textContent = '';
    lastOutputs = null;
    if ($('lab-panel')) { labReset(); tryReset(); }
  }

  var separations = 0;
  // ===================== Analysis lab =====================
  var SUB = null;                       // fixed subsample of moments for quick meters
  function subIdx() { if (!SUB) { SUB = []; for (var t = 0; t < N; t += 18) SUB.push(t); } return SUB; }
  function scoreOf(Y) {                 // sum of non-Gaussianity over outputs, on the subsample
    var idx = subIdx(), s = 0;
    Y.forEach(function (y) { var v = new Float64Array(idx.length); for (var k = 0; k < idx.length; k++) v[k] = y[idx[k]]; s += nonGaussianity(v); });
    return s;
  }
  function lerpW(A0, B0, u) { return A0.map(function (r, i) { return r.map(function (v, j) { return v + (B0[i][j] - v) * u; }); }); }

  // ---- FFT + spectrogram ----
  var FFTN = 512, HOP = 512, FBINS = 128;   // 128 bins of 31.25 Hz = 0-4 kHz
  var HANN = (function () { var w = new Float64Array(FFTN); for (var i = 0; i < FFTN; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FFTN - 1)); return w; })();
  function fft(re, im) {
    var n = re.length, i, j, k, len;
    for (i = 1, j = 0; i < n; i++) { var bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { var t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (len = 2; len <= n; len <<= 1) {
      var ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (i = 0; i < n; i += len) {
        var cr = 1, ci = 0;
        for (k = 0; k < len / 2; k++) {
          var ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci, ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k + len / 2] = re[i + k] - ar; im[i + k + len / 2] = im[i + k] - ai; re[i + k] += ar; im[i + k] += ai;
          var nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
  }
  function spectrogram(x) {
    var frames = Math.floor((x.length - FFTN) / HOP) + 1, out = new Float32Array(frames * FBINS), mx = -Infinity;
    var re = new Float64Array(FFTN), im = new Float64Array(FFTN);
    for (var f = 0; f < frames; f++) {
      for (var i = 0; i < FFTN; i++) { re[i] = x[f * HOP + i] * HANN[i]; im[i] = 0; }
      fft(re, im);
      for (var b = 0; b < FBINS; b++) { var db = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-12); out[f * FBINS + b] = db; if (db > mx) mx = db; }
    }
    return { d: out, frames: frames, max: mx };
  }
  function specImage(sp, color) {
    var cv = document.createElement('canvas'); cv.width = sp.frames; cv.height = FBINS;
    var c = cv.getContext('2d'), img = c.createImageData(sp.frames, FBINS), rgb = hexRgb(color);
    for (var f = 0; f < sp.frames; f++) for (var b = 0; b < FBINS; b++) {
      var v = Math.max(0, Math.min(1, (sp.d[f * FBINS + b] - sp.max + 60) / 60)); v = v * v;
      var o = ((FBINS - 1 - b) * sp.frames + f) * 4;
      img.data[o] = 8 + rgb[0] * v; img.data[o + 1] = 12 + rgb[1] * v; img.data[o + 2] = 19 + rgb[2] * v; img.data[o + 3] = 255;
    }
    c.putImageData(img, 0, 0); return cv;
  }
  function drawImg(canvas, img, h) { var g = setupCanvas(canvas, h); g.c.imageSmoothingEnabled = true; g.c.drawImage(img, 0, 0, g.w, g.h); return g; }

  // ---- 3 · You try first ----
  var T = { Z: null, best: 0, ref: 1, outs: null, raf: null };
  function rot3(a) {   // turn by a[0] in plane 1-2, a[1] in plane 1-3, a[2] in plane 2-3
    function R(p, q, th) { var M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], c = Math.cos(th), s = Math.sin(th); M[p][p] = c; M[q][q] = c; M[p][q] = s; M[q][p] = -s; return M; }
    var d = Math.PI / 180;
    return matMul(R(1, 2, a[2] * d), matMul(R(0, 2, a[1] * d), R(0, 1, a[0] * d)));
  }
  function tryReset() {
    T.Z = whiten(center(mixtures).Xc).Z; T.best = 0;
    var idx = subIdx(), Zs = T.Z.map(function (z) { var v = new Float64Array(idx.length); for (var k = 0; k < idx.length; k++) v[k] = z[idx[k]]; return v; });
    var r = fastICA(Zs, { seed: 1 });     // meter scale only: the best score the computer finds on the same moments
    T.ref = scoreOf(r.Y.map(function (y) { var f = new Float64Array(N); for (var k = 0; k < idx.length; k++) f[idx[k]] = y[k]; return f; })) || 1;
    // (scoreOf reads only the subsample, so the zeros in between don't matter)
    for (var i = 0; i < 3; i++) { $('try-a' + i).value = 0; }
    tryUpdate();
  }
  function tryUpdate() {
    if (!T.Z) return;
    var a = [0, 1, 2].map(function (i) { var v = +$('try-a' + i).value; $('try-a' + i + '-out').textContent = v + '°'; return v; });
    T.outs = applyMatrix(rot3(a), T.Z);
    var sc = scoreOf(T.outs);
    if (sc > T.best) T.best = sc;
    $('try-meter-out').textContent = sc.toFixed(4);
    $('try-meter-bar').style.width = Math.min(100, 100 * sc / T.ref).toFixed(1) + '%';
    $('try-best').textContent = 'Your best so far: ' + T.best.toFixed(4) + (L.res ? ' · Computer: ' + L.final.toFixed(4) : '');
    for (var i = 0; i < 3; i++) drawWave($('try' + i + '-wave'), T.outs[i], COLORS.mix);
    if (L.res) $('lab-vs').textContent = 'Your best score: ' + T.best.toFixed(4) + ' · Computer’s final score: ' + L.final.toFixed(4);
  }
  function tryInit() {
    var box = $('try-out'); box.innerHTML = '';
    for (var i = 0; i < 3; i++) (function (i) {
      box.appendChild(trackRow('My output ' + (i + 1), 'try' + i, null, COLORS.mix));
      playButton($('try' + i + '-play'), function () { return T.outs && T.outs[i]; });
    })(i);
    [0, 1, 2].forEach(function (i) {
      $('try-a' + i).addEventListener('input', function () { if (T.raf) return; T.raf = requestAnimationFrame(function () { T.raf = null; tryUpdate(); }); });
    });
  }

  // ---- 4 · round-by-round playback ----
  var L = { res: null, Ws: [], Y: [], scores: [], match: [], R: [], pos: 0, playing: false, raf: null, cmp: 0, specs: [], micSpec: null };
  function labOutputs(pos) {
    var k = Math.floor(pos), u = pos - k;
    var W = u > 0 && k + 1 < L.Ws.length ? lerpW(L.Ws[k], L.Ws[k + 1], u) : L.Ws[Math.min(k, L.Ws.length - 1)];
    return applyMatrix(W, L.res.Z);
  }
  function labColor(i, pos) {   // fades from mix-gray to the matched sound's color as the match grows
    var k = Math.round(pos), m = L.match[k][i];
    return blend(COLORS.mix, colorOf(demoClips[m.source]), Math.max(0, Math.min(1, (m.abs - 0.5) / 0.5)));
  }
  function labDrawChart(pos) {
    var g = setupCanvas($('lab-chart'), 120), c = g.c, w = g.w, h = g.h, n = L.scores.length;
    c.fillStyle = PLOT_BG; c.fillRect(0, 0, w, h); graticule(c, 0, 0, w, h, Math.max(2, n - 1), 4);
    var mx = Math.max.apply(null, L.scores) * 1.1 || 1, px = function (r) { return 8 + (w - 16) * r / Math.max(1, n - 1); }, py = function (v) { return h - 8 - (h - 16) * v / mx; };
    c.save(); c.strokeStyle = COLORS.hall; c.lineWidth = 2; c.shadowColor = COLORS.hall; c.shadowBlur = 8; c.beginPath();
    var k = Math.floor(pos), u = pos - k;
    for (var r = 0; r <= k; r++) { if (r) c.lineTo(px(r), py(L.scores[r])); else c.moveTo(px(0), py(L.scores[0])); }
    if (u > 0 && k + 1 < n) c.lineTo(px(k + u), py(L.scores[k] + (L.scores[k + 1] - L.scores[k]) * u));
    c.stroke(); c.restore();
    c.fillStyle = '#fff';
    for (r = 0; r <= k; r++) { c.beginPath(); c.arc(px(r), py(L.scores[r]), 2.5, 0, 7); c.fill(); }
  }
  function labBars(i, pos) {
    var k = Math.floor(pos), u = pos - k, k2 = Math.min(k + 1, L.R.length - 1), html = '';
    for (var j = 0; j < 3; j++) {
      var v = Math.abs(L.R[k][i][j]) + (Math.abs(L.R[k2][i][j]) - Math.abs(L.R[k][i][j])) * u;
      html += '<div class="mbar"><span>' + esc(demoClips[j].short || demoClips[j].label) + '</span><div class="mtrack"><div style="width:' + (100 * v).toFixed(1) + '%;background:' + colorOf(demoClips[j]) + '"></div></div><span>' + (u ? '' : Math.abs(L.R[k][i][j]).toFixed(2)) + '</span></div>';
    }
    $('lab' + i + '-note').innerHTML = html;
  }
  function labSpecs() {
    var k = Math.round(L.pos), Y = L.Y[k] || (L.Y[k] = labOutputs(k)), box = $('lab-specs');
    if (!L.micSpec) L.micSpec = specImage(spectrogram(mixtures[0]), COLORS.mix);
    L.specs = Y.map(function (y, i) { return specImage(spectrogram(y), labColor(i, k)); });
    if (!box.children.length) {
      box.innerHTML = ['Mic 1 (mixed)', 'Output 1', 'Output 2', 'Output 3'].map(function (t, i) { return '<figure><canvas class="spec" id="spec' + i + '"></canvas><figcaption>' + t + '</figcaption></figure>'; }).join('');
    }
    drawImg($('spec0'), L.micSpec, 70);
    for (var i = 0; i < 3; i++) drawImg($('spec' + (i + 1)), L.specs[i], 70);
    labCompare();
  }
  function labCompare() {
    if (!L.specs.length) return;
    var cv = $('cmp-canvas'), g = drawImg(cv, L.micSpec, 140), f = +$('cmp-slider').value / 100, x = g.w * f, img = L.specs[L.cmp];
    g.c.save(); g.c.beginPath(); g.c.rect(x, 0, g.w - x, g.h); g.c.clip(); g.c.drawImage(img, 0, 0, g.w, g.h); g.c.restore();
    g.c.fillStyle = '#fff'; g.c.fillRect(x - 1, 0, 2, g.h);
    $('cmp-label').textContent = 'Output ' + (L.cmp + 1) + ' (round ' + Math.round(L.pos) + ')';
  }
  function labRender(full) {
    var pos = L.pos, k = Math.round(pos), exact = Math.abs(pos - k) < 1e-9;
    var Y = exact ? (L.Y[k] || (L.Y[k] = labOutputs(k))) : labOutputs(pos);
    for (var i = 0; i < 3; i++) {
      var col = labColor(i, pos);
      $('lab' + i).style.setProperty('--c', col);
      drawWave($('lab' + i + '-wave'), Y[i], col);
      labBars(i, pos);
    }
    labDrawChart(pos);
    $('lab-scrub').value = k;
    $('lab-round-out').textContent = 'Round ' + k + ' of ' + (L.Ws.length - 1) + (k === 0 ? ' (start)' : '');
    $('lab-score-out').textContent = exact ? L.scores[k].toFixed(4) : '';
    if (full) labSpecs();
  }
  function labGo(k) {
    k = Math.max(0, Math.min(L.Ws.length - 1, k));
    cancelAnimationFrame(L.raf);
    var from = L.pos;
    if (reduceMotion || from === k) { L.pos = k; labRender(true); return; }
    var t0 = null, dur = 650;
    function step(ts) {
      if (t0 === null) t0 = ts;
      var u = Math.min(1, (ts - t0) / dur);
      L.pos = from + (k - from) * ease(u);
      if (u < 1) { labRender(false); L.raf = requestAnimationFrame(step); }
      else { L.pos = k; labRender(true); if (L.playing) { if (k < L.Ws.length - 1) setTimeout(function () { if (L.playing) labGo(k + 1); }, 250); else labStop(); } }
    }
    L.raf = requestAnimationFrame(step);
  }
  function labStop() { L.playing = false; $('lab-play').textContent = 'Play rounds'; }
  function labInit() {
    $('lab-prev').addEventListener('click', function () { labStop(); labGo(Math.round(L.pos) - 1); });
    $('lab-next').addEventListener('click', function () { labStop(); labGo(Math.round(L.pos) + 1); });
    $('lab-scrub').addEventListener('input', function () { labStop(); cancelAnimationFrame(L.raf); L.pos = +this.value; labRender(false); });
    $('lab-scrub').addEventListener('change', function () { labRender(true); });
    $('lab-play').addEventListener('click', function () {
      if (L.playing) { labStop(); return; }
      L.playing = true; this.textContent = 'Pause';
      if (reduceMotion) { labStop(); labGo(L.Ws.length - 1); return; }
      labGo(Math.round(L.pos) >= L.Ws.length - 1 ? 0 : Math.round(L.pos) + 1);
    });
    $('cmp-slider').addEventListener('input', labCompare);
  }
  function labReset() {
    labStop(); cancelAnimationFrame(L.raf);
    L.res = null; L.micSpec = null; L.specs = []; $('lab-specs').innerHTML = '';
    $('lab-panel').hidden = true; $('lab-spec-wrap').hidden = true; $('lab-out').innerHTML = ''; $('lab-vs').textContent = '';
    $('game-check').disabled = true; $('game-score').textContent = '';
  }

  // ---- 5 · Which is which? ----
  var G = { guess: [null, null, null], revealed: false };
  function gameReveal() {
    G.revealed = true; var right = 0, any = false;
    L.final_match.forEach(function (mt, i) {
      var color = colorOf(demoClips[mt.source]), el = $('ic' + i + '-wave').parentNode;
      el.style.setProperty('--c', color);
      drawWave($('ic' + i + '-wave'), lastOutputs[i].y, color); lastOutputs[i].color = color;
      var g = G.guess[i], ok = g === mt.source; if (g != null) { any = true; if (ok) right++; }
      $('ic' + i + '-note').innerHTML = (g == null ? '' : (ok ? '✓ ' : '✗ You said ' + esc(demoClips[g].label) + '. ')) +
        'Sounds like: <b>' + esc(demoClips[mt.source].label) + '</b>, match ' + mt.abs.toFixed(2) + (mt.corr < 0 ? ' (came out upside down, which sounds the same)' : '');
    });
    $('game-check').disabled = true;
    $('game-score').textContent = any ? 'You got ' + right + ' of 3.' : '';
  }

  function separate() {
    stopPlaying(); labReset();
    var seed = separations++ === 0 ? 1 : (Date.now() & 0xffff), Ws = [];
    var res = fastICA(mixtures, { maxIter: 200, tol: 1e-6, seed: seed, onRound: function (it, W) { Ws.push(W.map(function (r) { return r.slice(); })); } });
    var match = matchComponents(res.Y, sources);
    L.res = res; L.Ws = Ws; L.Y = []; L.scores = []; L.match = []; L.R = []; L.pos = 0;
    Ws.forEach(function (W, k) {
      var Y = applyMatrix(W, res.Z);
      L.scores.push(scoreOf(Y)); L.match.push(matchComponents(Y, sources));
      L.R.push(Y.map(function (y) { return sources.map(function (s) { return corr(y, s); }); }));
    });
    L.final = L.scores[L.scores.length - 1]; L.final_match = match;
    var lo = $('lab-out'); lo.innerHTML = '';
    for (var i = 0; i < 3; i++) (function (i) {
      var row = trackRow('Output ' + (i + 1), 'lab' + i, ' ', COLORS.mix); row.id = 'lab' + i; lo.appendChild(row);
      playButton($('lab' + i + '-play'), function () { var k = Math.round(L.pos); return L.Y[k] || (L.Y[k] = labOutputs(k)); });
    })(i);
    $('cmp-pick').innerHTML = '';
    [0, 1, 2].forEach(function (i) {
      var b = document.createElement('button'); b.type = 'button'; b.textContent = 'Output ' + (i + 1);
      b.addEventListener('click', function () { L.cmp = i; labCompare(); }); $('cmp-pick').appendChild(b);
    });
    $('lab-panel').hidden = false; $('lab-spec-wrap').hidden = false;
    $('lab-scrub').max = Ws.length - 1;
    var lr = /[?&]labround=(\d+)/.exec(params);
    L.pos = lr ? Math.min(+lr[1], Ws.length - 1) : 0;
    labRender(true);
    if (!lr && !/[?&]autorun=1\b/.test(params)) { L.playing = true; $('lab-play').textContent = 'Pause'; if (reduceMotion) { labStop(); labGo(Ws.length - 1); } else setTimeout(function () { if (L.playing) labGo(1); }, 400); }
    $('lab-vs').textContent = (T.best ? 'Your best score: ' + T.best.toFixed(4) + ' · ' : '') + 'Computer’s final score: ' + L.final.toFixed(4);
    tryUpdate();

    // final outputs for the matching game (gray until checked)
    var out = $('ica-out'); out.innerHTML = ''; lastOutputs = []; G.guess = [null, null, null]; G.revealed = false;
    res.Y.forEach(function (y, i) {
      out.appendChild(trackRow('Track ' + (i + 1), 'ic' + i, ' ', COLORS.mix));
      var yn = normalizePeak(y, 0.9);
      drawWave($('ic' + i + '-wave'), yn, COLORS.mix);
      lastOutputs.push({ y: yn, color: COLORS.mix });
      playButton($('ic' + i + '-play'), function () { return yn; });
      var chips = document.createElement('div'); chips.className = 'chips'; chips.setAttribute('role', 'group'); chips.setAttribute('aria-label', 'Track ' + (i + 1) + ' is');
      demoClips.forEach(function (c, j) {
        var b = document.createElement('button'); b.type = 'button'; b.textContent = c.label; b.style.setProperty('--c', colorOf(c)); b.setAttribute('aria-pressed', 'false');
        b.addEventListener('click', function () {
          if (G.revealed) return; G.guess[i] = j;
          [].forEach.call(chips.children, function (x, q) { x.setAttribute('aria-pressed', q === j ? 'true' : 'false'); });
        });
        chips.appendChild(b);
      });
      $('ic' + i + '-note').innerHTML = ''; $('ic' + i + '-note').parentNode.insertBefore(chips, $('ic' + i + '-note'));
    });
    $('game-check').disabled = false;
    $('ica-status').textContent = (res.converged ? 'Done! ' : 'Stopped. ') + 'It took ' + res.iterations +
      ' round' + (res.iterations === 1 ? '' : 's') + ' of adjusting.';
    root.__icaResult = { iterations: res.iterations, converged: res.converged, match: match, scores: L.scores };
    if (/[?&]autorun=1\b/.test(params)) gameReveal();
  }

  function buildSources() {
    var srcBox = $('sources'); srcBox.innerHTML = '';
    for (var i = 0; i < 3; i++) (function (i) {
      srcBox.appendChild(trackRow(demoClips[i].label, 'src' + i, null, colorOf(demoClips[i])));
      drawWave($('src' + i + '-wave'), sources[i], colorOf(demoClips[i]));
      playButton($('src' + i + '-play'), function () { return sources[i]; });
    })(i);
  }
  function demoInit() {
    var micBox = $('mics'), i;
    micBox.innerHTML = '';
    for (i = 0; i < 3; i++) micBox.appendChild(trackRow(MICS[i], 'mic' + i, null, COLORS.mix));
    buildSources();
    for (i = 0; i < 3; i++) (function (i) {
      playButton($('mic' + i + '-play'), function () { return mixtures[i]; });
    })(i);
    var sw = $('demo-src');
    if (sw) {
      var studs = clips.filter(function (c) { return c.kind === 'kids' && c.demo; });
      if (studs.length !== 3) sw.hidden = true;
      [].forEach.call(sw.querySelectorAll('button'), function (b) {
        b.addEventListener('click', function () {
          var list = b.dataset.src === 'students' ? studs : clips.filter(function (c) { return c.kind === 'example'; });
          if (list.length !== 3 || list[0] === demoClips[0]) return;
          [].forEach.call(sw.querySelectorAll('button'), function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
          stopPlaying(); demoClips = list; sources = list.map(function (c) { return c.x; });
          buildSources(); updateMixtures();
        });
      });
    }
    tryInit(); labInit();
    $('game-check').addEventListener('click', gameReveal);
    buildMatrix();
    updateMixtures();
    playButton($('play-all'), function () {
      var all = new Float64Array(N);
      for (var t = 0; t < N; t++) all[t] = sources[0][t] + sources[1][t] + sources[2][t];
      return all;
    });
    $('randomize').addEventListener('click', function () {
      var r = Math.random;
      do { for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) A[i][j] = Math.round((0.1 + 0.9 * r()) * 20) / 20; }
      while (Math.abs(det(A)) < 0.05);
      buildMatrix(); updateMixtures();
    });
    $('separate').addEventListener('click', function () {
      $('ica-status').textContent = 'Working…';
      setTimeout(separate, 30);
    });
    $('separate').disabled = false; $('randomize').disabled = false;
  }

  // Student recordings & submissions: real cards from CLASSROOM_CLIPS / STUDENT_SUBMISSIONS, placeholders otherwise.
  function kidsSlots() {
    var box = $('kids-slots'), html = '';
    CLASSROOM_CLIPS.forEach(function (c) {
      var col = colorOf(c);
      html += '<div class="card" style="border-color:' + rgba(col, 0.5) + '"><strong style="color:' + col + '">' + esc(c.label) + '</strong>' +
        '<span class="card-tag">' + (c.seconds ? c.seconds + ' s' + (c.seconds >= 30 ? ' excerpt' : '') + ' · ' : '') + (c.description ? esc(c.description) : '[Description — to be added]') + '</span>' +
        (c.strip ? '<img class="strip-img" src="' + esc(c.strip) + '" alt="Spectrogram of ' + esc(c.label) + '" loading="lazy">' : '') +
        '<audio controls preload="none" src="' + esc(c.src) + '"></audio></div>';
    });
    if (!CLASSROOM_CLIPS.length) ['kitchen', 'hall', 'classroom'].forEach(function (place) {
      html += '<div class="card empty" style="border-color:' + rgba(COLORS[place], 0.35) + '"><strong>[Student recording — to be added]</strong><div class="audio-slot" aria-hidden="true"></div></div>';
    });
    box.innerHTML = html;
    var sub = $('submissions'), sh = '';
    if (!sub) return;
    STUDENT_SUBMISSIONS.forEach(function (c) {
      sh += '<figure class="card shot"><img src="' + esc(c.src) + '" alt="' + esc(c.title || '') + '" loading="lazy">' +
        '<figcaption><strong>' + esc(c.title || '') + '</strong>' + (c.caption ? ' ' + esc(c.caption) : '') + '</figcaption></figure>';
    });
    if (!STUDENT_SUBMISSIONS.length) for (var k = 0; k < 2; k++)
      sh += '<div class="card empty shot"><div class="img-slot" aria-hidden="true"></div><strong>[Submission example — to be added]</strong></div>';
    sub.innerHTML = sh;
  }

  // ===================== Hero: a mixed signal that unmixes itself =====================
  function hero() {
    var cv = $('hero-canvas'), scope = $('hero-scope');
    if (!cv || !scope || !cv.getContext) return;
    var box = cv.parentNode, ctx = cv.getContext('2d'), W = 0, H = 0, dpr = 1, area = null;
    var raf = null, tStart = null, visible = true, p = reduceMotion ? 1 : 0, flow = 0;
    var DELAY = 500, DUR_SEP = 2800, TAU = Math.PI * 2;
    var LANES = [{ name: 'KITCHEN', color: COLORS.kitchen }, { name: 'HALL', color: COLORS.hall }, { name: 'CLASSROOM', color: COLORS.classroom }];
    function fract(v) { return v - Math.floor(v); }
    // three made-up "sounds": clattery bursts, a steady murmur, and syllable-like speech
    function sig(i, s) {
      if (i === 0) {
        var e = Math.exp(-7 * fract(s * 0.83)) * 0.95 + Math.exp(-10 * fract(s * 1.31 + 0.37)) * 0.6;
        return 0.65 * e * (Math.sin(TAU * s * 37) * 0.7 + Math.sin(TAU * s * 61.3 + 1.3) * 0.45 + Math.sin(TAU * s * 89.7) * 0.25);
      }
      if (i === 1) {
        return 0.42 * (Math.sin(TAU * s * 6.1) + 0.7 * Math.sin(TAU * s * 11.3 + 1) + 0.45 * Math.sin(TAU * s * 19.7 + 2)) *
          (0.75 + 0.25 * Math.sin(TAU * s * 0.37));
      }
      var env = Math.pow(Math.max(0, Math.sin(TAU * s * 1.9)), 1.6) * (0.65 + 0.35 * Math.sin(TAU * s * 0.53 + 0.6));
      return 0.95 * env * Math.sin(TAU * s * 22 + 2.2 * Math.sin(TAU * s * 2.7));
    }
    function size() {
      dpr = Math.min(2, root.devicePixelRatio || 1);
      var r = box.getBoundingClientRect(), s = scope.getBoundingClientRect();
      W = r.width; H = r.height;
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      area = { y: s.top - r.top, h: s.height };
    }
    function draw() {
      var c = ctx, e = ease(p), y0 = area.y, ah = area.h, mid = y0 + ah / 2, i, x;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.clearRect(0, 0, W, H);
      // scope frame + graticule over the wave area
      var pad = Math.min(24, W * 0.04);
      graticule(c, pad, y0, W - 2 * pad, ah, Math.max(6, Math.round((W - 2 * pad) / 90)), 6);
      c.strokeStyle = 'rgba(150,175,210,0.18)'; c.lineWidth = 1;
      c.strokeRect(Math.round(pad) + 0.5, Math.round(y0) + 0.5, Math.round(W - 2 * pad), Math.round(ah));
      var span = W / 230, step = W > 700 ? 1.5 : 1.25, lanes = [mid - ah / 3, mid, mid + ah / 3];
      var ampMix = ah * 0.24, ampLane = ah * 0.12;
      for (i = 2; i >= 0; i--) {
        var yc = mid + (lanes[i] - mid) * e, amp = ampMix + (ampLane - ampMix) * e;
        var col = blend(COLORS.mix, LANES[i].color, Math.min(1, e * 1.2));
        c.beginPath();
        for (x = pad; x <= W - pad; x += step) {
          var s = (x / W) * span * 3 + flow * 0.42;
          var m = (sig(0, s) + sig(1, s + 0.9) + sig(2, s + 0.3)) * 0.6;
          var v = m + (sig(i, s + (i === 1 ? 0.9 : i === 2 ? 0.3 : 0)) - m) * e;
          var Y = yc - v * amp;
          if (x === pad) c.moveTo(x, Y); else c.lineTo(x, Y);
        }
        glowStroke(c, col, 1.5, 12 * (0.4 + 0.6 * e));
      }
      // readouts
      c.font = '600 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      c.textAlign = 'left';
      c.fillStyle = rgba(COLORS.mix, 0.75 * (1 - Math.min(1, e * 2)));
      c.fillText('MIXED SIGNAL', pad + 2, y0 - 7);
      for (i = 0; i < 3; i++) {
        c.fillStyle = rgba(LANES[i].color, Math.max(0, e * 1.6 - 0.6));
        c.fillText('CH.' + 'ABC'.charAt(i) + '  ' + LANES[i].name, pad + 10, lanes[i] - ah / 6 + 16);
      }
      c.textAlign = 'right'; c.fillStyle = 'rgba(154,166,182,0.8)';
      c.fillText('ICA  ' + (p >= 1 ? 'SEPARATED' : 'UNMIXING ' + String(Math.round(p * 100)).padStart(3, ' ') + '%'), W - pad - 2, y0 - 7);
    }
    function frame(ts) {
      raf = null;
      if (tStart == null) tStart = ts;
      var el = ts - tStart;
      p = Math.max(0, Math.min(1, (el - DELAY) / DUR_SEP));
      flow = el / 1000;
      draw();
      if (p >= 1) box.setAttribute('data-hero', 'done');
      if (visible && !document.hidden) raf = requestAnimationFrame(frame);
    }
    function wake() { if (!reduceMotion && !raf && visible && !document.hidden) raf = requestAnimationFrame(frame); }
    size();
    if (reduceMotion) { draw(); box.setAttribute('data-hero', 'done'); }
    else {
      wake();
      if ('IntersectionObserver' in root) {
        new IntersectionObserver(function (es) { visible = es[0].isIntersecting; wake(); }).observe(box);
      }
      document.addEventListener('visibilitychange', wake);
    }
    var rt;
    root.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(function () { size(); draw(); }, 100); });
  }

  // ===================== Small page touches =====================
  function reveals() {
    if (reduceMotion || !('IntersectionObserver' in root)) return;
    var boxes = document.querySelectorAll('.page > .box:not(.toc)');
    document.documentElement.classList.add('reveal-on');
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -6% 0px', threshold: 0 });
    [].forEach.call(boxes, function (b) { io.observe(b); });
  }

  // Hand-rolled highlighting for the code and the test log (no libraries).
  function highlight() {
    var KW = /^(var|for|if|else|break|new|function|return|true|false|null)$/;
    [].forEach.call(document.querySelectorAll('pre > code'), function (el) {
      var src = el.textContent, out = '', last = 0, m;
      var re = /(\/\/[^\n]*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_$][\w$]*)|([{}()[\];,.=<>+\-*\/!&|]+)/g;
      while ((m = re.exec(src))) {
        out += esc(src.slice(last, m.index)); last = re.lastIndex;
        if (m[1]) out += '<span class="tk-c">' + esc(m[1]) + '</span>';
        else if (m[2]) out += '<span class="tk-n">' + m[2] + '</span>';
        else if (m[3]) {
          if (KW.test(m[3])) out += '<span class="tk-k">' + m[3] + '</span>';
          else if (src.charAt(last) === '(') out += '<span class="tk-f">' + m[3] + '</span>';
          else out += m[3];
        } else out += '<span class="tk-p">' + esc(m[4]) + '</span>';
      }
      el.innerHTML = out + esc(src.slice(last));
    });
    [].forEach.call(document.querySelectorAll('pre.log'), function (el) {
      el.innerHTML = esc(el.textContent)
        .replace(/\b(Kitchen|Hall|Classroom)\b/g, function (w) { return '<span class="tk-' + w.toLowerCase() + '">' + w + '</span>'; })
        .replace(/\bPASS\b/g, '<span class="tk-pass">PASS</span>');
    });
  }

  function init() {
    hero();
    reveals();
    highlight();
    kidsSlots();
    var status = $('load-status');
    status.textContent = 'Loading sounds…';
    var list = EXAMPLE_CLIPS.map(function (c) { var o = Object.assign({}, c); o.kind = 'example'; return o; })
      .concat(CLASSROOM_CLIPS.map(function (c, i) { var o = Object.assign({ id: 'kids' + i }, c); o.kind = 'kids'; return o; }));
    Promise.all(list.map(function (c) { return loadClip(c).catch(function (e) { console.warn(e); return null; }); }))
      .then(function (loaded) {
        clips = loaded.filter(Boolean);
        var kids = clips.filter(function (c) { return c.kind === 'kids'; });
        var ex = clips.filter(function (c) { return c.kind === 'example'; });
        demoClips = ex;
        if (demoClips.length < 3 || clips.length < 2) throw new Error('Some sounds did not load.');
        sources = demoClips.map(function (c) { return c.x; });
        status.textContent = 'Sounds loaded: ' + clips.length + ' clips, ' + DUR + ' seconds each.';
        document.body.setAttribute('data-sounds', 'loaded');
        vizInit();
        demoInit();
        if (/[?&](autorun=1\b|labround=)/.test(params)) separate();
      })
      .catch(function (e) {
        status.textContent = 'Sorry, the sounds could not be loaded (' + e.message + '). If you opened this file directly, try a local web server.';
        document.body.setAttribute('data-sounds', 'error');
      });

    var resizeTimer;
    root.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        if (!sources) return;
        for (var i = 0; i < 3; i++) {
          drawWave($('src' + i + '-wave'), sources[i], colorOf(demoClips[i]));
          drawWave($('mic' + i + '-wave'), mixtures[i], COLORS.mix);
          if (lastOutputs && $('ic' + i + '-wave')) drawWave($('ic' + i + '-wave'), lastOutputs[i].y, lastOutputs[i].color);
        }
        liveSize();
        if (!V.running) vizRender();
      }, 150);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(typeof window !== 'undefined' ? window : this);
