/* Unmixing School Sounds: a small FastICA project. Vanilla JS, no dependencies.
   The math part (top) also runs in Node for testing: node test/ica-test.js */

// ===================== Settings you can change =====================
// Kids' own recordings. When this list is empty, the page uses the example sounds below
// and shows empty slots for the kids' recordings. Add one entry per recording, e.g.
//   { label: 'Kitchen (Class 4B)', place: 'kitchen', src: 'audio/kids/kitchen-4b.mp3' },
//   { label: 'Hall (Class 4B)',    place: 'hall',    src: 'audio/kids/hall-4b.mp3' },
//   { label: 'Classroom (Class 4B)', place: 'classroom', src: 'audio/kids/classroom-4b.mp3' }
// place is one of: 'kitchen', 'hall', 'classroom'. Any audio format the browser can play works;
// each recording is cut or padded to 9 seconds. With 3 or more recordings, the full demo uses the
// first three of them instead of the examples. All recordings also show up in the visualizer's lists.
var CLASSROOM_CLIPS = [];

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
    EXAMPLE_CLIPS: EXAMPLE_CLIPS, CLASSROOM_CLIPS: CLASSROOM_CLIPS
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = ICA; return; }
  root.ICA = ICA;

  // ============================ Browser UI ============================
  var ACCENT = '#1f3a5f';
  var PLACE_NAMES = { kitchen: 'the kitchen', hall: 'the hall', classroom: 'a classroom' };
  var MICS = ['Mic by the door', 'Mic by the window', 'Mic at the back'];
  var reduceMotion = !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var params = root.location ? root.location.search : '';
  var clips = [];          // all loaded clips: { id, label, place, src, kind: 'example' | 'kids', x: Float64Array }
  var demoClips = [];      // the three used in the full demo
  var sources = null, A = DEFAULT_MIX.map(function (r) { return r.slice(); }), mixtures = null;
  var audioCtx = null, playing = null, playingBtn = null;

  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

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
    return { c: c, w: w, h: h };
  }

  function drawWave(canvas, x) {
    var g = setupCanvas(canvas, 64), c = g.c, w = g.w, h = g.h;
    c.fillStyle = '#f6f7f9'; c.fillRect(0, 0, w, h);
    c.strokeStyle = '#dde1e6'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(0, h / 2 + 0.5); c.lineTo(w, h / 2 + 0.5); c.stroke();
    if (!x) return;
    var peak = 0, i;
    for (i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    if (peak === 0) return;
    var scale = (h / 2 - 3) / peak, per = x.length / w;
    c.strokeStyle = ACCENT; c.beginPath();
    for (var px = 0; px < w; px++) {
      var a = Math.floor(px * per), b = Math.min(x.length, Math.floor((px + 1) * per));
      var lo = Infinity, hi = -Infinity;
      for (i = a; i < b; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
      if (lo === Infinity) continue;
      c.moveTo(px + 0.5, h / 2 - hi * scale); c.lineTo(px + 0.5, h / 2 - lo * scale + 0.01);
    }
    c.stroke();
  }

  // ---------- playback ----------
  function stopPlaying() {
    if (playing) { try { playing.stop(); } catch (e) {} playing = null; }
    if (playingBtn) { playingBtn.textContent = playingBtn.dataset.label || 'Play'; playingBtn = null; }
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
    src.buffer = buf; src.connect(audioCtx.destination); src.start();
    playing = src;
    if (btn) {
      btn.dataset.label = btn.dataset.label || btn.textContent;
      btn.textContent = 'Stop'; playingBtn = btn;
    }
    src.onended = function () { if (playing === src) stopPlaying(); };
  }
  function playButton(btn, getSignal) {
    btn.addEventListener('click', function () { var x = getSignal(); if (x) play(x, btn); });
  }

  function trackRow(label, id, note) {
    var div = document.createElement('div');
    div.className = 'track';
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
      text: 'First the computer slides the whole cloud so its center sits on the middle of the plot. (I added a small steady offset to each mic so you can see this step. Real sound files are usually already close to the middle.)' },
    { title: 'Make it round',
      text: 'Next it stretches and squashes the cloud until it is spread out the same amount in every direction, so each microphone counts fairly. The cross is still there, but its two arms are now at a right angle.' },
    { title: 'Turn the dial',
      text: 'Now only one thing is left: which way to turn. Drag the dial. The two lines are the two outputs. When the lines run along the arms of the cross, the independence meter is highest and each output is one clean sound. Listen to the difference.' }
  ];

  function sampleIdx(n, k, seed) {   // k random moments (fixed seed so the picture is stable)
    var r = rng(seed), idx = new Uint32Array(k);
    for (var i = 0; i < k; i++) idx[i] = Math.floor(r() * n);
    return idx;
  }
  function std(x) { var m = 0, s = 0, t; for (t = 0; t < x.length; t++) m += x[t]; m /= x.length; for (t = 0; t < x.length; t++) s += (x[t] - m) * (x[t] - m); return Math.sqrt(s / x.length) || 1; }
  function rot(theta) { var c = Math.cos(theta), s = Math.sin(theta); return [[c, s], [-s, c]]; }

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
    // independence meter for every angle (0.5 degree steps) on 16,000 moments
    var z1 = new Float64Array(V.sub.length), z2 = new Float64Array(V.sub.length);
    for (t = 0; t < V.sub.length; t++) { z1[t] = V.Z[0][V.sub[t]]; z2[t] = V.Z[1][V.sub[t]]; }
    V.z1 = z1; V.z2 = z2;
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

  function drawScatter(pts, view, angle) {
    var cv = $('viz-canvas'), g = setupCanvas(cv, 420), c = g.c, w = g.w, h = g.h;
    var half = Math.min(w, h) / 2 - 18, cx = w / 2, cy = h / 2, k = half / view;
    c.fillStyle = '#fff'; c.fillRect(0, 0, w, h);
    c.strokeStyle = '#d5dae0'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(cx - half - 8, cy + 0.5); c.lineTo(cx + half + 8, cy + 0.5);
    c.moveTo(cx + 0.5, cy - half - 8); c.lineTo(cx + 0.5, cy + half + 8); c.stroke();
    c.fillStyle = 'rgba(31,58,95,0.30)';
    for (var i = 0; i < pts.length; i += 2) c.fillRect(cx + pts[i] * k - 1, cy - pts[i + 1] * k - 1, 2, 2);
    var lbl = V.step === 0 ? ['Sound A', 'Sound B'] : (V.step >= 3 ? ['', ''] : ['Mic 1', 'Mic 2']);
    c.fillStyle = '#5b6470'; c.font = '12px system-ui, -apple-system, sans-serif';
    c.textAlign = 'right'; c.fillText(lbl[0], cx + half + 8, cy - 6);
    c.textAlign = 'left'; c.fillText(lbl[1], cx + 6, cy - half - 2);
    if (angle != null) {
      var th = angle * Math.PI / 180, L = half + 6;
      [[Math.cos(th), Math.sin(th), '1'], [-Math.sin(th), Math.cos(th), '2']].forEach(function (d) {
        c.strokeStyle = ACCENT; c.lineWidth = 2;
        c.beginPath(); c.moveTo(cx - d[0] * L, cy + d[1] * L); c.lineTo(cx + d[0] * L, cy - d[1] * L); c.stroke();
        c.fillStyle = ACCENT; c.font = 'bold 13px system-ui, -apple-system, sans-serif'; c.textAlign = 'center';
        c.fillText('Output ' + d[2], cx + d[0] * (L - 26), cy - d[1] * (L - 26) - 6);
      });
    }
  }

  function drawHist(canvas, y) {
    var g = setupCanvas(canvas, 70), c = g.c, w = g.w, h = g.h, bins = 41, lo = -4, hi = 4, cnt = new Array(bins).fill(0), i, mx = 0;
    for (i = 0; i < y.length; i++) { var b = Math.floor((y[i] - lo) / (hi - lo) * bins); if (b >= 0 && b < bins) cnt[b]++; }
    for (i = 0; i < bins; i++) mx = Math.max(mx, cnt[i]);
    c.fillStyle = '#fff'; c.fillRect(0, 0, w, h);
    c.fillStyle = ACCENT;
    var bw = w / bins;
    for (i = 0; i < bins; i++) { var bh = (h - 4) * cnt[i] / mx; c.fillRect(i * bw + 0.5, h - bh, Math.max(1, bw - 1), bh); }
  }

  function vizRender() {
    var s = V.step;
    if (V.anim) return;
    drawScatter(V.frames[s], V.views[s], s === 4 ? V.angle : null);
    if (s === 4) vizDial();
  }

  function vizDial() {
    var j = meterAt(V.angle), pct = Math.round(100 * j / V.maxJ);
    $('viz-angle-out').textContent = Math.round(V.angle) + '°';
    $('viz-meter-bar').style.width = Math.max(1, pct) + '%';
    $('viz-meter-out').textContent = pct + '% of the best';
    var y = project(V.angle);
    drawHist($('viz-hist1'), y[0]); drawHist($('viz-hist2'), y[1]);
    if ($('viz-strip-wrap').open) drawStrip();
  }

  function drawStrip() {
    var cv = $('viz-strip'), g = setupCanvas(cv, 60), c = g.c, w = g.w, h = g.h;
    c.fillStyle = '#fff'; c.fillRect(0, 0, w, h);
    c.strokeStyle = ACCENT; c.lineWidth = 1.5; c.beginPath();
    V.table.forEach(function (j, d) { var x = d / 360 * w, y = h - 4 - (h - 10) * j / V.maxJ; if (d) c.lineTo(x, y); else c.moveTo(x, y); });
    c.stroke();
    var xa = V.angle / 180 * w;
    c.strokeStyle = '#8a2b2b'; c.beginPath(); c.moveTo(xa, 0); c.lineTo(xa, h); c.stroke();
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
    $('viz-done').textContent = ''; $('viz-rounds').textContent = '';
    vizRender();
  }

  function vizMixer() {
    var box = $('viz-mixer'), html = '';
    var la = V.clipList[V.a].label, lb = V.clipList[V.b].label;
    for (var i = 0; i < 2; i++) {
      html += '<div class="mixrow"><span class="mixmic">Mic ' + (i + 1) + ' hears</span>';
      for (var j = 0; j < 2; j++) {
        html += '<label class="slider">' + esc(j ? 'B: ' + lb : 'A: ' + la) +
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
      $('viz-rounds').textContent = 'Round ' + h.r + ' · dial at ' + Math.round(angleOf(h.W)) + '° · meter ' +
        Math.round(100 * meterAt(angleOf(h.W)) / V.maxJ) + '%' + (h.change != null ? ' · moved ' + h.change.toExponential(1) : '');
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
    var opts = clips.map(function (c, i) { return '<option value="' + i + '">' + esc(c.label) + (c.kind === 'example' ? ' (example)' : '') + '</option>'; }).join('');
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
    for (j = 0; j < 3; j++) html += '<th scope="col">' + esc(demoClips[j].label) + '</th>';
    html += '</tr>';
    for (i = 0; i < 3; i++) {
      html += '<tr><th scope="row">' + MICS[i] + '</th>';
      for (j = 0; j < 3; j++)
        html += '<td><input type="number" min="0" max="1" step="0.05" id="a' + i + j + '" value="' +
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
    mixtures = mix(A, sources);
    for (var i = 0; i < 3; i++) drawWave($('mic' + i + '-wave'), mixtures[i]);
    $('mix-warn').textContent = Math.abs(det(A)) < 1e-3
      ? 'With these numbers, the mics hear (almost) the same blend, so the sounds can’t be pulled apart. Try changing a number.' : '';
    $('ica-out').innerHTML = '<p class="muted">Press “Separate” to let the computer pull the sounds apart.</p>';
    $('ica-status').textContent = '';
  }

  var separations = 0;
  function separate() {
    var seed = separations++ === 0 ? 1 : (Date.now() & 0xffff);
    var res = fastICA(mixtures, { maxIter: 200, tol: 1e-6, seed: seed });
    var match = matchComponents(res.Y, sources);
    var out = $('ica-out');
    out.innerHTML = '';
    res.Y.forEach(function (y, i) {
      var mt = match[i];
      var note = 'Sounds like: ' + esc(demoClips[mt.source].label) + ', match ' + mt.abs.toFixed(2) +
        (mt.corr < 0 ? ' (came out upside down, which sounds the same)' : '');
      out.appendChild(trackRow('Track ' + (i + 1), 'ic' + i, note));
      var yn = normalizePeak(y, 0.9);
      drawWave($('ic' + i + '-wave'), yn);
      playButton($('ic' + i + '-play'), function () { return yn; });
    });
    $('ica-status').textContent = (res.converged ? 'Done! ' : 'Stopped. ') + 'It took ' + res.iterations +
      ' round' + (res.iterations === 1 ? '' : 's') + ' of adjusting.';
    root.__icaResult = { iterations: res.iterations, converged: res.converged, match: match };
  }

  function demoInit() {
    var srcBox = $('sources'), micBox = $('mics'), i;
    srcBox.innerHTML = ''; micBox.innerHTML = '';
    for (i = 0; i < 3; i++) {
      srcBox.appendChild(trackRow(demoClips[i].label + (demoClips[i].kind === 'example' ? ' (example sound)' : ''), 'src' + i));
      micBox.appendChild(trackRow(MICS[i], 'mic' + i));
    }
    for (i = 0; i < 3; i++) (function (i) {
      drawWave($('src' + i + '-wave'), sources[i]);
      playButton($('src' + i + '-play'), function () { return sources[i]; });
      playButton($('mic' + i + '-play'), function () { return mixtures[i]; });
    })(i);
    buildMatrix();
    updateMixtures();
    var all = new Float64Array(N);
    for (i = 0; i < N; i++) all[i] = sources[0][i] + sources[1][i] + sources[2][i];
    playButton($('play-all'), function () { return all; });
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

  // Slots for the kids' recordings (filled from CLASSROOM_CLIPS, empty placeholders otherwise).
  function kidsSlots() {
    var box = $('kids-slots'), html = '';
    ['kitchen', 'hall', 'classroom'].forEach(function (place) {
      var mine = CLASSROOM_CLIPS.filter(function (c) { return c.place === place; });
      if (!mine.length) html += '<div class="slot empty">[Kids’ recording from ' + PLACE_NAMES[place] + ' — to be added]</div>';
      mine.forEach(function (c) {
        html += '<div class="slot"><strong>' + esc(c.label) + '</strong><audio controls preload="none" src="' + esc(c.src) + '"></audio></div>';
      });
    });
    box.innerHTML = html;
  }

  function init() {
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
        demoClips = kids.length >= 3 ? kids.slice(0, 3) : ex;
        if (demoClips.length < 3 || clips.length < 2) throw new Error('Some sounds did not load.');
        sources = demoClips.map(function (c) { return c.x; });
        status.textContent = 'Sounds loaded: ' + clips.length + ' clips, ' + DUR + ' seconds each.';
        document.body.setAttribute('data-sounds', 'loaded');
        vizInit();
        demoInit();
        if (/[?&]autorun=1\b/.test(params)) separate();
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
        for (var i = 0; i < 3; i++) { drawWave($('src' + i + '-wave'), sources[i]); drawWave($('mic' + i + '-wave'), mixtures[i]); }
        if (!V.running) vizRender();
      }, 150);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(typeof window !== 'undefined' ? window : this);
