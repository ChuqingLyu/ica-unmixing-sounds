/* Unmixing Sounds: a small FastICA demo. Vanilla JS, no dependencies.
   The math section (top) also runs in Node for testing. */
(function (root) {
  'use strict';

  var SR = 16000;         // sample rate (Hz)
  var BPM = 120;          // tempo
  var BEAT = 60 / BPM;    // seconds per beat
  var DUR = 8;            // 16 beats = 4 bars of 4/4
  var N = SR * DUR;

  // ---------- small helpers ----------
  function rng(seed) {    // mulberry32, deterministic
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

  // ---------- sources: a tiny class band playing "Twinkle Twinkle Little Star" ----------
  // Melody notes as [MIDI note, start beat, length in beats]. Key of C, 120 BPM.
  var MELODY = [
    [60, 0, 1], [60, 1, 1], [67, 2, 1], [67, 3, 1], [69, 4, 1], [69, 5, 1], [67, 6, 2],
    [65, 8, 1], [65, 9, 1], [64, 10, 1], [64, 11, 1], [62, 12, 1], [62, 13, 1], [60, 14, 2]
  ];
  // Bass: root notes (two beats each) under the melody: C C F C | F C G C
  var BASS = [
    [36, 0, 2], [36, 2, 2], [41, 4, 2], [36, 6, 2],
    [41, 8, 2], [36, 10, 2], [43, 12, 2], [36, 14, 2]
  ];

  function midiHz(m) { return 440 * Math.pow(2, (m - 69) / 12); }

  // Add one decaying, harmonic note into buffer `out`.
  function addNote(out, hz, t0, len, harmonics, tau, attack) {
    var i0 = Math.round(t0 * SR), n = Math.round(len * SR), i, h;
    for (i = 0; i < n && i0 + i < out.length; i++) {
      var t = i / SR;
      var env = Math.min(1, t / attack) * Math.exp(-t / tau) * Math.min(1, (n - i) / (0.02 * SR));
      var v = 0;
      for (h = 0; h < harmonics.length; h++) v += harmonics[h] * Math.sin(2 * Math.PI * hz * (h + 1) * t);
      out[i0 + i] += env * v;
    }
  }

  function makeSources() {
    var piano = new Float64Array(N), drums = new Float64Array(N), bass = new Float64Array(N);
    var i, k;

    // Piano (melody): a few harmonics, quick attack, exponential decay per note.
    MELODY.forEach(function (nt) {
      addNote(piano, midiHz(nt[0]), nt[1] * BEAT, nt[2] * BEAT, [1, 0.45, 0.22, 0.1], 0.35, 0.004);
    });

    // Bass (low notes): low plucked tone, slightly behind the beat like a real player.
    BASS.forEach(function (nt) {
      addNote(bass, midiHz(nt[0]), nt[1] * BEAT + 0.03, nt[2] * BEAT - 0.03, [1, 0.6, 0.3, 0.15], 0.5, 0.01);
    });

    // Drums (beat): kick on 1 and 3, snare on 2 and 4, soft hi-hat on the off-beats.
    var r = rng(7);
    for (k = 0; k < 16; k++) {
      var i0 = Math.round(k * BEAT * SR), j, len;
      if (k % 2 === 0) {                        // kick: low sine thump with falling pitch
        len = Math.round(0.25 * SR);
        var ph = 0;
        for (j = 0; j < len && i0 + j < N; j++) {
          var t = j / SR, f = 50 + 70 * Math.exp(-t / 0.04);
          ph += 2 * Math.PI * f / SR;
          drums[i0 + j] += Math.exp(-t / 0.09) * Math.sin(ph);
        }
      } else {                                  // snare: short noise burst
        len = Math.round(0.18 * SR);
        for (j = 0; j < len && i0 + j < N; j++)
          drums[i0 + j] += 0.8 * Math.exp(-(j / SR) / 0.05) * (2 * r() - 1);
      }
      var h0 = Math.round((k + 0.5) * BEAT * SR), prev = 0;  // hi-hat: high, very short tick
      len = Math.round(0.04 * SR);
      for (j = 0; j < len && h0 + j < N; j++) {
        var w = 2 * r() - 1;
        drums[h0 + j] += 0.25 * Math.exp(-(j / SR) / 0.01) * (w - prev);
        prev = w;
      }
    }

    return [normalizePeak(piano, 0.9), normalizePeak(drums, 0.9), normalizePeak(bass, 0.9)];
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

  function det3(A) {
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

  // Jacobi eigen-decomposition of a symmetric matrix.
  // Returns { values: [...], vectors: V } with A = V diag(values) V^T (columns of V are eigenvectors).
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
          for (i = 0; i < n; i++) {           // A = A * J
            var aip = A[i][p], aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[i][q] = s * aip + c * aiq;
          }
          for (j = 0; j < n; j++) {           // A = J^T * A
            var apj = A[p][j], aqj = A[q][j];
            A[p][j] = c * apj - s * aqj;
            A[q][j] = s * apj + c * aqj;
          }
          for (i = 0; i < n; i++) {           // V = V * J
            var vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
          }
        }
      }
    }
    var values = [];
    for (i = 0; i < n; i++) values.push(A[i][i]);
    return { values: values, vectors: V };
  }

  // Symmetric decorrelation: W <- (W W^T)^(-1/2) W
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

  // ---------- FastICA (symmetric, tanh / logcosh nonlinearity) ----------
  // X: array of m signals (each length n). Returns { Y, iterations, converged }.
  function fastICA(X, opts) {
    opts = opts || {};
    var maxIter = opts.maxIter || 200, tol = opts.tol || 1e-6;
    var rand = rng(opts.seed == null ? 1 : opts.seed);
    var m = X.length, n = X[0].length, i, j, t;

    // 1) center
    var Xc = [];
    for (i = 0; i < m; i++) {
      var mean = 0;
      for (t = 0; t < n; t++) mean += X[i][t];
      mean /= n;
      var xc = new Float64Array(n);
      for (t = 0; t < n; t++) xc[t] = X[i][t] - mean;
      Xc.push(xc);
    }

    // 2) whiten: C = E D E^T, K = D^(-1/2) E^T, Z = K Xc
    var C = [];
    for (i = 0; i < m; i++) {
      C.push(new Array(m).fill(0));
      for (j = 0; j < m; j++) {
        var sum = 0;
        for (t = 0; t < n; t++) sum += Xc[i][t] * Xc[j][t];
        C[i][j] = sum / n;
      }
    }
    var eig = jacobiEig(C);
    var K = [];
    for (i = 0; i < m; i++) {
      K.push(new Array(m).fill(0));
      var d = 1 / Math.sqrt(Math.max(eig.values[i], 1e-12));
      for (j = 0; j < m; j++) K[i][j] = d * eig.vectors[j][i];
    }
    var Z = [];
    for (i = 0; i < m; i++) {
      var z = new Float64Array(n);
      for (j = 0; j < m; j++) { var kij = K[i][j]; for (t = 0; t < n; t++) z[t] += kij * Xc[j][t]; }
      Z.push(z);
    }

    // 3) iterate: w <- E[z g(w^T z)] - E[g'(w^T z)] w, then symmetric decorrelation
    var W = [];
    for (i = 0; i < m; i++) { W.push([]); for (j = 0; j < m; j++) W[i].push(rand() * 2 - 1); }
    W = symDecorrelate(W);

    var it, converged = false, wz = new Float64Array(n);
    for (it = 1; it <= maxIter; it++) {
      var Wn = [];
      for (i = 0; i < m; i++) {
        for (t = 0; t < n; t++) {
          var v = 0;
          for (j = 0; j < m; j++) v += W[i][j] * Z[j][t];
          wz[t] = v;
        }
        var row = new Array(m).fill(0), gp = 0;
        for (t = 0; t < n; t++) {
          var g = Math.tanh(wz[t]);
          gp += 1 - g * g;
          for (j = 0; j < m; j++) row[j] += Z[j][t] * g;
        }
        for (j = 0; j < m; j++) row[j] = row[j] / n - (gp / n) * W[i][j];
        Wn.push(row);
      }
      Wn = symDecorrelate(Wn);
      var lim = 0;
      for (i = 0; i < m; i++) {
        var dot = 0;
        for (j = 0; j < m; j++) dot += Wn[i][j] * W[i][j];
        lim = Math.max(lim, Math.abs(Math.abs(dot) - 1));
      }
      W = Wn;
      if (lim < tol) { converged = true; break; }
    }

    var Y = [];
    for (i = 0; i < m; i++) {
      var y = new Float64Array(n);
      for (j = 0; j < m; j++) { var w = W[i][j]; for (t = 0; t < n; t++) y[t] += w * Z[j][t]; }
      Y.push(y);
    }
    return { Y: Y, iterations: Math.min(it, maxIter), converged: converged };
  }

  // ---------- matching recovered components to sources ----------
  function corr(a, b) {
    var n = a.length, ma = 0, mb = 0, t;
    for (t = 0; t < n; t++) { ma += a[t]; mb += b[t]; }
    ma /= n; mb /= n;
    var sab = 0, saa = 0, sbb = 0;
    for (t = 0; t < n; t++) {
      var da = a[t] - ma, db = b[t] - mb;
      sab += da * db; saa += da * da; sbb += db * db;
    }
    return (saa > 0 && sbb > 0) ? sab / Math.sqrt(saa * sbb) : 0;
  }

  // For each recovered component, the source it best matches (one-to-one, by |corr|).
  function matchComponents(Y, S) {
    var m = Y.length, R = [], i, j;
    for (i = 0; i < m; i++) { R.push([]); for (j = 0; j < S.length; j++) R[i].push(corr(Y[i], S[j])); }
    var best = null, bestScore = -1;
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
    return Y.map(function (_, i) {
      return { source: best[i], corr: R[i][best[i]], abs: Math.abs(R[i][best[i]]) };
    });
  }

  var ICA = {
    SR: SR, N: N, rng: rng, makeSources: makeSources, mix: mix, det3: det3,
    jacobiEig: jacobiEig, fastICA: fastICA, corr: corr, matchComponents: matchComponents,
    normalizePeak: normalizePeak
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = ICA; return; }
  root.ICA = ICA;

  // ==================== Browser UI ====================
  var NAMES = ['Piano (melody)', 'Drums (beat)', 'Bass (low notes)'];
  var SHORT = ['Piano', 'Drums', 'Bass'];
  var MICS = ['Mic near piano', 'Mic near drums', 'Mic at the back'];
  var sources = makeSources();
  var A = [[1.0, 0.35, 0.45], [0.3, 1.0, 0.4], [0.6, 0.55, 0.8]];
  var mixtures = null;
  var audioCtx = null, playing = null;

  function $(id) { return document.getElementById(id); }

  function drawWave(canvas, x) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || 600, h = canvas.clientHeight || 64;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    var c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#f1f3f5'; c.fillRect(0, 0, w, h);
    c.strokeStyle = '#d5dae0'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(0, h / 2 + 0.5); c.lineTo(w, h / 2 + 0.5); c.stroke();
    if (!x) return;
    var peak = 0, i;
    for (i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    if (peak === 0) return;
    var scale = (h / 2 - 3) / peak, per = x.length / w;
    c.strokeStyle = '#1f3a5f'; c.beginPath();
    for (var px = 0; px < w; px++) {
      var a = Math.floor(px * per), b = Math.min(x.length, Math.floor((px + 1) * per));
      var lo = Infinity, hi = -Infinity;
      for (i = a; i < b; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
      if (lo === Infinity) continue;
      c.moveTo(px + 0.5, h / 2 - hi * scale); c.lineTo(px + 0.5, h / 2 - lo * scale + 0.01);
    }
    c.stroke();
  }

  function play(x) {
    if (!audioCtx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { alert('Sorry, this browser does not support the Web Audio API.'); return; }
      audioCtx = new AC();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
    if (playing) { try { playing.stop(); } catch (e) {} }
    var y = normalizePeak(x, 0.8);
    var buf = audioCtx.createBuffer(1, y.length, SR);
    buf.copyToChannel ? buf.copyToChannel(y, 0) : buf.getChannelData(0).set(y);
    var src = audioCtx.createBufferSource();
    src.buffer = buf; src.connect(audioCtx.destination); src.start();
    playing = src;
  }

  function row(label, id, note) {
    var div = document.createElement('div');
    div.className = 'track';
    div.innerHTML =
      '<div class="track-head"><span class="track-label">' + label + '</span>' +
      '<button type="button" class="play" id="' + id + '-play">Play</button></div>' +
      '<canvas id="' + id + '-wave" class="wave" aria-label="Waveform: ' + label + '"></canvas>' +
      (note ? '<div class="match" id="' + id + '-note">' + note + '</div>' : '');
    return div;
  }

  function buildMatrix() {
    var tbl = $('mix-matrix'), html = '<tr><th></th>', i, j;
    for (j = 0; j < 3; j++) html += '<th scope="col">' + SHORT[j] + '</th>';
    html += '</tr>';
    for (i = 0; i < 3; i++) {
      html += '<tr><th scope="row">' + MICS[i] + '</th>';
      for (j = 0; j < 3; j++)
        html += '<td><input type="number" min="0" max="1" step="0.05" id="a' + i + j + '" value="' +
          A[i][j].toFixed(2) + '" aria-label="How loud ' + SHORT[j] + ' is at the ' + MICS[i].toLowerCase() + '"></td>';
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
    var warn = $('mix-warn');
    warn.textContent = Math.abs(det3(A)) < 1e-3
      ? 'With these numbers, the mics hear (almost) the same blend, so the instruments can\'t be pulled apart. Try changing a number.'
      : '';
    clearOutput();
  }

  function clearOutput() {
    var out = $('ica-out');
    out.innerHTML = '<p class="muted">Press “Separate” to let the computer pull the instruments apart.</p>';
    $('ica-status').textContent = '';
  }

  function separate() {
    var status = $('ica-status');
    var res = fastICA(mixtures, { maxIter: 200, tol: 1e-6, seed: (Date.now() & 0xffff) });
    var match = matchComponents(res.Y, sources);
    var out = $('ica-out');
    out.innerHTML = '';
    res.Y.forEach(function (y, i) {
      var mt = match[i];
      var note = 'Sounds like: ' + NAMES[mt.source] + ', match ' + mt.abs.toFixed(2) +
        (mt.corr < 0 ? ' (came out upside down, which sounds the same)' : '');
      out.appendChild(row('Track ' + (i + 1), 'ic' + i, note));
      var yn = normalizePeak(y, 0.9);
      drawWave($('ic' + i + '-wave'), yn);
      $('ic' + i + '-play').addEventListener('click', function () { play(yn); });
    });
    status.textContent = (res.converged ? 'Done! ' : 'Stopped. ') + 'It took ' + res.iterations +
      ' round' + (res.iterations === 1 ? '' : 's') + ' of adjusting.';
    root.__icaResult = { iterations: res.iterations, converged: res.converged, match: match };
  }

  function init() {
    var srcBox = $('sources'), micBox = $('mics'), i;
    for (i = 0; i < 3; i++) {
      srcBox.appendChild(row(NAMES[i], 'src' + i));
      micBox.appendChild(row(MICS[i], 'mic' + i));
    }
    for (i = 0; i < 3; i++) (function (i) {
      drawWave($('src' + i + '-wave'), sources[i]);
      $('src' + i + '-play').addEventListener('click', function () { play(sources[i]); });
      $('mic' + i + '-play').addEventListener('click', function () { play(mixtures[i]); });
    })(i);
    buildMatrix();
    updateMixtures();

    var band = new Float32Array(sources[0].length);
    for (i = 0; i < band.length; i++) band[i] = sources[0][i] + sources[1][i] + sources[2][i];
    $('play-all').addEventListener('click', function () { play(band); });

    $('randomize').addEventListener('click', function () {
      var r = Math.random;
      do {
        for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) A[i][j] = Math.round((0.1 + 0.9 * r()) * 20) / 20;
      } while (Math.abs(det3(A)) < 0.05);
      buildMatrix();
      updateMixtures();
    });
    $('separate').addEventListener('click', function () {
      $('ica-status').textContent = 'Working…';
      setTimeout(separate, 30);   // let the "Working…" text show first
    });

    var resizeTimer;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        for (var i = 0; i < 3; i++) {
          drawWave($('src' + i + '-wave'), sources[i]);
          drawWave($('mic' + i + '-wave'), mixtures[i]);
        }
      }, 150);
    });

    if (/[?&]autorun=1\b/.test(location.search)) separate();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(typeof window !== 'undefined' ? window : this);
