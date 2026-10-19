// Separation test for the three example sounds. Run from the repo root: node test/ica-test.js
// Uses test/sources-16k.s16: the three clips in audio/ decoded to 16 kHz, 16-bit, 9 s each
// (made by tools/make_audio.sh), so the test does not need ffmpeg or a browser.
'use strict';
const fs = require('fs');
const path = require('path');
const ICA = require('../ica.js');

const raw = fs.readFileSync(path.join(__dirname, 'sources-16k.s16'));
const i16 = new Int16Array(raw.buffer, raw.byteOffset, raw.length / 2);
const N = ICA.N;
const names = ICA.EXAMPLE_CLIPS.map(c => c.label);
const S = [0, 1, 2].map(k => Float64Array.from(i16.subarray(k * N, (k + 1) * N), v => v / 32768));

const kurt = x => { const n = x.length; let m = 0; for (const v of x) m += v; m /= n; let s2 = 0, s4 = 0;
  for (const v of x) { const d = (v - m) * (v - m); s2 += d; s4 += d * d; } s2 /= n; s4 /= n; return s4 / (s2 * s2) - 3; };
console.log(`Sources: ${names.join(', ')} | ${N} samples each at ${ICA.SR} Hz (${ICA.DUR} s)`);
console.log('Excess kurtosis (0 = bell curve):', names.map((n, k) => `${n} ${kurt(S[k]).toFixed(2)}`).join(', '));
console.log('Correlation between sources:', `K-H ${ICA.corr(S[0], S[1]).toFixed(4)}, K-C ${ICA.corr(S[0], S[2]).toFixed(4)}, H-C ${ICA.corr(S[1], S[2]).toFixed(4)}`);

const fmt = m => m.map((x, i) => `out${i + 1}=${x.src} |r|=${x.abs.toFixed(4)}`).join(', ');
let fails = 0;
function run(label, A, Ssub, subNames, seed) {
  const res = ICA.fastICA(ICA.mix(A, Ssub), { maxIter: 200, seed });
  const m = ICA.matchComponents(res.Y, Ssub).map(x => Object.assign(x, { src: subNames[x.source] }));
  const ok = new Set(m.map(x => x.source)).size === Ssub.length && m.every(x => x.abs > 0.95);
  if (!ok) fails++;
  console.log(`${label} det=${ICA.det(A).toFixed(3)} rounds=${res.iterations} ${fmt(m)} ${ok ? 'PASS' : 'FAIL'}`);
  return m;
}

console.log('\n3 sounds, 3 mics: 12 random mixes (weights 0.1-1.0)');
const per3 = names.map(() => []);
for (let k = 0; k < 12; k++) {
  const seed = 3 + 7 * k, r = ICA.rng(seed); let A;
  do { A = [0, 1, 2].map(() => [0, 1, 2].map(() => 0.1 + 0.9 * r())); } while (Math.abs(ICA.det(A)) < 0.05);
  run(`mix ${String(k + 1).padStart(2)}`, A, S, names, seed + 1).forEach(x => per3[x.source].push(x.abs));
}
const summary = per => per.map((v, k) => `${names[k]} min ${Math.min(...v).toFixed(4)} / mean ${(v.reduce((a, b) => a + b, 0) / v.length).toFixed(4)}`).join('; ');
console.log('Per-source |r| over the 12 mixes:', summary(per3));
run('page default 3x3 mix', ICA.DEFAULT_MIX, S, names, 1);

console.log('\n2 sounds, 2 mics: every pair, 6 random mixes each');
const per2 = names.map(() => []);
[[0, 1], [0, 2], [1, 2]].forEach(([a, b]) => {
  for (let k = 0; k < 6; k++) {
    const seed = 101 + 13 * k + 7 * a + b, r = ICA.rng(seed); let A;
    do { A = [0, 1].map(() => [0, 1].map(() => 0.1 + 0.9 * r())); } while (Math.abs(ICA.det(A)) < 0.05);
    run(`${names[a]}+${names[b]} mix ${k + 1}`, A, [S[a], S[b]], [names[a], names[b]], seed + 1)
      .forEach(x => per2[[a, b][x.source]].push(x.abs));
  }
});
console.log('Per-source |r| over the 18 two-sound mixes:', summary(per2));
run('visualizer default (Kitchen+Classroom, 2x2)', ICA.DEFAULT_MIX2, [S[0], S[2]], [names[0], names[2]], 1);

console.log(`\n${fails === 0 ? 'ALL PASS' : fails + ' FAILED'} (pass = every output matches a different source with |r| > 0.95)`);
process.exit(fails ? 1 : 0);
