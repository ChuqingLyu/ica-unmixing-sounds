const ICA = require('../ica.js');
const S = ICA.makeSources();
const names=['melody','beat','siren'];
let fails=0;
for (const seed of [11,22,33,44,55]) {
  const r = ICA.rng(seed); let A;
  do { A=[0,1,2].map(()=>[0,1,2].map(()=>0.1+0.9*r())); } while (Math.abs(ICA.det3(A))<0.05);
  const X = ICA.mix(A,S);
  const t0=Date.now();
  const res = ICA.fastICA(X,{maxIter:200,seed});
  const m = ICA.matchComponents(res.Y,S);
  const distinct = new Set(m.map(x=>x.source)).size===3;
  const ok = distinct && m.every(x=>x.abs>0.9);
  if(!ok) fails++;
  console.log(`seed ${seed} det=${ICA.det3(A).toFixed(3)} iters=${res.iterations} conv=${res.converged} ${Date.now()-t0}ms ->`,
    m.map((x,i)=>`IC${i+1}≈S${x.source+1}(${names[x.source]}) r=${x.corr.toFixed(6)}`).join(', '), ok?'PASS':'FAIL');
}
// default matrix from page
const A=[[1.0,0.6,0.4],[0.5,1.0,0.7],[0.3,0.5,1.0]];
const res=ICA.fastICA(ICA.mix(A,S),{seed:1}); const m=ICA.matchComponents(res.Y,S);
console.log('default A iters',res.iterations, m.map((x,i)=>`IC${i+1}≈S${x.source+1} |r|=${x.abs.toFixed(6)}`).join(', '));
process.exit(fails?1:0);
