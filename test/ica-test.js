const ICA = require('../ica.js');
const S = ICA.makeSources();
const names=['piano','drums','bass'];
console.log('N',S[0].length,'SR',ICA.SR,'source corr p-d',ICA.corr(S[0],S[1]).toFixed(4),'p-b',ICA.corr(S[0],S[2]).toFixed(4),'d-b',ICA.corr(S[1],S[2]).toFixed(4));
let fails=0,minAbs=1, runs=0;
const seeds=[...Array(12).keys()].map(i=>i*7+3);
for (const seed of seeds) {
  const r = ICA.rng(seed); let A;
  do { A=[0,1,2].map(()=>[0,1,2].map(()=>0.1+0.9*r())); } while (Math.abs(ICA.det3(A))<0.05);
  const res = ICA.fastICA(ICA.mix(A,S),{maxIter:200,seed:seed+1});
  const m = ICA.matchComponents(res.Y,S);
  const ok = new Set(m.map(x=>x.source)).size===3 && m.every(x=>x.abs>0.9);
  runs++; if(!ok) fails++; m.forEach(x=>minAbs=Math.min(minAbs,x.abs));
  console.log(`seed ${seed} det=${ICA.det3(A).toFixed(3)} iters=${res.iterations} conv=${res.converged} ->`,
    m.map((x,i)=>`IC${i+1}≈${names[x.source]} r=${x.corr.toFixed(4)}`).join(', '), ok?'PASS':'FAIL');
}
console.log(`runs ${runs} fails ${fails} min|r| ${minAbs.toFixed(4)}`);
const A0=[[1.0,0.35,0.45],[0.3,1.0,0.4],[0.6,0.55,0.8]];
const r0=ICA.fastICA(ICA.mix(A0,S),{seed:1}); const m0=ICA.matchComponents(r0.Y,S);
console.log('default page mix det',ICA.det3(A0).toFixed(3),'iters',r0.iterations, m0.map((x,i)=>`IC${i+1}≈${names[x.source]} r=${x.corr.toFixed(4)}`).join(', '));
process.exit(fails?1:0);
