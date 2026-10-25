from pathlib import Path
import json,statistics,collections,re
r=Path('/home/lxe/volta-sixhour-20261024');run=r/'graph-traces/diag-v2-stept-001'
pat=re.compile(r'^GPU(\d+) step at ([\d.]+): begin ([\d.]+) ms, layers ([\d.]+), finish ([\d.]+)$')
logs=collections.defaultdict(list)
for line in (run/'server.log').read_text().splitlines():
 m=pat.match(line.strip())
 if m: logs[int(m[1])].append({'at_ms':float(m[2]),'begin_ms':float(m[3]),'layers_ms':float(m[4]),'finish_ms':float(m[5])})
steps=collections.defaultdict(dict)
for g in range(8):
 a=[json.loads(l) for l in (run/f'gpu{g}.jsonl').read_text().splitlines()];a=[x for x in a if x['kind']=='step']; pos=[x for x in a if x['t']>0]
 assert len(logs[g])>=len(pos)
 joined={x['seq']:y for x,y in zip(pos,logs[g])}
 for x in a:steps[x['seq']][g]={**x,**joined.get(x['seq'],{})}
def dist(v):return {'n':len(v),'min':min(v),'median':statistics.median(v),'mean':statistics.mean(v),'max':max(v)} if v else None
out={}
for phase,lo,hi in [('fixed',100,393),('mixed',394,768)]:
 pure=[]
 for seq,gs in steps.items():
  if not lo<=seq<=hi or len(gs)!=8:continue
  active=[v for v in gs.values() if v['t']>0]
  if len(active)!=4 or not all(1<=v['t']<=6 for v in active):continue
  pure.extend([{**v,'global_t':sum(x['t'] for x in active),'seq':seq} for v in active])
 groups={}
 for t in range(1,7):
  rows=[x for x in pure if x['t']==t]
  groups[str(t)]={'n':len(rows),'main_phase_sum_ms':dist([x['begin_ms']+x['layers_ms']+x['finish_ms'] for x in rows]),'layers_ms':dist([x['layers_ms'] for x in rows])}
 byglobal={}
 for total in sorted(set(x['global_t'] for x in pure)):
  seqs=sorted({x['seq'] for x in pure if x['global_t']==total})
  times=[max(x['begin_ms']+x['layers_ms']+x['finish_ms'] for x in pure if x['seq']==s) for s in seqs]
  byglobal[str(total)]={'critical_gpu_phase_sum_ms':dist(times)}
 out[phase]={'local_t':groups,'global_t':byglobal,'pure_four_client_steps':len(pure)//4}
print(json.dumps(out,indent=2));(r/'speculation-timing-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
