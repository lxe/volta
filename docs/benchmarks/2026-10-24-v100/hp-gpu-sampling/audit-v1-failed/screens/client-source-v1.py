#!/usr/bin/env python3
"""Run bounded real-model audit workloads against an admitted, guarded HP candidate."""
import argparse,concurrent.futures,fcntl,json,pathlib,threading,time
from screen import get_stats,run_request
r=pathlib.Path(__file__).parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',required=True);a=ap.parse_args()
out=r/'screens'/a.name/'audit';out.mkdir(exist_ok=False)
lock=(r/'screen.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=json.loads((r/'server-runs'/a.name/'configuration.json').read_text())
assert cfg['head_parallel'] and cfg['ctx']==262144 and cfg['slots']==4
assert cfg['volta_environment']['VOLTA_HP_GPU_SAMPLE']=='1'
assert int(cfg['volta_environment']['VOLTA_HP_SAMPLE_AUDIT'])>=1024
assert (r/'memory-guards'/a.name/'configuration.json').exists()
fixtures=json.loads((r/'screen-fixtures.json').read_text())
url='http://127.0.0.1:8090/v1/chat/completions';admin='http://127.0.0.1:8091/stats'
end=min(time.time()+720,1792908986)
def body(n,budget,temperature,i,**extra):
 prompt=(f'Audit work item {a.name}, case {i}. Review this Python module.\n'+fixtures[str(n)]['text']+'\nWrite a detailed code review and revised implementation. Keep writing until the output budget is exhausted.')
 return {'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':1,'seed':32000+i,'max_tokens':budget,'reasoning_effort':'none','stream':True,'stream_options':{'include_usage':True},**extra}
def phase(label,bodies,stagger=False):
 p=out/label;p.mkdir();initial=get_stats(admin);assert initial['active']==initial['queued']==0
 (p/'requests.json').write_text(json.dumps(bodies,indent=2));(p/'initial-stats.json').write_text(json.dumps(initial,indent=2))
 starts=[threading.Event() for _ in bodies];firsts=[threading.Event() for _ in bodies];rows=[]
 deadline=min(time.time()+240,end);assert deadline>time.time()+15
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
  fs=[pool.submit(run_request,url,b,starts[i],firsts[i],deadline) for i,b in enumerate(bodies)]
  for e in starts[:3 if stagger else len(starts)]:e.set()
  admitted=not stagger
  while not all(f.done() for f in fs):
   if not admitted and all(e.is_set() for e in firsts[:3]):starts[3].set();admitted=True
   if time.time()>deadline:
    for e in starts:e.set()
   s=get_stats(admin);s.pop('history',None);s['observed_epoch']=time.time();rows.append(s);time.sleep(.25)
  results=[f.result() for f in fs]
 final=get_stats(admin)
 (p/'results.json').write_text(json.dumps(results,indent=2));(p/'stats.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(p/'final-stats.json').write_text(json.dumps(final,indent=2))
 assert all('error' not in x for x in results),results
 assert final['active']==final['queued']==0
 summary={'phase':label,'requests':len(bodies),'prompt_tokens':sum(x['usage']['prompt_tokens'] for x in results),'output_tokens':sum(x['usage']['completion_tokens'] for x in results),'max_active':max([s['active'] for s in rows] or [0]),'max_queued':max([s['queued'] for s in rows] or [0]),'cached_tokens_delta':final['tokens_cached']-initial['tokens_cached']}
 (p/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
phase('sampled-prefill-decode',[body(512,128,.7,0)])
phase('mixed-greedy-sampled',[body(512,256,t,10+i) for i,t in enumerate([0,.7,0,.7])])
phase('top-p-fallback',[body(512,64,.7,30,top_p=.9)])
phase('hesitation-fallback',[body(512,64,.7,31,hesitation_penalty=1.0,reasoning_effort='low')])
phase('pure-greedy',[body(512,64,0,32)])
phase('staggered-prefill',[body(8192 if i==3 else 512,512,.7,20+i) for i in range(4)],True)
(out/'CLIENT_DONE').write_text('API workloads completed. Audit log coverage and parity still require review.\n')
