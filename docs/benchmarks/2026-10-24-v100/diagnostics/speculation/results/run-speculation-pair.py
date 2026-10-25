#!/usr/bin/env python3
"""Run the single budgeted threshold hypothesis in fresh-server A/B/B/A order."""
import pathlib,subprocess,json,time
r=pathlib.Path(__file__).parent
assert (r/'speculation-budget-review.md').exists()
def execute(name,theta):
 args=['--engine','sampler','--name',name,'--ctx','131072','--slots','1','--graphs','4','--theta',str(theta),'--spec-k','5']
 subprocess.run(['python',str(r/'server-control.py'),*args],check=True)
 if name=='spec-a1':
  # The reviewed candidate CPU build may overlap this first model load, never timed inference.
  deadline=time.time()+240
  while not (r/'HP_SAMPLING_BUILD_COMPLETE').exists():
   assert time.time()<deadline,'Candidate build did not finish; inspect before admitting timing'
   time.sleep(2)
 subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--reserve','192','--admit-free','384'],check=True)
 (r/'screens'/name).mkdir(exist_ok=False)
 subprocess.run(['python',str(r/'run-arm.py'),*args,'--pair','spec-theta-01','--resume-existing'],check=True)
 subprocess.run([str(r/'venv/bin/python'),str(r/'screen.py'),'--clients','4','--input','4096','--output','1024','--nonce','spec-theta-01-long','--label',name,'--out',str(r/'screens'/name/'long')],check=True)
 for remote,local in [('server-'+name+'.log',r/'screens'/name/'server.log'),('memory-'+name+'.jsonl',r/'memory-guards'/name/'observations-through-arm.jsonl')]:
  subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/'+remote,str(local)],check=True)
 data={p:json.loads((r/'screens'/name/p/'summary.json').read_text()) for p in ['fixed','mixed','long']}
 for s in data.values():assert s['all_hit_output_budget'] and s['cached_tokens_delta']==0
 print(json.dumps({'name':name,'theta':theta,'tps':{p:s['output_tokens_per_wall_s'] for p,s in data.items()}}),flush=True)
 return data
results={}
for name,theta in [('spec-a1',.35),('spec-b1',.2),('spec-b2',.2),('spec-a2',.35)]:
 assert time.time()<1792907606
 results[name]=execute(name,theta)
 (r/'receipts/speculation-progress.json').write_text(json.dumps(results,indent=2))
 if name=='spec-b1':
  ratios={p:results[name][p]['output_tokens_per_wall_s']/results['spec-a1'][p]['output_tokens_per_wall_s'] for p in ['fixed','mixed','long']}
  if ratios['long']<.97 or ratios['mixed']<.97:
   (r/'receipts/SPECULATION_EARLY_KILLED').write_text(json.dumps(ratios,indent=2));raise SystemExit('Candidate clearly regressed; hypothesis killed, no sweep')
(r/'receipts/SPECULATION_ABBA_DONE').write_text(str(time.time())+'\n')
