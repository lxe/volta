#!/usr/bin/env python3
"""Profile one/four/eight clients on the qualified HP server after capacity completes."""
import datetime,json,pathlib,subprocess,time,urllib.request
r=pathlib.Path(__file__).parent
assert (r/'receipts/HP_GPU_PROMOTION_GATE_PASSED').exists()
assert time.time()<datetime.datetime.fromisoformat('2026-10-25T05:40:00+00:00').timestamp()
x=json.loads((r/'screens/four-hp-250k/summary.json').read_text())
assert x['completed']==4 and x['needle_verified'] and x['cached_tokens_delta']==0
name='hp-gpu-capacity4';cfg=json.loads((r/'server-runs'/name/'configuration.json').read_text())
with urllib.request.urlopen('http://127.0.0.1:8091/stats',timeout=10) as f:s=json.load(f)
assert s['active']==0 and s['queued']==0 and s['heads'] and s['ctx']==262144
out=r/'screens/hp-gpu-concurrency';assert not out.exists()
subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--receipt-name','hp-gpu-concurrency','--reserve','192','--admit-free','512','--duration','3600'],check=True)
try:
 subprocess.run(['python',str(r/'concurrency-hp-gpu-batch.py'),'--name','hp-gpu-concurrency','--allow-queued','--case-timeout','120'],check=True)
finally:
 out.mkdir(exist_ok=True)
 (out/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
 (out/'prior-work.txt').write_text('Same four-slot HP GPU-sampler process as the successful four-client 250K retrieval. Cache/graph history retained; individual short cases use distinct cold request nonces. Descriptive profile; no controlled cross-configuration gain claimed. Eight submissions queue behind four active slots.\n')
 for remote,local in [('server-'+name+'.log',out/'server.log'),('memory-hp-gpu-concurrency.jsonl',out/'memory-observations.jsonl')]:
  subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/'+remote,str(local)],check=True)
