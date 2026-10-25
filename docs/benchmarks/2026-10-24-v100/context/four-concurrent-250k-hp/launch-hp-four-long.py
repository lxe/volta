#!/usr/bin/env python3
"""Run the predeclared four-client 250K retrieval after all HP promotion gates."""
import datetime,json,pathlib,subprocess,time,urllib.request
r=pathlib.Path(__file__).parent
assert (r/'receipts/HP_GPU_PROMOTION_GATE_PASSED').exists()
assert time.time()<datetime.datetime.fromisoformat('2026-10-25T05:10:00+00:00').timestamp()
name='hp-gpu-capacity4';out=r/'screens/four-hp-250k'
assert not out.exists()
args=['--engine','hp-sample','--name',name,'--ctx','262144','--slots','4','--head-parallel','--graphs','4','--hp-gpu-sample']
subprocess.run(['python',str(r/'server-control.py'),*args],check=True)
cfg=json.loads((r/'server-runs'/name/'configuration.json').read_text());proof=json.loads((r/'hp-sampling-v3-build-proof.json').read_text())
assert cfg['binary_sha256']==proof['binaries_sha256'][cfg['binary']]
assert not cfg['profile'] and not cfg['hp_sample_audit']
assert time.time()<datetime.datetime.fromisoformat('2026-10-25T05:10:00+00:00').timestamp()
subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--reserve','192','--admit-free','512','--duration','3600'],check=True)
cmd=[str(r/'venv/bin/python'),str(r/'screen-four-hp-long.py'),'--out',str(out),'--label',name,'--clients','4','--input','250000','--output','64','--needle','--timeout','2400']
try:
 subprocess.run(cmd,check=True)
finally:
 out.mkdir(exist_ok=True)
 (out/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
 for remote,local in [('server-'+name+'.log',out/'server.log'),('memory-'+name+'.jsonl',out/'memory-observations.jsonl')]:
  subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/'+remote,str(local)],check=True)
print('Four HP long-context client completed; independently validate tokens, keys, active sessions and guard coverage.',flush=True)
