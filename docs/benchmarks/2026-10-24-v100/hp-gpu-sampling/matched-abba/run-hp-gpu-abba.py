#!/usr/bin/env python3
"""Complete the precommitted HP A/B/B/A experiment after real audit and teacher gates."""
import pathlib,subprocess,json,time
r=pathlib.Path(__file__).parent
assert (r/'receipts/HP_GPU_AUDIT3_PASSED').exists() and (r/'receipts/HP_GPU_QUALITY3_PASSED').exists()
assert (r/'screens/hp-cpu-a1/ARM_DONE').exists()
proof=json.loads((r/'hp-sampling-v3-build-proof.json').read_text())
expected=proof['binaries_sha256']['/root/volta-hp-sampling-v3/target-hp-sampling-v3/release/volta-server']
for name,gpu in [('hp-gpu-b1',True),('hp-gpu-b2',True),('hp-cpu-a2',False)]:
 assert time.time()<1792907006,'Leave time for the gauntlet, capacity check and release'
 args=['--engine','hp-sample' if gpu else 'sampler','--name',name,'--ctx','262144','--slots','4','--head-parallel','--graphs','4']
 if gpu:args.append('--hp-gpu-sample')
 subprocess.run(['python',str(r/'server-control.py'),*args],check=True)
 cfg=json.loads((r/'server-runs'/name/'configuration.json').read_text())
 assert cfg['binary_sha256']==(expected if gpu else 'b9506fb2fd48ff71b95c855294b7cc0cab442fd18e0655ee617d35b69d0a61c4')
 assert not cfg['profile'] and not cfg['hp_sample_audit']
 subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--reserve','192','--admit-free','512'],check=True)
 (r/'screens'/name).mkdir(exist_ok=False)
 subprocess.run(['python',str(r/'run-arm.py'),*args,'--pair','hp-gpu-01','--resume-existing'],check=True)
 for remote,local in [('server-'+name+'.log',r/'screens'/name/'server.log'),('memory-'+name+'.jsonl',r/'memory-guards'/name/'observations-through-arm.jsonl')]:
  subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/'+remote,str(local)],check=True)
 data={p:json.loads((r/'screens'/name/p/'summary.json').read_text()) for p in ['fixed','mixed']}
 for s in data.values():assert s['all_hit_output_budget'] and s['cached_tokens_delta']==0 and s['output_tokens']==2048
 print(json.dumps({'arm':name,'gpu_sampling':gpu,'tps':{p:s['output_tokens_per_wall_s'] for p,s in data.items()}}),flush=True)
 if name=='hp-gpu-b1':
  ratios={p:data[p]['output_tokens_per_wall_s']/json.loads((r/'screens/hp-cpu-a1'/p/'summary.json').read_text())['output_tokens_per_wall_s'] for p in ['fixed','mixed']}
  if min(ratios.values())<.97:
   (r/'receipts/HP_GPU_ABBA_EARLY_KILLED').write_text(json.dumps(ratios,indent=2));raise SystemExit('HP GPU candidate clearly regressed; no further repeats')
(r/'receipts/HP_GPU_ABBA_DONE').write_text(str(time.time())+'\n')
