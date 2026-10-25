#!/usr/bin/env python3
"""One predeclared same-V3 flag0/flag1 teacher pair. Parent launches only after review."""
import argparse,datetime,fcntl,hashlib,json,pathlib,re,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).parent;SSH=['ssh','-F',str(ROOT/'ssh.conf'),'volta-run']
HARD=datetime.datetime.fromisoformat('2026-10-25T06:16:26+00:00').timestamp()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',default='hp-v3-flag-pair');p.add_argument('--observe-seconds',type=int,default=450);a=p.parse_args()
 assert re.fullmatch('[a-zA-Z0-9_-]+',a.name);assert 420<=a.observe_seconds<=600
 assert (ROOT/'receipts/HP_GPU_MATCHED_TIMING_GATE_PASSED').exists(),'matched HP promotion gate marker required'
 assert (ROOT/'receipts/HP_GPU_AUDIT3_PASSED').exists() and (ROOT/'receipts/HP_GPU_QUALITY3_PASSED').exists()
 assert HARD-time.time()>=420,'insufficient seven-minute pair budget before06:16:26'
 lock=(ROOT/'server-transition.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((ROOT/'server-runs/hp-cpu-a2/configuration.json').read_text());assert cfg['name']=='hp-cpu-a2'
 proof=json.loads((ROOT/'hp-sampling-v3-build-proof.json').read_text());commit=(ROOT/'hp-sampling-v3-source-commit.txt').read_text().strip();assert commit==proof['source_commit']=='2e0e16166709d11f4e98802eb74343cdba77a605'
 cli='/root/volta-hp-sampling-v3/target-hp-sampling-v3/release/volta';digest=proof['binaries_sha256'][cli];assert digest=='bcfe792f228ba7216237c13e5d582dd7e4a34e0872ddd4c293c77d10a237cd67'
 model=json.loads((ROOT/'receipts/model-sha256.json').read_text());assert model['sha256']=='9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c'
 with urllib.request.urlopen('http://127.0.0.1:8091/stats',timeout=10) as f:stats=json.load(f)
 assert stats['active']==0 and stats['queued']==0
 out=ROOT/'teacher-runs'/a.name;out.mkdir(exist_ok=False)
 (out/'initial-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
 spec={'source_commit':commit,'source_root':'/root/volta-hp-sampling-v3','target':'/root/volta-hp-sampling-v3/target-hp-sampling-v3','source_sha256':proof['remote_source_sha256'],'cli':cli,'cli_sha256':digest,'fatbin_sha256':'4387f0e32b257dd12f30f5bc1d2fad266a0bd2405e53f92d0b54f9d321c3ddf5','model':'/root/model/DeepSeek-V4.1-Flash-1017-Volta.gguf','model_receipt':model,'a2':cfg,'baseline':json.loads((ROOT/'receipts/teacher-eval.jsonl').read_text())}
 worker=(ROOT/'teacher-hp-flag-pair-worker.py').read_text();remote='/root/teacher-'+a.name
 spec['worker_source_sha256']=hashlib.sha256(worker.encode()).hexdigest();(out/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
 # A one-shot attempt marker survives observation/SSH failures. Never retry launch after unknown outcome.
 with (ROOT/'receipts/HP_FLAG_PAIR_ATTEMPTED').open('x') as f:f.write(json.dumps({'name':a.name,'remote':remote,'epoch':time.time()})+'\n')
 launch=f'''import pathlib,subprocess,json,sys
r=pathlib.Path({remote!r});r.mkdir(exist_ok=False)
(r/'worker.py').write_text({worker!r})
(r/'specification.json').write_text({json.dumps(spec)!r})
p=subprocess.Popen([sys.executable,str(r/'worker.py'),str(r/'specification.json')],stdin=subprocess.DEVNULL,stdout=open(r/'worker.log','x'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'worker_pid':p.pid,'remote':str(r)}}))
'''
 try:q=subprocess.run(SSH+['python3 -'],input=launch,text=True,capture_output=True,timeout=20)
 except subprocess.TimeoutExpired:raise RuntimeError('Launch outcome unknown; preserve/inspect the remote worker, never duplicate it')
 (out/'launch-stdout.txt').write_text(q.stdout);(out/'launch-stderr.txt').write_text(q.stderr);assert q.returncode==0,q.stderr
 info=json.loads(q.stdout);(out/'worker.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
 end=min(time.time()+a.observe_seconds,HARD);last=None
 while time.time()<end:
  try:q=subprocess.run(SSH+['cat '+remote+'/state.json'],capture_output=True,text=True,timeout=15)
  except subprocess.TimeoutExpired:time.sleep(2);continue
  if q.returncode==0:
   state=json.loads(q.stdout);(out/'last-observed-state.json').write_text(json.dumps(state,indent=2)+'\n')
   if state['phase']!=last or state.get('planned_budget_exceeded'):
    print(json.dumps({'phase':state['phase'],'budget_exceeded':state.get('planned_budget_exceeded',False)}),flush=True);last=state['phase']
   if state['phase'] in ('COMPLETE','FAILED'):
    subprocess.run(['scp','-r','-l','1900000','-F',str(ROOT/'ssh.conf'),'volta-run:'+remote,str(out/'remote')],check=True)
    assert state['phase']=='COMPLETE',state.get('error')
    assert state['both_tight_gate_passed'],'paired quality failed; preserve both arms, no repeat'
    (ROOT/'receipts/HP_GPU_FLAG_PAIR_QUALITY_PASSED').write_text(json.dumps({'name':a.name,'arms':state['arms']},indent=2)+'\n');return 0
  time.sleep(5)
 raise RuntimeError('Observation timeout: preserve same live worker/CLI; inspect saved PID/state, do not restart or duplicate. Remote worker enforces06:16:26 cutoff.')
if __name__=='__main__':sys.exit(main())
