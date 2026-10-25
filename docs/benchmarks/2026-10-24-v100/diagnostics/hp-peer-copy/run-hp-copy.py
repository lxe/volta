import subprocess,pathlib,json,time,urllib.request,statistics
r=pathlib.Path(__file__).resolve().parents[1]; out=r/'lab/hp-copy-run'
def ssh(cmd): return subprocess.run(['ssh','-F',str(r/'ssh.conf'),'volta-run',cmd],capture_output=True,text=True,timeout=80)
def stats():
 with urllib.request.urlopen('http://127.0.0.1:8091/stats',timeout=10) as f: return json.load(f)
s=stats(); assert s['active']==s['queued']==0 and s['heads'] and s['ctx']==262144
check=ssh("test $(readlink /proc/19770/exe) = /root/volta-loop-prof-v3/target-v3/release/volta-server && sha256sum /root/hp-copy-20261024/hp_pitched_copy && nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
(out/'launch-check.txt').write_text(check.stdout+check.stderr); assert check.returncode==0
assert check.stdout.split()[0]=='5f30685eba565b2a33657731066c88dac7d52e8b42026f1d1fe38a0e3d0a9c08'
assert all(int(x.strip())>=1024 for x in check.stdout.splitlines()[1:])
assert not (out/'result.json').exists()
t=time.time(); p=ssh('/usr/bin/timeout --signal=KILL 60s /root/hp-copy-20261024/hp_pitched_copy')
(out/'stdout.jsonl').write_text(p.stdout); (out/'stderr.txt').write_text(p.stderr)
post=ssh("test $(readlink /proc/19770/exe) = /root/volta-loop-prof-v3/target-v3/release/volta-server && nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits && ps -eo pid,comm | awk '$2==\"hp_pitched_copy\" {print}'")
(out/'postcheck.txt').write_text(post.stdout+post.stderr); a=stats(); (out/'after-stats.json').write_text(json.dumps(a,indent=2))
rows=[json.loads(x) for x in p.stdout.splitlines()]; samples=[x for x in rows if x['kind']=='sample']
report={'exit_code':p.returncode,'wall_s':time.time()-t,'postcheck_exit':post.returncode,'model_pid_preserved':post.returncode==0,'model_idle':a['active']==a['queued']==0,'model_requests_unchanged':a['requests']==s['requests'],'shapes':{str(n):{'count':len(v:=[x['copy_event_ms'] for x in samples if x['rows']==n]),'median_ms':statistics.median(v),'max_ms':max(v)} for n in (1,3,12,64) if any(x['rows']==n for x in samples)},'success':[x for x in rows if x['kind']=='success']}
(out/'result.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
assert p.returncode==0 and report['success'] and report['model_idle'] and report['model_requests_unchanged'] and post.returncode==0
assert len(post.stdout.splitlines())==8 and min(int(x) for x in post.stdout.splitlines())>=1681
