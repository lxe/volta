#!/usr/bin/env python3
"""Switch only this rental's known idle server after the frozen gauntlet finishes."""
import argparse
import fcntl
import json
import pathlib
import re
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).parent
SSH = ['ssh', '-F', str(ROOT / 'ssh.conf'), 'volta-run']


def stats():
    with urllib.request.urlopen('http://127.0.0.1:8091/stats', timeout=5) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', choices=['frozen', 'sampler', 'graph-diag', 'loop-prof', 'hp-sample'], required=True)
    ap.add_argument('--ctx', type=int, choices=[131072, 262144], default=131072)
    ap.add_argument('--chunk', type=int, choices=[32, 64], default=64)
    ap.add_argument('--slots', type=int, choices=[1, 4, 8], default=1)
    ap.add_argument('--graphs', type=int, choices=[0, 4, 8, 16, 32], default=4)
    ap.add_argument('--head-parallel', action='store_true')
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--loop-profile', type=int, choices=[0, 1, 64], default=0)
    ap.add_argument('--graph-trace', action='store_true', help='bounded CPU diagnostics; never primary timing')
    ap.add_argument('--hp-gpu-sample', action='store_true')
    ap.add_argument('--hp-sample-audit', type=int, default=0)
    ap.add_argument('--theta', type=float, default=0.35)
    ap.add_argument('--spec-k', type=int, choices=range(0,9), default=5)
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    transition_lock = (ROOT / 'server-transition.lock').open('w')
    fcntl.flock(transition_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not a.graph_trace or a.engine in ['graph-diag', 'loop-prof']
    assert not a.loop_profile or a.engine == 'loop-prof'
    assert 0 <= a.hp_sample_audit <= 4096
    assert not a.hp_gpu_sample or (a.engine == 'hp-sample' and a.head_parallel)
    assert not a.hp_sample_audit or a.hp_gpu_sample
    assert 0 < a.theta <= 1
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', a.name)
    assert (ROOT / 'baseline/GAUNTLET_DONE').exists(), 'frozen gauntlet must finish first'
    assert time.time() < 1792908986, 'rental deadline too close'
    out = ROOT / 'server-runs' / a.name
    assert not out.exists(), 'use a fresh run name'
    out.mkdir(parents=True)
    try:
        old = stats()
        assert old['active'] == 0 and old['queued'] == 0, 'server has active requests'
        (out / 'previous-stats.json').write_text(json.dumps(old, indent=2))
    except OSError as error:
        # A closed SSH-forwarded API can reset the socket. Absence must be
        # confirmed remotely; an observation error never authorizes restarting
        # a known live server that might still be loading or serving work.
        probe = '''import pathlib,os,json
allowed={'/root/volta/target/release/volta-server','/root/volta-candidate/target/release/volta-server','/root/volta-graph-diag-v2/target-v2/release/volta-server','/root/volta-loop-prof-v3/target-v3/release/volta-server','/root/volta-hp-sampling-v1/target-hp-sampling-v1/release/volta-server'}
live=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if os.readlink(p/'exe') in allowed:live.append(int(p.name))
 except OSError:pass
print(json.dumps(live))
'''
        observed = subprocess.run(SSH + ['python3 -'], input=probe, text=True, capture_output=True)
        (out / 'initial-api-observation.json').write_text(json.dumps({
            'api_error': repr(error), 'remote_returncode': observed.returncode,
            'known_live_servers': observed.stdout, 'remote_stderr': observed.stderr}, indent=2))
        assert observed.returncode == 0 and json.loads(observed.stdout) == [], 'API unavailable but a known server may be live; observe it before restarting'
    cfg = vars(a)
    cfg['binary'] = {'frozen': '/root/volta/target/release/volta-server',
                     'sampler': '/root/volta-candidate/target/release/volta-server',
                     'graph-diag': '/root/volta-graph-diag-v2/target-v2/release/volta-server',
                     'loop-prof': '/root/volta-loop-prof-v3/target-v3/release/volta-server',
                     'hp-sample': '/root/volta-hp-sampling-v1/target-hp-sampling-v1/release/volta-server'}[a.engine]
    cfg['model'] = '/root/model/DeepSeek-V4.1-Flash-1017-Volta.gguf'
    cfg['log'] = '/root/server-' + a.name + '.log'
    # Only the exact known executables can be stopped; other processes are left alone.
    remote = '''import pathlib,os,signal,time,json,subprocess,hashlib
cfg=CONFIG
allowed={'/root/volta/target/release/volta-server','/root/volta-candidate/target/release/volta-server','/root/volta-graph-diag-v2/target-v2/release/volta-server','/root/volta-loop-prof-v3/target-v3/release/volta-server','/root/volta-hp-sampling-v1/target-hp-sampling-v1/release/volta-server'}
owned=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  if os.readlink(p/'exe') in allowed: owned.append(int(p.name))
 except OSError: pass
assert len(owned)<=1,('multiple known servers',owned)
for pid in owned:
 os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  try:
   if pathlib.Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z': break
  except FileNotFoundError: break
  time.sleep(.1)
 else: raise RuntimeError('known idle server did not exit; inspect it before retrying')
for _ in range(100):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 if all(int(x.strip())<128 for x in raw.splitlines()): break
 time.sleep(.2)
else: raise RuntimeError('GPU memory remains in use after known server stopped')
env=os.environ.copy()
for k in list(env):
 if k.startswith('VOLTA_'): env.pop(k)
env.update(VOLTA_GRAPHS=str(cfg['graphs']),VOLTA_GPU_SAMPLE='1',VOLTA_MEGA='0',VOLTA_KV='fp4',VOLTA_OVERLAP='1',VOLTA_SHARED_TP='1',VOLTA_SHARED_EARLY='0')
env['LD_LIBRARY_PATH']='/usr/local/cuda-12.9/targets/x86_64-linux/lib:'+env.get('LD_LIBRARY_PATH','')
if cfg.get('hp_gpu_sample'):env['VOLTA_HP_GPU_SAMPLE']='1'
if cfg.get('hp_sample_audit'):env['VOLTA_HP_SAMPLE_AUDIT']=str(cfg['hp_sample_audit'])
if cfg['profile']:env['VOLTA_STEPT']='1'
if cfg.get('loop_profile',0):env['VOLTA_LOOP_PROF']=str(cfg['loop_profile'])
if cfg['graph_trace']:env['VOLTA_GRAPH_TRACE']='/root/graph-traces/'+cfg['name']
cmd=[cfg['binary'],'--model',cfg['model'],'--gpus','8','--slots',str(cfg['slots']),'--ctx',str(cfg['ctx']),'--chunk',str(cfg['chunk']),'--spec-k',str(cfg['spec_k']),'--theta',str(cfg['theta']),'--temperature','0.7','--top-p','1','--max-tokens','16384','--host','127.0.0.1','--port','8090','--admin-port','8091']
if cfg['head_parallel']:cmd.append('--head-parallel')
log=open(cfg['log'],'w')
p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
cfg.update(pid=p.pid,command=cmd,binary_sha256=hashlib.file_digest(open(cfg['binary'],'rb'),'sha256').hexdigest(),volta_environment={k:v for k,v in env.items() if k.startswith('VOLTA_')},started_epoch=time.time(),stopped_pids=owned)
pathlib.Path('/root/current-server.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps(cfg))
'''.replace('CONFIG', repr(cfg))
    switched = subprocess.run(SSH + ['python3 -'], input=remote, text=True, capture_output=True)
    (out / 'switch-stderr.txt').write_text(switched.stderr)
    assert switched.returncode == 0, switched.stderr
    result = json.loads(switched.stdout)
    (out / 'configuration.json').write_text(json.dumps(result, indent=2))
    deadline = min(time.time() + 480, 1792909046)
    polls = 0
    while time.time() < deadline:
        try:
            s = stats()
            assert s['ctx'] == a.ctx and s['heads'] == a.head_parallel
            assert s['capacity'] == (a.slots if a.head_parallel else 8 * a.slots)
            (out / 'ready-stats.json').write_text(json.dumps(s, indent=2))
            gpu = subprocess.check_output(SSH + ['nvidia-smi --query-gpu=index,name,memory.used,clocks.sm,clocks.mem,power.draw --format=csv'], text=True)
            (out / 'ready-gpus.csv').write_text(gpu)
            print(json.dumps({'ready': True, 'pid': result['pid'], 'run': str(out)}))
            return
        except (urllib.error.URLError, OSError, AssertionError):
            polls += 1
            if polls % 10 == 0:
                observed = subprocess.run(SSH + ['ps -o stat= -p ' + str(result['pid'])], text=True, capture_output=True)
                if observed.returncode == 1 or (observed.returncode == 0 and observed.stdout.lstrip().startswith('Z')):
                    log = subprocess.run(SSH + ['tail -80 ' + result['log']], text=True, capture_output=True)
                    (out / 'failed-server.log').write_text(log.stdout + log.stderr)
                    raise RuntimeError('new server exited; inspect ' + str(out / 'failed-server.log'))
            time.sleep(3)
    raise TimeoutError('server did not become ready; inspect ' + result['log'])


if __name__ == '__main__':
    main()
