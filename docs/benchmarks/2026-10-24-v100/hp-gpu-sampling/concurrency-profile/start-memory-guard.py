#!/usr/bin/env python3
"""Start the reserve guard for one already-ready owned rental server."""
import argparse
import json
import pathlib
import re
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--name', required=True)
ap.add_argument('--receipt-name')
ap.add_argument('--reserve', type=int, default=192)
ap.add_argument('--duration', type=int, default=1800)
ap.add_argument('--admit-free', type=int, default=512)
a = ap.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+', a.name)
receipt_name = a.receipt_name or a.name
assert re.fullmatch(r'[a-zA-Z0-9_-]+', receipt_name)
out = ROOT / 'memory-guards' / receipt_name
out.mkdir(parents=True, exist_ok=False)
cfg = json.loads((ROOT / 'server-runs' / a.name / 'configuration.json').read_text())
with urllib.request.urlopen('http://127.0.0.1:8091/stats', timeout=10) as f:
    stats = json.load(f)
assert stats['ctx'] == cfg['ctx'] and stats['heads'] == cfg['head_parallel']
assert stats['active'] == 0 and stats['queued'] == 0
assert 60 <= a.duration <= 3600
until = min(time.time() + a.duration, 1792908986)
code = '''import pathlib,json,os,subprocess,time
cfg=json.loads(pathlib.Path('/root/current-server.json').read_text())
assert cfg['name']==@@NAME@@
assert os.readlink('/proc/'+str(cfg['pid'])+'/exe')==cfg['binary']
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
rows=[[int(x.strip()) for x in line.split(',')] for line in raw.splitlines()]
assert len(rows)==8 and min(r[1] for r in rows)>=@@ADMIT@@,rows
cmd=['python3','/root/memory-guard.py','--name',cfg['name'],'--receipt-name',@@RECEIPT@@,'--min-free-mib',str(@@RESERVE@@),'--until',str(@@UNTIL@@)]
log='/root/memory-guard-'+@@RECEIPT@@+'.log'
p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=open(log,'x'),stderr=subprocess.STDOUT,start_new_session=True)
observation=pathlib.Path('/root/memory-'+@@RECEIPT@@+'.jsonl')
for attempt in range(100):
 if p.poll() is not None:raise RuntimeError('guard exited before first observation: '+pathlib.Path(log).read_text()[-2000:])
 if observation.exists() and observation.stat().st_size:
  first=json.loads(observation.read_text().splitlines()[0])
  assert first['pid']==cfg['pid'] and not first['stop'],first
  break
 time.sleep(.1)
else:raise RuntimeError('guard did not produce its first observation')
print(json.dumps({'first_observation_verified':True,'observation_path':str(observation),'server_pid':cfg['pid'],'guard_pid':p.pid,'command':cmd,'ready_free_mib':rows,'log':log,'until':@@UNTIL@@}))
'''
for key, value in {'NAME': a.name, 'RECEIPT': receipt_name, 'ADMIT': a.admit_free, 'RESERVE': a.reserve, 'UNTIL': until}.items():
    code = code.replace('@@' + key + '@@', repr(value))
q = subprocess.run(['ssh', '-F', str(ROOT / 'ssh.conf'), 'volta-run', 'python3 -'],
                   input=code, text=True, capture_output=True)
(out / 'launch-stderr.txt').write_text(q.stderr)
assert q.returncode == 0, q.stderr
result = json.loads(q.stdout)
(out / 'configuration.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
