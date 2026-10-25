#!/usr/bin/env python3
"""Run one fresh-server arm with an identical smoke, warmup and fixed/mixed trace."""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--engine', choices=['frozen', 'sampler', 'graph-diag', 'loop-prof', 'hp-sample'], required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--pair', default='guard-01')
ap.add_argument('--hp-gpu-sample', action='store_true')
ap.add_argument('--hp-sample-audit', type=int, default=0)
ap.add_argument('--theta', type=float, default=0.35)
ap.add_argument('--spec-k', type=int, choices=range(0,9), default=5)
ap.add_argument('--resume-existing', action='store_true', help='observe an already started fresh arm; never restart it')
ap.add_argument('--graph-trace', action='store_true')
ap.add_argument('--ctx', type=int, choices=[131072, 262144], default=131072)
ap.add_argument('--slots', type=int, choices=[1, 4, 8], default=1)
ap.add_argument('--graphs', type=int, choices=[0, 4, 8, 16, 32], default=4)
ap.add_argument('--head-parallel', action='store_true')
ap.add_argument('--chunk', type=int, choices=[32, 64], default=64)
ap.add_argument('--profile', action='store_true', help='enable synchronizing STEPT diagnostics; timing is not primary evidence')
ap.add_argument('--loop-profile', type=int, choices=[0, 1, 64], default=0)
a = ap.parse_args()
assert not a.graph_trace or a.engine in ['graph-diag', 'loop-prof']
assert (ROOT / 'baseline/GAUNTLET_DONE').exists()
assert time.time() < 1792907606, 'leave time for receipts and rental release'
out = ROOT / 'screens' / a.name
if a.resume_existing:
    cfg = json.loads((ROOT / 'server-runs' / a.name / 'configuration.json').read_text())
    assert cfg['engine'] == a.engine and cfg['ctx'] == a.ctx and cfg['graphs'] == a.graphs
    assert cfg['slots'] == a.slots and cfg['head_parallel'] == a.head_parallel
    assert cfg['chunk'] == a.chunk and cfg['profile'] == a.profile
    assert cfg.get('hp_gpu_sample',False) == a.hp_gpu_sample and cfg.get('hp_sample_audit',0) == a.hp_sample_audit
    assert cfg.get('loop_profile',0) == a.loop_profile
    assert cfg.get('theta',0.35) == a.theta and cfg.get('spec_k',5) == a.spec_k
    assert out.is_dir() and not (out / 'smoke.json').exists(), 'resume only a fresh arm before any smoke'
    deadline = min(cfg['started_epoch'] + 480, 1792908986)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8091/stats', timeout=5) as f:
                s = json.load(f)
            assert s['requests'] == 0 and s['ctx'] == a.ctx and s['capacity'] == a.slots * (1 if a.head_parallel else 8) and s['heads'] == a.head_parallel
            (ROOT / 'server-runs' / a.name / 'ready-stats.json').write_text(json.dumps(s, indent=2))
            break
        except OSError:
            time.sleep(3)
    else:
        raise TimeoutError('existing server not ready; inspect its known PID and log')
else:
    out.mkdir(parents=True, exist_ok=False)
    control = [sys.executable, str(ROOT / 'server-control.py'), '--engine', a.engine,
                    '--ctx', str(a.ctx), '--chunk', str(a.chunk), '--slots', str(a.slots), '--graphs', str(a.graphs),
                    '--name', a.name, '--theta', str(a.theta), '--spec-k', str(a.spec_k)]
    if a.hp_gpu_sample:
        control.append('--hp-gpu-sample')
    if a.hp_sample_audit:
        control.extend(['--hp-sample-audit',str(a.hp_sample_audit)])
    if a.graph_trace:
        control.append('--graph-trace')
    if a.head_parallel:
        control.append('--head-parallel')
    if a.profile:
        control.append('--profile')
    if a.loop_profile:
        control.extend(['--loop-profile',str(a.loop_profile)])
    subprocess.run(control, check=True)
body = {'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content':
        'Write one sentence confirming that you can respond.'}], 'temperature': 0, 'max_tokens': 32}
req = urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions',
      headers={'Content-Type': 'application/json'}, data=json.dumps(body).encode())
with urllib.request.urlopen(req, timeout=120) as f:
    smoke = json.load(f)
assert smoke.get('choices') and smoke['usage']['completion_tokens'] > 0
(out / 'smoke.json').write_text(json.dumps(smoke, indent=2))
for suffix, inp, output, mixed in [('warm1', 512, 64, False), ('warm2', 512, 64, False),
                                  ('fixed', 4096, 512, False), ('mixed', 4096, 512, True)]:
    cmd = [str(ROOT / 'venv/bin/python'), str(ROOT / 'screen.py'), '--clients', '4',
           '--input', str(inp), '--output', str(output), '--nonce', a.pair + '-' + suffix,
           '--label', a.engine + '-' + a.name, '--out', str(out / suffix)]
    if mixed:
        cmd.append('--mixed')
    with (out / (suffix + '.log')).open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    summary = json.loads((out / suffix / 'summary.json').read_text())
    print(json.dumps({'arm': a.name, 'trace': suffix, **summary}), flush=True)
gpu = subprocess.check_output(['ssh', '-F', str(ROOT / 'ssh.conf'), 'volta-run',
      'nvidia-smi --query-gpu=index,memory.used,memory.free,power.draw,clocks.sm --format=csv'], text=True)
(out / 'final-gpus.csv').write_text(gpu)
(out / 'ARM_DONE').write_text(str(time.time()) + '\n')
