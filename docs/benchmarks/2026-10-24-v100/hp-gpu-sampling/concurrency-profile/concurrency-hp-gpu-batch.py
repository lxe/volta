#!/usr/bin/env python3
"""Profile the existing idle HP GPU-sampling server at one, four and eight submitted clients."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--name', required=True)
ap.add_argument('--case-timeout', type=int, default=120)
ap.add_argument('--allow-queued', action='store_true', help='explicitly permit eight clients on fewer active slots')
a = ap.parse_args()
out = ROOT / 'screens' / a.name
out.mkdir(parents=True, exist_ok=False)
with urllib.request.urlopen('http://127.0.0.1:8091/stats', timeout=10) as f:
    initial = json.load(f)
assert initial['active'] == 0 and initial['queued'] == 0
assert initial['ctx'] in [131072, 262144]
assert initial['capacity']==4 and a.allow_queued and initial['heads'] and initial['ctx']==262144
assert time.time()<1792906800, 'Full six-case HP profile must begin before05:40UTC'
assert (ROOT/'receipts/HP_GPU_PROMOTION_GATE_PASSED').exists()
(out / 'initial-stats.json').write_text(json.dumps(initial, indent=2))
(out / 'harness-sha256.txt').write_text(hashlib.sha256((ROOT / 'screen-timed.py').read_bytes()).hexdigest() + '\n')
(out / 'batch-configuration.json').write_text(json.dumps({'name': a.name, 'context': initial['ctx'],
    'heads': initial['heads'], 'active_capacity': initial['capacity'], 'allow_queued': a.allow_queued,
    'batch_source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}, indent=2))
cases = [(1, 'a'), (4, 'a'), (8, 'a'), (8, 'b'), (4, 'b'), (1, 'b')]
for clients, repeat in cases:
    assert a.case_timeout==120
    remaining=int(1792907520-time.time())
    assert remaining>=10, 'No new profile case within ten seconds of05:52UTC'
    case_timeout=min(a.case_timeout,remaining)
    name = f'clients-{clients}-{repeat}'
    cmd = [str(ROOT / 'venv/bin/python'), str(ROOT / 'screen-timed.py'), '--clients', str(clients),
           '--input', '4096', '--output', '1024', '--timeout',str(case_timeout), '--nonce', f'concurrency-{clients}-{repeat}',
           '--label', a.name, '--out', str(out / name)]
    with (out / (name + '.log')).open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    summary=json.loads((out/name/'summary.json').read_text())
    assert summary['all_hit_output_budget'] and summary['cached_tokens_delta']==0, 'Partial or cached profile: preserve evidence, no retry'
    print(json.dumps({'case': name, **summary}), flush=True)
(out / 'BATCH_DONE').write_text(str(time.time()) + '\n')
