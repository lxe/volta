#!/usr/bin/env python3
"""Screen existing idle 128K DP server at one/four/eight clients, then real retrieval."""
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
ap.add_argument('--needle', action='store_true')
ap.add_argument('--needle-input', type=int, choices=[126000, 250000], default=126000)
ap.add_argument('--allow-queued', action='store_true', help='explicitly permit eight clients on fewer active slots')
a = ap.parse_args()
out = ROOT / 'screens' / a.name
out.mkdir(parents=True, exist_ok=False)
with urllib.request.urlopen('http://127.0.0.1:8091/stats', timeout=10) as f:
    initial = json.load(f)
assert initial['active'] == 0 and initial['queued'] == 0
assert initial['ctx'] in [131072, 262144]
assert initial['capacity'] >= 8 or a.allow_queued
(out / 'initial-stats.json').write_text(json.dumps(initial, indent=2))
(out / 'harness-sha256.txt').write_text(hashlib.sha256((ROOT / 'screen-timed.py').read_bytes()).hexdigest() + '\n')
(out / 'batch-configuration.json').write_text(json.dumps({'name': a.name, 'context': initial['ctx'],
    'heads': initial['heads'], 'active_capacity': initial['capacity'], 'allow_queued': a.allow_queued,
    'batch_source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}, indent=2))
cases = [(1, 'a'), (4, 'a'), (8, 'a'), (8, 'b'), (4, 'b'), (1, 'b')]
for clients, repeat in cases:
    assert time.time() < 1792907606
    name = f'clients-{clients}-{repeat}'
    cmd = [str(ROOT / 'venv/bin/python'), str(ROOT / 'screen-timed.py'), '--clients', str(clients),
           '--input', '4096', '--output', '1024', '--nonce', f'concurrency-{clients}-{repeat}',
           '--label', a.name, '--out', str(out / name)]
    with (out / (name + '.log')).open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({'case': name, **json.loads((out / name / 'summary.json').read_text())}), flush=True)
if a.needle:
    cmd = [str(ROOT / 'venv/bin/python'), str(ROOT / 'screen-timed.py'), '--clients', '1',
           '--input', str(a.needle_input), '--output', '64', '--timeout', '900', '--needle',
           '--label', a.name, '--out', str(out / ('needle-' + str(a.needle_input)))]
    with (out / 'needle.log').open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({'case': 'needle-' + str(a.needle_input),
          **json.loads((out / ('needle-' + str(a.needle_input)) / 'summary.json').read_text())}), flush=True)
(out / 'BATCH_DONE').write_text(str(time.time()) + '\n')
