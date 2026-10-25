#!/usr/bin/env python3
"""Rental-side guard for a specifically named owned model server, after readiness."""
import argparse
import json
import os
import pathlib
import signal
import re
import subprocess
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--name', required=True)
ap.add_argument('--receipt-name')
ap.add_argument('--min-free-mib', type=int, required=True)
ap.add_argument('--until', type=float, required=True)
a = ap.parse_args()
receipt_name = a.receipt_name or a.name
assert re.fullmatch(r'[a-zA-Z0-9_-]+', receipt_name)
cfg = json.loads(pathlib.Path('/root/current-server.json').read_text())
assert cfg['name'] == a.name
assert cfg['binary'] in {'/root/volta/target/release/volta-server',
                         '/root/volta-candidate/target/release/volta-server',
                         '/root/volta-graph-diag/target/release/volta-server',
                         '/root/volta-graph-diag-v2/target-v2/release/volta-server',
                         '/root/volta-loop-prof-v3/target-v3/release/volta-server','/root/volta-hp-sampling-v1/target-hp-sampling-v1/release/volta-server'}
pid, exe = cfg['pid'], cfg['binary']
out = pathlib.Path('/root/memory-' + receipt_name + '.jsonl')
with out.open('x', buffering=1) as log:
    while time.time() < a.until:
        try:
            if os.readlink(f'/proc/{pid}/exe') != exe:
                break
        except FileNotFoundError:
            break
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.used',
              '--format=csv,noheader,nounits'], text=True, timeout=10)
        rows = [[int(v.strip()) for v in line.split(',')] for line in raw.splitlines()]
        assert len(rows) == 8
        stop = min(r[1] for r in rows) < a.min_free_mib
        row = {'epoch': time.time(), 'pid': pid, 'gpus_index_free_used_mib': rows,
               'threshold_mib': a.min_free_mib, 'stop': stop}
        log.write(json.dumps(row) + '\n')
        if stop:
            # Validate executable again to avoid signalling a recycled PID.
            if os.readlink(f'/proc/{pid}/exe') == exe:
                os.kill(pid, signal.SIGTERM)
            pathlib.Path('/root/MEMORY_GUARD_' + receipt_name).write_text(json.dumps(row, indent=2))
            raise SystemExit('known server stopped below its memory reserve')
        time.sleep(1)
