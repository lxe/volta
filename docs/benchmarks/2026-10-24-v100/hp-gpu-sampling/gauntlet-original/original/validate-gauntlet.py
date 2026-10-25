#!/usr/bin/env python3
"""Capture bounded live animation evidence; optionally validate a closed recording."""
import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('directory', type=pathlib.Path)
ap.add_argument('--closed-cast', action='store_true')
a = ap.parse_args()
r = a.directory
state = json.loads((r / 'panes.json').read_text())
panes = state['panes'][4:]
samples = []
for _ in range(3):
    samples.append([subprocess.check_output(['tmux', 'capture-pane', '-p', '-t', p]) for p in panes])
    time.sleep(.7)
proof = []
for i, d in enumerate(state['dirs']):
    p = r / 'gauntlet' / d / 'render.py'
    tree = ast.parse(p.read_text())
    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.extend(x.name.split('.')[0] for x in n.names)
        elif isinstance(n, ast.ImportFrom):
            imports.append(n.module.split('.')[0])
    assert all(x in sys.stdlib_module_names for x in imports), imports
    frames = [x[i] for x in samples]
    for j, f in enumerate(frames):
        (r / f'visual-{d}-{j}.txt').write_bytes(f)
    assert len(set(frames)) >= 2, (d, 'no observed motion')
    assert all(sum(0x2801 <= ord(c) <= 0x28ff for c in f.decode()) >= 10 for f in frames)
    command = subprocess.check_output(['tmux', 'display-message', '-p', '-t', panes[i], '#{pane_current_command}'], text=True).strip()
    assert command.startswith('python'), (d, command)
    proof.append({'shape': d, 'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                  'imports': imports, 'syntax': 'valid', 'distinct_frames': len(set(frames)),
                  'frame_sha256': [hashlib.sha256(f).hexdigest() for f in frames],
                  'scope': 'syntax, standard-library imports, running process and distinct live braille frames; visual geometry reviewed separately'})
(r / 'animation-evidence.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))
if a.closed_cast:
    assert (r / 'GAUNTLET_DONE').exists()
    cast = r / 'gauntlet.cast'
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / 'cmdline').read_bytes().split(b'\0')
            assert not (argv and pathlib.Path(argv[0].decode(errors='replace')).name == 'asciinema' and str(cast).encode() in argv), 'recorder still running'
        except OSError:
            pass
    digest, lines = hashlib.sha256(), 0
    with cast.open('rb') as f:
        for line in f:
            digest.update(line)
            json.loads(line)
            lines += 1
    result = {'cast_sha256': digest.hexdigest(), 'cast_bytes': cast.stat().st_size,
              'json_lines': lines, 'recording_closed': True,
              'controller_complete': (r / 'GAUNTLET_DONE').read_text().strip()}
    (r / 'cast-receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
