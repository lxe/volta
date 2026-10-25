#!/usr/bin/env python3
"""Measure HP CPU sampling separately with the verified accounting-only V3 binary."""
import pathlib,subprocess,json
r=pathlib.Path(__file__).parent
name='hp4-loop-prof-001'
p=json.loads((r/'loop-prof-v3-build-proof.json').read_text())
assert p['binary_sha256']=='c50eb1df6dc084b116dfe26a07a750ceb9e8b7db3797c38e2f4e956637de5972'
args=['--engine','loop-prof','--name',name,'--ctx','262144','--slots','4','--head-parallel','--graphs','4','--profile','--loop-profile','1']
subprocess.run(['python',str(r/'server-control.py'),*args],check=True)
cfg=json.loads((r/'server-runs'/name/'configuration.json').read_text())
assert cfg['binary_sha256']==p['binary_sha256']
subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--reserve','192','--admit-free','512'],check=True)
(r/'screens'/name).mkdir(exist_ok=False)
subprocess.run(['python',str(r/'run-arm.py'),*args,'--pair','guard-02','--resume-existing'],check=True)
subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),f'volta-run:/root/server-{name}.log',str(r/'screens'/name/'server.log')],check=True)
subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),f'volta-run:/root/memory-{name}.jsonl',str(r/'memory-guards'/name/'observations-through-profile.jsonl')],check=True)
(r/'receipts/HP_LOOP_PROFILE_DONE').write_text('Actual CPU sampling profile copied; instrumented, not primary timing.\n')
