#!/usr/bin/env python3
"""Admit the reviewed HP audit only after the original threshold experiment cleanly terminates."""
import pathlib,time,os,subprocess,json,hashlib,urllib.request
r=pathlib.Path(__file__).parent;deadline=min(time.time()+2600,1792907006)
assert (r/'hp-sampling-independent-review.md').exists()
proof=json.loads((r/'hp-sampling-v1-build-proof.json').read_text())
expected='63f597b080dee291d0f146d88d6cef70c075f546d82dcfec192880c13c7d6820'
assert proof['binaries_sha256']['/root/volta-hp-sampling-v1/target-hp-sampling-v1/release/volta-server']==expected
assert not proof['compiler_processes']
while not any((r/'receipts'/n).exists() for n in ['SPECULATION_ABBA_DONE','SPECULATION_EARLY_KILLED']):
 live=False
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if len(argv)>1 and os.fsdecode(argv[1])==str(r/'run-speculation-pair.py'):live=True
  except OSError:pass
 assert live,'Original threshold wrapper disappeared without clean terminal receipt; inspect it before candidate launch'
 assert time.time()<deadline,'Audit admission deadline passed'
 time.sleep(3)
name='hp-gpu-audit1';args=['--engine','hp-sample','--name',name,'--ctx','262144','--slots','4','--head-parallel','--graphs','4','--hp-gpu-sample','--hp-sample-audit','1024']
subprocess.run(['python',str(r/'server-control.py'),*args],check=True)
cfg=json.loads((r/'server-runs'/name/'configuration.json').read_text());assert cfg['binary_sha256']==expected
subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--reserve','192','--admit-free','512'],check=True)
(r/'screens'/name).mkdir(exist_ok=False)
subprocess.run([str(r/'venv/bin/python'),str(r/'hp-sampling-audit-client.py'),'--name',name],check=True)
for remote,local in [('server-'+name+'.log',r/'screens'/name/'server.log'),('memory-'+name+'.jsonl',r/'memory-guards'/name/'observations-through-audit.jsonl')]:
 subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/'+remote,str(local)],check=True)
subprocess.run(['python',str(r/'analyze-hp-sampling-audit.py'),'--log',str(r/'screens'/name/'server.log'),'--out',str(r/'receipts/hp-gpu-audit1-analysis.json')],check=True)
(r/'receipts/HP_GPU_AUDIT_PASSED').write_text(str(time.time())+'\n')
print('HP_GPU_AUDIT_PASSED',flush=True)
