import pathlib,time,os,subprocess
r=pathlib.Path(__file__).parent;deadline=time.time()+900
while not (r/'receipts/HP_CPU_A1_DONE').exists():
 live=False
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   a=(p/'cmdline').read_bytes().split(b'\0')
   if len(a)>1 and os.fsdecode(a[1])==str(r/'run-hp-cpu-a1.py'):live=True
  except OSError:pass
 assert live,'Original HP baseline wrapper disappeared without success marker; inspect it'
 assert time.time()<deadline
 time.sleep(3)
subprocess.run(['python',str(r/'run-speculation-pair.py')],check=True)
