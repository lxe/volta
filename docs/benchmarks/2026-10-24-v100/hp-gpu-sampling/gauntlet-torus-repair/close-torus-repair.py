from pathlib import Path
import subprocess,json,time,os,signal,urllib.request,hashlib
r=Path(__file__).parent;g=r/'hp-256k-gauntlet';p=g/'repair'
def tm(*a):return subprocess.check_output(['tmux',*a],text=True)
assert 'Working' not in tm('capture-pane','-p','-t','%137')
with urllib.request.urlopen('http://127.0.0.1:8091/stats') as f:s=json.load(f)
assert s['active']==0
(p/'final-stats.json').write_text(json.dumps(s,indent=2));(p/'agent-final.txt').write_text(tm('capture-pane','-p','-S','-300','-t','%137'))
cast=p/'repair.cast'
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  args=(d/'cmdline').read_bytes().split(b'\0')
  if args and Path(os.fsdecode(args[0])).name=='asciinema' and os.fsencode(str(cast)) in args:os.kill(int(d.name),signal.SIGINT)
 except (OSError,ValueError):pass
time.sleep(2)
subprocess.run(['tmux','kill-session','-t','volta-repair-rec'],capture_output=True)
lines=0
for line in cast.open('rb'):json.loads(line);lines+=1
(p/'cast-receipt.json').write_text(json.dumps({'sha256':hashlib.sha256(cast.read_bytes()).hexdigest(),'bytes':cast.stat().st_size,'lines':lines,'closed':True},indent=2))
(p/'render.py').write_bytes((g/'gauntlet/03-torus/render.py').read_bytes())
a=json.loads((p/'initial-stats.json').read_text());(p/'summary.json').write_text(json.dumps({'scope':'separate parent-requested Pi torus repair, not original gauntlet','requests':s['requests']-a['requests'],'tokens_out':s['tokens_out']-a['tokens_out'],'source_sha256':hashlib.sha256((p/'render.py').read_bytes()).hexdigest()},indent=2))
print((p/'summary.json').read_text())
