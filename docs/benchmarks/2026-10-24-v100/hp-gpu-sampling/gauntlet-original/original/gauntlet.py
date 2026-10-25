import subprocess,pathlib,json,time,signal,os,shlex,urllib.request,statistics,hashlib
root=pathlib.Path(__file__).parent;state=json.loads((root/'panes.json').read_text());panes=state['panes'];session=state['session'];g=root/'gauntlet'
def tm(*args):return subprocess.check_output(['tmux',*args],text=True).strip()
def stats():
 with urllib.request.urlopen('http://127.0.0.1:8091/stats',timeout=4) as f:s=json.load(f)
 s.pop('history',None);s['wall']=time.time();return s
# Record cold prefix-cache baseline counters before admitting the four prompts.
initial=stats();(root/'initial-stats.json').write_text(json.dumps(initial,indent=2))
for i,d in enumerate(state['dirs']):
 cmd='clear && env -u NODE_OPTIONS PI_CODING_AGENT_DIR=/home/lxe/volta-sixhour-20261024/pi-config-262144 pi --offline --provider volta --model deepseek-v4.1-flash:off --tools read,bash,edit,write --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files --session-dir '+shlex.quote(str(root/'sessions'/d))
 tm('send-keys','-t',panes[i],cmd,'Enter')
time.sleep(7)
for i,d in enumerate(state['dirs']):
 tm('send-keys','-t',panes[i],'C-o');tm('send-keys','-t',panes[i],'-l',(g/d/'TASK.md').read_text().replace('\n',' '))
w=int(tm('display-message','-p','-t',session,'#{window_width}'));h=int(tm('display-message','-p','-t',session,'#{window_height}'))+1
cast=root/'gauntlet.cast';assert not cast.exists()
cmd='env -u TMUX asciinema rec -q --window-size '+str(w)+'x'+str(h)+' -c '+shlex.quote('tmux attach-session -t '+session+' -f ignore-size')+' '+shlex.quote(str(cast))
tm('new-session','-d','-s','volta-baseline-rec','-x',str(w),'-y',str(h),cmd);time.sleep(2)
for p in panes[:4]:tm('send-keys','-t',p,'Enter')
t0=time.time();idle=0;seen=False;rows=[];reason='time_limit'
with (root/'stats.jsonl').open('w',buffering=1) as f:
 f.write(json.dumps(initial)+'\n')
 while time.time()-t0<900:
  try:
   s=stats();f.write(json.dumps(s)+'\n');rows.append(s);seen=seen or s.get('requests',0)>initial.get('requests',0)
  except Exception as e:(root/'monitor-errors.log').open('a').write(str(e)+'\n');s={}
  captures=[tm('capture-pane','-p','-t',p) for p in panes[:4]]
  busy=any('Working' in c for c in captures) or s.get('active',0)>0
  idle=0 if busy or not seen else idle+5
  if idle>=60:reason='agents_idle';break
  time.sleep(5)
for i,p in enumerate(panes): (root/('pane-'+str(i)+'.txt')).write_text(tm('capture-pane','-p','-S','-300','-t',p))
if reason=='time_limit':
 for p in panes[:4]:tm('send-keys','-t',p,'Escape')
# Stop only the recorder whose exact argv names this run's cast, before treating it as a closed artifact.
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  args=(d/'cmdline').read_bytes().split(b'\0')
  if args and pathlib.Path(os.fsdecode(args[0])).name=='asciinema' and os.fsencode(str(cast)) in args:os.kill(int(d.name),signal.SIGINT)
 except (OSError,ValueError):pass
time.sleep(2)
subprocess.run(['tmux','kill-session','-t','volta-baseline-rec'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if rows:
 end=rows[-1];wall=end['wall']-initial['wall'];active=[r for r in rows if r.get('active',0)>0];multi=[r for r in rows if r.get('active',0)>=2]
 summary={'stop_reason':reason,'wall_s':wall,'tokens_out':end['tokens_out']-initial['tokens_out'],'tokens_prompt':end['tokens_prompt']-initial['tokens_prompt'],'tokens_cached':end['tokens_cached']-initial['tokens_cached'],'requests':end['requests']-initial['requests'],'out_tok_s_mean_wall':(end['tokens_out']-initial['tokens_out'])/wall,'peak_out_tok_s':max(r['out_tok_s'] for r in rows)}
 if active:summary.update(active_median_tps=statistics.median(r['out_tok_s'] for r in active),active_median_step_ms=statistics.median(r['step_ms'] for r in active))
 if len(multi)>1:summary.update(multi_median_tps=statistics.median(r['out_tok_s'] for r in multi),multi_mean_tps=(multi[-1]['tokens_out']-multi[0]['tokens_out'])/(multi[-1]['wall']-multi[0]['wall']),multi_median_step_ms=statistics.median(r['step_ms'] for r in multi))
 (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
(root/'GAUNTLET_DONE').write_text(reason+'\n')
