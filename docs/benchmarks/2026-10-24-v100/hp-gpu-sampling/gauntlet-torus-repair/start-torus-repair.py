from pathlib import Path
import subprocess,json,time,shlex,urllib.request
r=Path(__file__).parent;g=r/'hp-256k-gauntlet';p=g/'repair';p.mkdir(exist_ok=False)
assert (g/'original/original-SHA256SUMS.json').exists()
prompt='Source review found your torus does NOT display normal shading: shade is computed but discarded when emitting Braille. Also the depth buffer is per character rather than per Braille dot and OR retains occluded samples. Correct render.py now: use per-dot depth/intensity and actual visible normal lighting (ordered dithering or ANSI grayscale). Preserve solid torus, rotation, adaptive size, cursor cleanup, stdlib only. Stop ONLY bottom pane %140 with Ctrl-C, edit, syntax check, then run the corrected renderer ONLY in pane %140. Do not change other panes or models. Finish within two minutes and state exactly what changed.'
(p/'prompt.txt').write_text(prompt)
with urllib.request.urlopen('http://127.0.0.1:8091/stats') as f:(p/'initial-stats.json').write_bytes(f.read())
def tm(*a):return subprocess.check_output(['tmux',*a],text=True).strip()
w=tm('display-message','-p','-t','volta-baseline','#{window_width}');h=str(int(tm('display-message','-p','-t','volta-baseline','#{window_height}'))+1)
cmd='env -u TMUX asciinema rec -q --window-size '+w+'x'+h+' -c '+shlex.quote('tmux attach-session -t volta-baseline -f ignore-size')+' '+shlex.quote(str(p/'repair.cast'))
tm('new-session','-d','-s','volta-repair-rec','-x',w,'-y',h,cmd)
tm('send-keys','-t','%137','-l',prompt);tm('send-keys','-t','%137','Enter')
(p/'STARTED').write_text(str(time.time()))
print(p)
