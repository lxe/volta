#!/usr/bin/env python3
"""Remote worker for one parent-authorized flag pair; never invoked by CPU preparation."""
import datetime,hashlib,json,math,os,pathlib,signal,subprocess,sys,time,traceback,urllib.request
ROOT=pathlib.Path(sys.argv[1]).parent
SPEC=json.loads(pathlib.Path(sys.argv[1]).read_text())
HARD=datetime.datetime.fromisoformat('2026-10-25T06:16:26+00:00').timestamp()
START=time.time();STATE={'phase':'preflight','started_epoch':START,'worker_pid':os.getpid(),'arms':[]}
CHILD=None

def save():
 p=ROOT/'state.tmp';p.write_text(json.dumps(STATE,indent=2)+'\n');p.replace(ROOT/'state.json')
def sha(p):
 with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gpu():
 p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
 a=[[int(v.strip()) for v in l.split(',')] for l in p.stdout.splitlines()];assert len(a)==8 and {r[0] for r in a}==set(range(8));return a
def idle():
 with urllib.request.urlopen('http://127.0.0.1:8091/stats',timeout=10) as f:x=json.load(f)
 assert x['active']==0 and x['queued']==0,'server no longer idle';return x
def metadata(p):
 s=pathlib.Path(p).stat();return {k:getattr(s,k) for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']}
def deadline():
 assert time.time()<HARD,'absolute06:16:26 cutoff'
 if time.time()-START>420 and not STATE.get('planned_budget_exceeded'):
  STATE['planned_budget_exceeded']=True;(ROOT/'BUDGET_EXCEEDED').write_text('Seven minute plan exceeded; inspect same worker/CLI, no restart.\n');save()
def owned_stop(reason):
 global CHILD
 if CHILD is None or CHILD.poll() is not None:return
 pid=CHILD.pid
 assert os.readlink(f'/proc/{pid}/exe')==SPEC['cli'],'owned CLI executable identity changed'
 STATE['owned_stop']={'pid':pid,'reason':reason,'epoch':time.time()};save();os.kill(pid,signal.SIGKILL if time.time()>=HARD else signal.SIGTERM)
 try:CHILD.wait(timeout=.1 if time.time()>=HARD else 5)
 except subprocess.TimeoutExpired:
  if CHILD.poll() is None:
   assert os.readlink(f'/proc/{pid}/exe')==SPEC['cli'];os.kill(pid,signal.SIGKILL);CHILD.wait(timeout=5)
def released():
 end=min(time.time()+60,HARD)
 while time.time()<end:
  rows=gpu()
  if all(x[2]==0 for x in rows):return rows
  time.sleep(1)
 raise RuntimeError('GPU memory not zero; no next CLI launched')
def gate(x):
 assert x['tokens']==8184
 assert all(type(x[k]) in (int,float) and math.isfinite(x[k]) for k in ['ppl','kl_top256','top1_agree','wall_s'])
 assert x['ppl']>0 and 0<=x['top1_agree']<=1 and x['wall_s']>=0
 b=SPEC['baseline'];return x['ppl']<=b['ppl']+.1 and x['kl_top256']<=b['kl_top256']+.005 and x['top1_agree']>=b['top1_agree']-.005
def hard_alarm(signum, frame):raise TimeoutError('absolute06:16:26 alarm')
signal.signal(signal.SIGALRM,hard_alarm)
signal.setitimer(signal.ITIMER_REAL,max(.01,HARD-time.time()))
try:
 deadline();assert HARD-time.time()>=420,'insufficient planned pair budget before hard cutoff'
 assert SPEC['source_commit']=='2e0e16166709d11f4e98802eb74343cdba77a605'
 assert sha(SPEC['cli'])==SPEC['cli_sha256']=='bcfe792f228ba7216237c13e5d582dd7e4a34e0872ddd4c293c77d10a237cd67'
 for rel,expected in SPEC['source_sha256'].items():assert sha(pathlib.Path(SPEC['source_root'])/rel)==expected
 fats=list(pathlib.Path(SPEC['target']).glob('release/build/*/out/dsx.fatbin'));assert fats and all(sha(p)==SPEC['fatbin_sha256'] for p in fats)
 cached=json.loads(pathlib.Path('/root/model-sha256.json').read_text());assert cached['sha256']==SPEC['model_receipt']['sha256'] and cached['bytes']==SPEC['model_receipt']['bytes']
 model_meta=metadata(SPEC['model']);assert model_meta['st_size']==cached['bytes'];STATE['cached_model_hash_receipt']=cached;STATE['model_metadata']=model_meta
 cfg=json.loads(pathlib.Path('/root/current-server.json').read_text());expected=SPEC['a2']
 for k in ['name','pid','binary','binary_sha256']:assert cfg[k]==expected[k],f'A2 identity mismatch {k}'
 assert cfg['name']=='hp-cpu-a2' and cfg['head_parallel'] and cfg['slots']==4 and cfg['ctx']==262144
 assert os.readlink(f"/proc/{cfg['pid']}/exe")==cfg['binary'] and sha(cfg['binary'])==cfg['binary_sha256']
 STATE['initial_stats']=idle();STATE['stopped_A2']={'pid':cfg['pid'],'binary':cfg['binary'],'epoch':time.time()};save()
 os.kill(cfg['pid'],signal.SIGTERM);STATE['after_A2_stop_gpu']=released();save()
 base=os.environ.copy()
 for k in list(base):
  if k.startswith('VOLTA_'):base.pop(k)
 base['LD_LIBRARY_PATH']='/usr/local/cuda-12.9/targets/x86_64-linux/lib:'+base.get('LD_LIBRARY_PATH','')
 base['VOLTA_HEADS']='1';base['VOLTA_GRAPHS']='4'
 input_names=['heldout.i32','teacher_idx.i32','teacher_lp.f32','teacher_kl.json'];input_hashes=None
 for flag in [0,1]:
  deadline();assert all(x[2]==0 for x in gpu()),'memory not zero before fresh CLI'
  assert metadata(SPEC['model'])==model_meta,'model metadata changed; no rehash attempted'
  inputs={n:sha(pathlib.Path('/root/eval')/n) for n in input_names}
  if input_hashes is None:input_hashes=inputs
  else:assert inputs==input_hashes,'teacher input bytes changed between arms'
  env=base.copy();env['VOLTA_HP_GPU_SAMPLE']=str(flag)
  cmd=[SPEC['cli'],'eval','--model',SPEC['model'],'--eval','/root/eval','--seqs','8','--len','1024','--chunk','1']
  stdout=ROOT/f'flag{flag}.stdout.jsonl';stderr=ROOT/f'flag{flag}.stderr.log';observations=ROOT/f'flag{flag}.memory.jsonl'
  assert HARD-time.time()>=180,'insufficient time to start another full teacher arm'
  with stdout.open('x') as out,stderr.open('x') as err:
   CHILD=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  arm={'flag':flag,'pid':CHILD.pid,'command':cmd,'started_epoch':time.time(),'input_sha256':inputs,'model_metadata':model_meta,'binary_sha256':SPEC['cli_sha256'],'volta_environment':{k:v for k,v in env.items() if k.startswith('VOLTA_')},'other_runtime_environment':{k:v for k,v in env.items() if k.startswith('CUDA_') or k in ['LD_LIBRARY_PATH','OMP_NUM_THREADS','RAYON_NUM_THREADS']}}
  (ROOT/f'flag{flag}.configuration.json').write_text(json.dumps(arm,indent=2)+'\n');STATE['phase']=f'flag{flag}';STATE['active_cli']=arm;save()
  with observations.open('x') as obs:
   while CHILD.poll() is None:
    deadline();rows=gpu();low=min(x[1] for x in rows);stop=low<192
    rec={'epoch':time.time(),'pid':CHILD.pid,'gpu_index_free_used_mib':rows,'reserve_mib':192,'stop':stop};obs.write(json.dumps(rec)+'\n');obs.flush()
    if stop:owned_stop('free memory below192MiB');raise RuntimeError('quality memory reserve violation')
    time.sleep(min(2,max(.01,HARD-time.time())))
  arm['exit_code']=CHILD.returncode;arm['exited_epoch']=time.time();assert CHILD.returncode==0,f'flag{flag} exited{CHILD.returncode}'
  metric=json.loads(stdout.read_text().strip());arm['metrics']=metric;arm['tight_gate_passed']=gate(metric);arm['after_exit_gpu']=released()
  (ROOT/f'flag{flag}.result.json').write_text(json.dumps(arm,indent=2)+'\n');STATE['arms'].append(arm);STATE.pop('active_cli',None);CHILD=None;save()
 STATE['phase']='COMPLETE';STATE['both_tight_gate_passed']=all(a['tight_gate_passed'] for a in STATE['arms']);STATE['finished_epoch']=time.time();save()
except BaseException as e:
 try:owned_stop('worker failure or hard cutoff')
 except BaseException as cleanup:STATE['cleanup_error']=str(cleanup)
 STATE['phase']='FAILED';STATE['error']=str(e);STATE['traceback']=traceback.format_exc();STATE['finished_epoch']=time.time();save();raise
