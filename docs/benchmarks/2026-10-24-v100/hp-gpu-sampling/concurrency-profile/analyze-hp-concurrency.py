#!/usr/bin/env python3
"""Strict descriptive validation of the completed six-case HP concurrency profile. CPU only; writes no promotion marker."""
import argparse,datetime,hashlib,json,math,pathlib,re,statistics
ROOT=pathlib.Path(__file__).resolve().parent
SERVER='69d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363'
SOURCE='2e0e16166709d11f4e98802eb74343cdba77a605'
MODEL='9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c'
CASES=[(1,'a'),(4,'a'),(8,'a'),(8,'b'),(4,'b'),(1,'b')]
def demand(ok,msg):
 if not ok:raise ValueError(msg)
def events(text):
 ad=[];done=[]
 for n,line in enumerate(re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',text).splitlines(),1):
  demand(not any(s in line for s in ['panicked at','volta engine stopped','thread caused non-unwinding panic','HP collected-logit audit','HP sampler supplied-table/uniform replay audit']),'runtime error in server log')
  kind='admitted' if ' admitted ' in line else 'done' if ' done ' in line else None
  if kind:
   fields={k:v.strip('"') for k,v in re.findall(r'(\w+)=("[^"]*"|[^ ]+)',line)}
   stamp=line.split()[0];x={'line':n,'utc':stamp,'epoch':datetime.datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp(),**fields}
   (ad if kind=='admitted' else done).append(x)
 return ad,done

def analyze(root,screen,memory):
 hashes={}
 def blob(p):
  b=p.read_bytes();hashes[str(p.relative_to(root)) if p.is_relative_to(root) else str(p)]=hashlib.sha256(b).hexdigest();return b
 def read(p):return json.loads(blob(p))
 cfg=read(screen/'configuration.json');demand(cfg==read(root/'server-runs/hp-gpu-capacity4/configuration.json'),'configuration differs from launch')
 proof=read(root/'hp-sampling-v3-build-proof.json');src=read(root/'hp-sampling-v3-source.json');shared=read(root/'receipts/hp-shared-source-binding.json');model=read(root/'receipts/model-sha256.json')
 demand(cfg['pid']==27988 and cfg['name']=='hp-gpu-capacity4','unexpected PID/run')
 demand(cfg['binary_sha256']==proof['binaries_sha256'][cfg['binary']]==SERVER,'binary hash mismatch')
 demand(src['source_commit']==SOURCE and src['source_sha256']==proof['source_receipt']['source_sha256']==shared['source_sha256'] and shared['exact_tested_engine_sources'] is True,'source binding mismatch')
 demand(model['sha256']==MODEL and model['bytes']==441574130240 and cfg['model']=='/root/model/DeepSeek-V4.1-Flash-1017-Volta.gguf','model binding mismatch')
 expected={'engine':'hp-sample','ctx':262144,'chunk':64,'slots':4,'graphs':4,'head_parallel':True,'profile':False,'loop_profile':0,'graph_trace':False,'hp_gpu_sample':True,'hp_sample_audit':0,'theta':.35,'spec_k':5}
 demand(all(cfg.get(k)==v for k,v in expected.items()),'runtime configuration mismatch')
 demand(cfg['volta_environment'].get('VOLTA_HP_GPU_SAMPLE')=='1' and cfg['volta_environment'].get('VOLTA_HP_SAMPLE_AUDIT','0')=='0','HP flag mismatch')
 batch=read(screen/'batch-configuration.json');demand(batch['context']==262144 and batch['heads'] and batch['active_capacity']==4 and batch['allow_queued'],'batch topology mismatch')
 demand(blob(screen/'harness-sha256.txt').decode().strip()==hashlib.sha256(blob(root/'screen-timed.py')).hexdigest(),'harness hash mismatch')
 demand(batch['batch_source_sha256']==hashlib.sha256(blob(root/'concurrency-hp-gpu-batch.py')).hexdigest(),'batch source mismatch')
 blob(screen/'BATCH_DONE');prior=blob(screen/'prior-work.txt').decode();overall=read(screen/'initial-stats.json');longfinal=read(root/'screens/four-hp-250k/final-stats.json')
 demand(all(overall[k]==longfinal[k] for k in ['requests','tokens_prompt','tokens_out','tokens_cached']),'prior capacity counter continuity differs')
 demand(overall['requests']==4 and overall['tokens_prompt']==1000184 and overall['tokens_out']==68 and overall['tokens_cached']==0,'unexpected prior workload')
 reports=[];previous=overall;normalized=None
 for n,repeat in CASES:
  name=f'clients-{n}-{repeat}';p=screen/name;req=read(p/'requests.json');res=read(p/'results.json');summary=read(p/'summary.json');initial=read(p/'initial-stats.json');final=read(p/'final-stats.json');stats=[json.loads(l) for l in blob(p/'stats.jsonl').decode().splitlines() if l.strip()]
  demand(len(req)==len(res)==n,'request/result count mismatch '+name)
  for x in [initial,final]:demand(x['ctx']==262144 and x['heads'] and x['capacity']==4 and x['gpus']==8 and x['active']==x['queued']==0,'API topology/idle mismatch '+name)
  demand(all(initial[k]==previous[k] for k in ['requests','tokens_prompt','tokens_out','tokens_cached']),'case counter discontinuity '+name)
  clients=[]
  for i,(q,r) in enumerate(zip(req,res)):
   demand(q['model']=='deepseek-v4.1-flash' and q['temperature']==.7 and q['top_p']==1 and q['seed']==12000+i and q['max_tokens']==1024 and q['reasoning_effort']=='none' and q['stream'] and q['stream_options']['include_usage'],'request sampler/budget mismatch '+name)
   prefix=f'Work item concurrency-{n}-{repeat}, client {i}. Review this Python module.\n';demand(len(q['messages'])==1 and q['messages'][0]['role']=='user' and q['messages'][0]['content'].startswith(prefix),'case nonce/client order mismatch '+name)
   tail=q['messages'][0]['content'][len(prefix):]
   if normalized is None:normalized=tail
   demand(tail==normalized,'fixture differs beyond intended nonce/index '+name)
   demand('error' not in r and r['usage']['prompt_tokens']==4145 and r['usage']['completion_tokens']==1024 and r['finish_reason']=='length','partial/failed request '+name)
   u=r['usage'];demand(u.get('prompt_tokens_details',{}).get('cached_tokens')==0 and u.get('prompt_cache_hit_tokens')==0,'client cache reuse '+name)
   demand(hashlib.sha256(r['output'].encode()).hexdigest()==r['output_sha256'],'output hash mismatch '+name)
   wall=r['wall_s'];ttft=r['ttft_s'];demand(0<ttft<wall and math.isclose(r['end_monotonic']-r['start_monotonic'],wall,abs_tol=1e-8),'invalid client timings '+name)
   decode=1023/(wall-ttft);demand(math.isclose(decode,r['decode_tokens_per_s'],rel_tol=1e-8),'decode metric mismatch '+name)
   times=r.get('content_event_times_monotonic',[]);gaps=[b-a for a,b in zip(times,times[1:])];demand(times and all(r['start_monotonic']<=t<=r['end_monotonic'] for t in times) and all(g>=0 for g in gaps),'invalid SSE event times '+name)
   clients.append({'client_input_index':i,'prompt_tokens':4145,'output_tokens':1024,'ttft_s':ttft,'wall_s':wall,'decode_tokens_per_s':decode,'sse_content_events':len(times),'sse_content_gap_median_s':statistics.median(gaps) if gaps else None,'sse_content_gap_max_s':max(gaps) if gaps else None,'output_sha256':r['output_sha256']})
  demand(final['requests']-initial['requests']==n and final['tokens_out']-initial['tokens_out']==1024*n and final['tokens_prompt']-initial['tokens_prompt']==4145*n and final['tokens_cached']-initial['tokens_cached']==0,'case counter mismatch '+name)
  demand(stats and not any('error' in s or 'monitor_error' in s for s in stats),'missing/error monitor '+name)
  active=max(s['active'] for s in stats);queued=max(s['queued'] for s in stats);demand(active==min(n,4) and queued==(4 if n==8 else 0),'active/queue observation mismatch '+name)
  for s in stats:
   ss=s.get('sessions',[]);demand(s['active']<=4 and len(ss)==s['active'] and len({x['slot'] for x in ss})==len(ss) and all(x['gpu']==0 for x in ss),'invalid HP active slots '+name)
  wall=max(r['end_monotonic'] for r in res)-min(r['start_monotonic'] for r in res);tps=n*1024/wall
  demand(summary['label']=='hp-gpu-concurrency' and summary['clients']==summary['completed']==n and summary['mixed'] is False and summary['all_hit_output_budget'] is True and summary['requested_output_tokens']==summary['output_tokens']==summary['server_tokens_out_delta']==n*1024 and summary['prompt_tokens']==n*4145 and summary['cached_tokens_delta']==0 and summary['ctx']==262144 and summary['heads'] is True and summary['ttft_s']==[r['ttft_s'] for r in res] and math.isclose(summary['wall_s'],wall,rel_tol=1e-9) and math.isclose(summary['output_tokens_per_wall_s'],tps,rel_tol=1e-9),'summary mismatch '+name)
  reports.append({'case':name,'submitted_clients':n,'max_observed_active':active,'max_observed_queued':queued,'wall_s':wall,'whole_aggregate_tokens_per_s':tps,'median_per_client_decode_tokens_per_s':statistics.median(c['decode_tokens_per_s'] for c in clients),'prompt_tokens':n*4145,'output_tokens':n*1024,'cached_tokens':0,'clients_in_input_order':clients})
  previous=final
 ad,done=events(blob(screen/'server.log').decode());demand(len(ad)==len(done)==30,'expected four prior long requests plus 26 profile requests')
 demand(all(int(x['prompt_tokens'])==250046 for x in ad[:4]) and all(int(x['generated'])==17 for x in done[:4]),'prior long log prefix mismatch')
 ad=ad[4:];done=done[4:];offset=0;previous_done=0
 for report,(n,repeat) in zip(reports,CASES):
  aa=ad[offset:offset+n];dd=done[offset:offset+n];offset+=n
  ids={x['id'] for x in aa};demand(len(ids)==n and ids=={x['id'] for x in dd},'case raw ID set mismatch')
  demand(all(int(x['prompt_tokens'])==4145 and int(x['cached_tokens'])==0 and int(x['gpu'])==0 and 0<=int(x['slot'])<4 for x in aa),'raw admission mismatch')
  demand(all(int(x['prompt_tokens'])==4145 and int(x['generated'])==1024 and x['finish']=='Length' for x in dd),'raw completion mismatch')
  byid={x['id']:x for x in aa};demand(all(x['epoch']>=byid[x['id']]['epoch'] for x in dd) and min(x['epoch'] for x in aa)>=previous_done,'raw event order mismatch');previous_done=max(x['epoch'] for x in dd)
  report['server_admissions']=aa;report['server_completions']=dd
 demand(len({x['id'] for x in ad})==26,'duplicate profile server IDs')
 first=min(x['epoch'] for x in ad);last=max(x['epoch'] for x in done);gc=read(root/'memory-guards/hp-gpu-concurrency/configuration.json');obs=[json.loads(l) for l in blob(memory).decode().splitlines() if l.strip()]
 demand(gc['first_observation_verified'] and gc['server_pid']==27988 and gc['until']>=last,'guard launch mismatch')
 left=[i for i,x in enumerate(obs) if x['epoch']<=first];right=[i for i,x in enumerate(obs) if x['epoch']>=last];demand(left and right,'refresh memory snapshot after final completion')
 lo=max(left);hi=min(right);demand(lo<hi,'invalid guard bracket');selected=obs[lo:hi+1]
 demand(all(x['pid']==27988 and x['stop'] is False and x['threshold_mib']>=192 and len(x['gpus_index_free_used_mib'])==8 and {g[0] for g in x['gpus_index_free_used_mib']}==set(range(8)) for x in selected),'guard identity/stop/threshold mismatch')
 gaps=[b['epoch']-a['epoch'] for a,b in zip(selected,selected[1:])];low=min(g[1] for x in selected for g in x['gpus_index_free_used_mib']);demand(min(gaps)>0 and max(gaps)<=2 and low>=192,'guard gap/reserve breach')
 ranges={str(n):{'whole_aggregate_tokens_per_s':[min(r['whole_aggregate_tokens_per_s'] for r in reports if r['submitted_clients']==n),max(r['whole_aggregate_tokens_per_s'] for r in reports if r['submitted_clients']==n)],'median_per_client_decode_tokens_per_s':[min(r['median_per_client_decode_tokens_per_s'] for r in reports if r['submitted_clients']==n),max(r['median_per_client_decode_tokens_per_s'] for r in reports if r['submitted_clients']==n)]} for n in [1,4,8]}
 return {'status':'PASS','scope':'Descriptive six-case HP concurrency profile; no controlled cross-mode gain.','source_binding':{'source_commit':SOURCE,'source_sha256':src['source_sha256'],'binary_sha256':SERVER,'model_sha256':MODEL,'model_bytes':441574130240,'model_hash_scope':'Earlier full-file hydration receipt and configured path; no new per-probe full rehash.','pid':27988},'case_order':[r['case'] for r in reports],'cases':reports,'ranges_across_two_repeats':ranges,'profile_prompt_tokens':26*4145,'profile_output_tokens':26*1024,'prior_work':prior,'first_profile_admitted_epoch':first,'last_profile_done_epoch':last,'memory':{'input':str(memory),'total_file_rows':len(obs),'selected_rows':len(selected),'selected_first_file_row_1based':lo+1,'selected_last_file_row_1based':hi+1,'selection_rule':'Last observation at/before first profile admission through first observation at/after last profile completion, inclusive. Entire raw input preserved and hashed.','first_epoch':selected[0]['epoch'],'last_epoch':selected[-1]['epoch'],'minimum_free_mib':low,'maximum_gap_s':max(gaps),'reserve_mib':192},'limits':['Same server retains prior four-long-request KV/cache/graph history; distinct nonces and zero cache counters establish cold requests, not a fresh process.','Eight submissions queue behind four active HP slots; this does not establish eight active sessions.','Client results are in input order, but API records lack server IDs. Raw server IDs are validated within each case without a client-to-ID join.','Per-client decode TPS uses (1024 - 1) / (client wall - TTFT); whole aggregate TPS includes prompt processing and queued waiting.','SSE gap measurements describe client-observed content chunks, which may contain multiple model tokens; they are not model inter-token latency.','Memory guard observations are sampled and do not establish unsampled behavior.','Sampled outputs can follow different RNG and speculative trajectories; this profile does not establish controlled cross-mode speedup or general quality.'],'input_sha256':hashes}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,default=ROOT);p.add_argument('--screen',type=pathlib.Path);p.add_argument('--memory',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path);a=p.parse_args();root=a.root.resolve();screen=a.screen or root/'screens/hp-gpu-concurrency';memory=a.memory or screen/'memory-final.jsonl';out=a.out or root/'receipts/hp-gpu-concurrency-analysis.json'
 try:report=analyze(root,screen,memory)
 except FileNotFoundError as e:report={'status':'INCOMPLETE','error':str(e)}
 except Exception as e:report={'status':'FAIL','error':str(e)}
 report['analyzer_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','ranges_across_two_repeats','error'] if k in report}));return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
