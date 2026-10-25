#!/usr/bin/env python3
"""Strict CPU-only validation of the completed four-key HP 250K retrieval probe.
No server/GPU operations and no promotion/pass marker writes. Preserve a transient queue as a separate failed planned gate.
"""
import argparse,datetime,hashlib,json,math,pathlib,re,sys
ROOT=pathlib.Path(__file__).resolve().parent
MODEL='9d15a1d6f263b25bb3bf0eb1b887c9c6d7176083b8a9bd55c998e8658e52000c'
SERVER='69d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363'
SOURCE='2e0e16166709d11f4e98802eb74343cdba77a605'
ANSI=re.compile(r'\x1b\[[0-9;]*[A-Za-z]');FIELDS=re.compile(r'(\w+)=("[^"]*"|[^ ]+)')
def epoch(stamp):return datetime.datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp()
def demand(ok,message):
 if not ok:raise ValueError(message)
def parse_events(text):
 admitted=[];done=[];errors=[]
 for n,line in enumerate(ANSI.sub('',text).splitlines(),1):
  if any(x in line for x in ['panicked at','volta engine stopped','thread caused non-unwinding panic','HP collected-logit audit','HP sampler supplied-table/uniform replay audit']):errors.append({'line':n,'text':line})
  kind='admitted' if ' admitted ' in line else 'done' if ' done ' in line else None
  if kind:
   fields={k:v.strip('"') for k,v in FIELDS.findall(line)}
   demand('id' in fields,'server event without request ID')
   item={'line':n,'utc':line.split()[0],'epoch':epoch(line.split()[0]),**fields}
   (admitted if kind=='admitted' else done).append(item)
 return admitted,done,errors

def analyze(root,screen,memory):
 hashes={}
 def blob(p):
  p=pathlib.Path(p);b=p.read_bytes();hashes[str(p.relative_to(root)) if p.is_relative_to(root) else str(p)]=hashlib.sha256(b).hexdigest();return b
 def read(p):return json.loads(blob(p))
 cfg=read(root/'server-runs/hp-gpu-capacity4/configuration.json');saved=read(screen/'configuration.json');demand(cfg==saved,'saved screen configuration differs from launch configuration')
 proof=read(root/'hp-sampling-v3-build-proof.json');source=read(root/'hp-sampling-v3-source.json');shared=read(root/'receipts/hp-shared-source-binding.json');model=read(root/'receipts/model-sha256.json')
 demand(cfg['name']=='hp-gpu-capacity4' and cfg['pid']==27988,'unexpected run/PID; review before reuse')
 demand(cfg['binary_sha256']==proof['binaries_sha256'][cfg['binary']]==SERVER,'binary binding mismatch')
 demand(source['source_commit']==SOURCE and proof['source_receipt']['source_sha256']==source['source_sha256']==shared['source_sha256'] and shared['exact_tested_engine_sources'] is True,'source binding mismatch')
 demand(model['sha256']==MODEL and model['bytes']==441574130240 and cfg['model']=='/root/model/DeepSeek-V4.1-Flash-1017-Volta.gguf','model hash/size/path binding mismatch')
 expected_cfg={'engine':'hp-sample','ctx':262144,'chunk':64,'slots':4,'graphs':4,'head_parallel':True,'profile':False,'loop_profile':0,'graph_trace':False,'hp_gpu_sample':True,'hp_sample_audit':0,'theta':.35,'spec_k':5}
 demand(all(cfg.get(k)==v for k,v in expected_cfg.items()),'capacity configuration mismatch')
 demand(cfg['volta_environment'].get('VOLTA_HP_GPU_SAMPLE')=='1' and cfg['volta_environment'].get('VOLTA_HP_SAMPLE_AUDIT','0')=='0','HP flags mismatch')
 req=read(screen/'requests.json');res=read(screen/'results.json');summary=read(screen/'summary.json');initial=read(screen/'initial-stats.json');final=read(screen/'final-stats.json')
 stats=[json.loads(l) for l in blob(screen/'stats.jsonl').decode().splitlines() if l.strip()]
 demand(len(req)==len(res)==4,'exactly four request/result entries required')
 keys=[f'VOLTA_LONG_FOUR_{i:02d}_8a271d60b7fc' for i in range(4)];results=[]
 base_prompt=None
 for i,(request,result,key) in enumerate(zip(req,res,keys)):
  demand(request['model']=='deepseek-v4.1-flash' and request['temperature']==0 and request['top_p']==1 and request['seed']==12000+i and request['max_tokens']==64 and request['reasoning_effort']=='none','request order/greedy/budget mismatch')
  demand(len(request['messages'])==1 and request['messages'][0]['role']=='user','unexpected message schema')
  prompt=request['messages'][0]['content'];demand(prompt.startswith('Remember this deployment verification key: '+key+'.\n'),'request key or position mismatch')
  demand(all(other not in prompt for other in keys if other!=key),'request contains another client key')
  normalized=prompt.replace(key,'EXPECTED_KEY')
  if base_prompt is None:base_prompt=normalized
  demand(normalized==base_prompt,'request body differs beyond intended distinct key')
  demand('error' not in result,'API request failed')
  usage=result['usage'];output=result['output'];demand(output.strip()==key,'wrong or cross-session key in result order')
  demand(usage['prompt_tokens']==250046 and 0<usage['completion_tokens']<request['max_tokens'] and result['finish_reason']=='stop','prompt size or natural stop mismatch')
  demand(hashlib.sha256(output.encode()).hexdigest()==result['output_sha256'],'output hash mismatch')
  demand(0<result['ttft_s']<=result['wall_s'] and result['end_monotonic']>=result['start_monotonic'],'invalid request timing')
  results.append({'client_index':i,'expected_key':key,'actual_output':output,'exact_after_outer_whitespace_trim':True,'raw_exact_match':output==key,'prompt_tokens':usage['prompt_tokens'],'output_tokens':usage['completion_tokens'],'finish_reason':result['finish_reason'],'ttft_s':result['ttft_s'],'wall_s':result['wall_s'],'output_sha256':result['output_sha256']})
 demand(len(set(x['output'].strip() for x in res))==4,'keys are not distinct')
 total_out=sum(x['output_tokens'] for x in results);total_prompt=4*250046
 demand(all(x['ctx']==262144 and x['heads'] is True and x['capacity']==4 and x['gpus']==8 for x in [initial,final]),'initial/final API capacity identity differs')
 demand(initial['requests']==0,'expected a fresh capacity server')
 demand(initial['active']==initial['queued']==final['active']==final['queued']==0,'initial/final server not idle')
 demand(final['requests']-initial['requests']==4 and final['tokens_out']-initial['tokens_out']==total_out and final['tokens_prompt']-initial['tokens_prompt']==total_prompt and final['tokens_cached']-initial['tokens_cached']==0,'server request/token/cache counters differ')
 demand(summary['label']==cfg['name'] and summary['ctx']==262144 and summary['heads'] is True,'summary runtime identity differs')
 demand(summary['completed']==4 and summary['clients']==4 and summary['needle_verified'] is True and summary['expected_keys']==keys and summary['prompt_tokens']==total_prompt and summary['output_tokens']==total_out and summary['server_tokens_out_delta']==total_out and summary['cached_tokens_delta']==0,'summary does not match actual results')
 demand(summary['all_hit_output_budget'] is False,'natural-stop probe unexpectedly claims full output-budget saturation')
 wall=max(x['end_monotonic'] for x in res)-min(x['start_monotonic'] for x in res);demand(math.isclose(wall,summary['wall_s'],rel_tol=1e-9),'wall recomputation differs')
 good_stats=[x for x in stats if 'active' in x and 'queued' in x];monitor_errors=[x for x in stats if 'error' in x or 'monitor_error' in x]
 demand(good_stats,'no usable active/queue observations');active=max(x['active'] for x in good_stats);queued=max(x['queued'] for x in good_stats)
 four_sessions=[]
 for x in good_stats:
  if x['active']==4:
   ss=x.get('sessions',[])
   demand(len(ss)==4 and len({s['slot'] for s in ss})==4 and all(s['gpu']==0 for s in ss),'four HP active sessions lack distinct slots')
   four_sessions.append(x)
 demand(active==4 and four_sessions,'did not observe four admitted active HP sessions')
 admissions,done,errors=parse_events(blob(screen/'server.log').decode());demand(not errors,'runtime failure in server log')
 demand(len(admissions)==len(done)==4 and len({x['id'] for x in admissions})==4 and {x['id'] for x in admissions}=={x['id'] for x in done},'server admission/completion IDs missing or duplicated')
 demand(all(int(x['prompt_tokens'])==250046 and int(x['cached_tokens'])==0 for x in admissions),'raw admission tokens/cache mismatch')
 demand(len({int(x['slot']) for x in admissions})==4 and all(int(x['gpu'])==0 for x in admissions),'raw HP slot assignment mismatch')
 demand(all(int(x['prompt_tokens'])==250046 and x['finish']=='Stop' for x in done) and sorted(int(x['generated']) for x in done)==sorted(x['output_tokens'] for x in results),'raw completions disagree with natural API completions')
 ad={x['id']:x for x in admissions};demand(all(x['epoch']>=ad[x['id']]['epoch'] for x in done),'server completion precedes admission')
 first=min(x['epoch'] for x in admissions);last=max(x['epoch'] for x in done)
 gc=read(root/'memory-guards/hp-gpu-capacity4/configuration.json');observations=[json.loads(l) for l in blob(memory).decode().splitlines() if l.strip()]
 demand(gc['first_observation_verified'] and gc['server_pid']==cfg['pid'] and gc['until']>=last,'guard admission/expiration mismatch')
 demand(len(observations)>1,'guard input has too few observations')
 # Hash the whole original file above, but validate only the minimal inclusive capacity bracket.
 # Preserve source order; do not sort or silently discard a gap inside the bracket.
 lefts=[i for i,x in enumerate(observations) if x['epoch']<=first]
 rights=[i for i,x in enumerate(observations) if x['epoch']>=last]
 demand(lefts and rights,'memory snapshot must bracket first admission through last completion; refresh after last done')
 lo_index=max(lefts);hi_index=min(rights)
 demand(lo_index<hi_index,'guard bracket order invalid')
 selected=observations[lo_index:hi_index+1]
 demand(all(x['pid']==cfg['pid'] and x['stop'] is False and x['threshold_mib']>=192 and len(x['gpus_index_free_used_mib'])==8 and {g[0] for g in x['gpus_index_free_used_mib']}==set(range(8)) for x in selected),'guard PID/stop/threshold/GPU identity mismatch within capacity bracket')
 gaps=[b['epoch']-a['epoch'] for a,b in zip(selected,selected[1:])];demand(min(gaps)>0 and max(gaps)<=2,'guard observation gap exceeds original two-second rule within capacity bracket')
 low=min(g[1] for x in selected for g in x['gpus_index_free_used_mib']);demand(low>=192,'guard reserve breach within capacity bracket')
 demand(selected[0]['epoch']<=first and selected[-1]['epoch']>=last,'selected memory bracket misses capacity interval')
 queue_pass=queued==0;retrieval=True;memory_pass=True;original_pass=queue_pass and not monitor_errors
 return {'status':'PASS' if original_pass else 'QUALIFIED_RETRIEVAL_AND_MEMORY_PASS','original_planned_all_gates_passed':original_pass,'retrieval_passed':retrieval,'memory_guard_passed':memory_pass,'four_admitted_active_sessions_observed':True,'original_no_queue_gate_passed':queue_pass,'max_observed_active':active,'max_observed_queued':queued,'monitor_errors':monitor_errors,'source_binding':{'source_commit':SOURCE,'source_sha256':source['source_sha256'],'server_sha256':SERVER,'pid':cfg['pid'],'model_sha256':MODEL,'model_bytes':441574130240,'model_hash_scope':'Earlier verified full-file hydration receipt plus unchanged configured path; no new per-probe full rehash.'},'results_in_client_input_order':results,'server_admissions':admissions,'server_completions':done,'request_id_mapping_limit':'API results do not store server request IDs. Client source returns futures in input order, exact distinct key results validate that order; raw server ID events are validated separately without fabricating an ID-to-client index join.','prompt_tokens_total':total_prompt,'output_tokens_total':total_out,'cached_tokens':0,'wall_s':wall,'first_admitted_epoch':first,'last_done_epoch':last,'memory':{'input':str(memory),'observations':len(selected),'total_file_rows':len(observations),'selected_rows':len(selected),'selected_first_file_row_1based':lo_index+1,'selected_last_file_row_1based':hi_index+1,'selection_rule':'Last original observation at/before first admission through first original observation at/after last completion, inclusive; full raw file retained and hashed.','first_epoch':selected[0]['epoch'],'last_epoch':selected[-1]['epoch'],'maximum_gap_s':max(gaps),'minimum_free_mib':low,'reserve_mib':192},'limits':['This checks four distinct-key retrieval requests at approximately250K prompt tokens each, not general long-context quality.','Four sessions were admitted and active in separate HP slots, but the scheduler shares one64-row HP prefill group; it does not establish four parallel64-row prefills.','These requests are greedy and use unchanged StepGreedy, so they do not measure new sampled-GPU-path throughput.','Guard readings are sampled; passing gaps and reserve checks do not prove unsampled memory behavior.','Original no-queue gate is reported separately and is not relaxed to erase transient admission queueing.','Naturally stopped outputs are expected to be shorter than64tokens; this is not a fixed-output throughput screen.'],'input_sha256':hashes}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=pathlib.Path,default=ROOT);p.add_argument('--screen',type=pathlib.Path);p.add_argument('--memory',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path);a=p.parse_args();root=a.root.resolve();screen=a.screen or root/'screens/four-hp-250k';memory=a.memory or screen/'memory-final.jsonl';out=a.out or root/'receipts/four-hp-250k-analysis.json'
 try:report=analyze(root,screen,memory)
 except FileNotFoundError as e:report={'status':'INCOMPLETE','original_planned_all_gates_passed':False,'error':str(e),'note':'Wait for completed client files and a parent-refreshed memory snapshot; analyzer performs no remote action.'}
 except Exception as e:report={'status':'FAIL','original_planned_all_gates_passed':False,'error':str(e),'note':'Original inputs are untouched; no gate or threshold was relaxed.'}
 report['analyzer_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','original_planned_all_gates_passed','retrieval_passed','memory_guard_passed','max_observed_active','max_observed_queued','error'] if k in report}));return 0 if report.get('original_planned_all_gates_passed') else 1
if __name__=='__main__':raise SystemExit(main())
