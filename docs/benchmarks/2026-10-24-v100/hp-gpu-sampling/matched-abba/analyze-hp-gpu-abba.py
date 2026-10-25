#!/usr/bin/env python3
"""CPU-only strict analysis of the four completed HP timing arms; never controls hardware."""
import hashlib,json,pathlib,statistics,time,math
R=pathlib.Path(__file__).resolve().parent
ARMS=['hp-cpu-a1','hp-gpu-b1','hp-gpu-b2','hp-cpu-a2']
CPU='b9506fb2fd48ff71b95c855294b7cc0cab442fd18e0655ee617d35b69d0a61c4'
GPU='69d61caa862e5b7558e2001d9266ef00583842a776fff923a0eed1092a80f363'
FAT='4387f0e32b257dd12f30f5bc1d2fad266a0bd2405e53f92d0b54f9d321c3ddf5'
def read(p):return json.loads((R/p).read_text())
def need(ok,msg):
 if not ok:raise ValueError(msg)
def digest(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
report={'status':'FAIL','arms':{},'pairs':{},'errors':[],'limits':['These are whole-request tokens/s at fixed output budgets, including prefill; generated trajectories can differ under CPU/GPU sampling RNG consumption and reductions.','Two swapped pairs on one rental are evidence for this workload, not a confidence interval or proof that HP beats DP.','Guard coverage is sampled at approximately1Hz, not a guarantee over unsampled subintervals. Guard wall alignment uses recorded server start epoch plus telemetry uptime.','A1 used an earlier fresh CPU server; source/binary differences, source-bound audit and matching unchanged CUDA module are retained for attribution. Quality and final promotion remain separate gates.']}
try:
 need((R/'receipts/HP_GPU_ABBA_DONE').exists(),'ABBA controller completion marker missing')
 need((R/'receipts/HP_GPU_AUDIT3_PASSED').exists(),'Source-bound v3 audit marker missing')
 gp=read('hp-sampling-v3-build-proof.json');cp=read('receipts/candidate-build-proof.json');sp=read('hp-sampling-v3-source.json')
 need(gp['source_receipt']['source_sha256']==sp['source_sha256'],'GPU build/source file hashes differ')
 need(set(gp['fatbins'].values())=={FAT} and cp['cuda_fatbin_sha256']==FAT and cp['candidate_binary_sha256']==CPU and cp['source_matches_local'],'build or CUDA binding mismatch')
 need(gp['binaries_sha256']['/root/volta-hp-sampling-v3/target-hp-sampling-v3/release/volta-server']==GPU,'GPU binary proof differs')
 report['source_binding']={'GPU_source_commit':sp['source_commit'],'GPU_source_sha256':sp['source_sha256'],'CPU_binary_sha256':CPU,'GPU_binary_sha256':GPU,'shared_fatbin_sha256':FAT,'CPU_build_receipt':'receipts/candidate-build-proof.json','GPU_build_receipt':'hp-sampling-v3-build-proof.json'}
 common=None;request_hash={}
 for name in ARMS:
  gpu='-gpu-' in name;c=read(f'server-runs/{name}/configuration.json');env=c['volta_environment']
  need(c['binary_sha256']==(GPU if gpu else CPU),f'{name}: binary identity')
  fields={'head_parallel':True,'ctx':262144,'slots':4,'graphs':4,'chunk':64,'spec_k':5,'theta':.35,'profile':False,'loop_profile':0,'graph_trace':False}
  need(all(c.get(k,0)==v for k,v in fields.items()),f'{name}: settings differ')
  need(not c.get('hp_sample_audit',0) and env.get('VOLTA_HP_SAMPLE_AUDIT','0')=='0',f'{name}: audit enabled')
  need((env.get('VOLTA_HP_GPU_SAMPLE')=='1')==gpu and c.get('hp_gpu_sample',False)==gpu,f'{name}: HP sampler flag differs')
  need(not any(env.get(k,'0') not in ['0',''] for k in ['VOLTA_PROF','VOLTA_STEPT','VOLTA_LOOP_PROF','VOLTA_GRAPH_TRACE']),f'{name}: profiling environment')
  normenv={k:v for k,v in env.items() if k not in ['VOLTA_HP_GPU_SAMPLE','VOLTA_HP_SAMPLE_AUDIT']}
  norm={'settings':fields,'model':c['model'],'environment':normenv,'command':c['command'][1:]}
  if common is None:common=norm
  need(norm==common,f'{name}: normalized full config differs')
  need((R/f'screens/{name}/ARM_DONE').exists(),f'{name}: ARM_DONE absent')
  guardc=read(f'memory-guards/{name}/configuration.json');guard=[json.loads(l) for l in (R/f'memory-guards/{name}/observations-through-arm.jsonl').read_text().splitlines()]
  need(guardc['first_observation_verified'] and guardc['server_pid']==c['pid'],f'{name}: guard admission PID')
  need(len(guard)>1 and all(x['pid']==c['pid'] and x['stop'] is False and x['threshold_mib']>=192 and len(x['gpus_index_free_used_mib'])==8 for x in guard),f'{name}: guard PID/stop/threshold/count')
  need(all({g[0] for g in x['gpus_index_free_used_mib']}==set(range(8)) for x in guard),f'{name}: guard GPU ordinals')
  low=min(g[1] for x in guard for g in x['gpus_index_free_used_mib']);need(low>=192,f'{name}: reserve breached')
  gaps=[b['epoch']-a['epoch'] for a,b in zip(guard,guard[1:])];need(min(gaps)>0 and max(gaps)<=2,f'{name}: guard continuity gap exceeds2s')
  arm={'config_sha256':digest(f'server-runs/{name}/configuration.json'),'binary_sha256':c['binary_sha256'],'pid':c['pid'],'gpu_sampling':gpu,'started_epoch':c['started_epoch'],'guard':{'observations':len(guard),'minimum_free_mib':low,'maximum_gap_s':max(gaps),'first_epoch':guard[0]['epoch'],'last_epoch':guard[-1]['epoch']},'phases':{}}
  for phase,prompts in [('fixed',16588),('mixed',20685)]:
   path=f'screens/{name}/{phase}';s=read(path+'/summary.json');req=read(path+'/requests.json');res=read(path+'/results.json');i=read(path+'/initial-stats.json');f=read(path+'/final-stats.json');h=digest(path+'/requests.json')
   request_hash.setdefault(phase,h);need(request_hash[phase]==h,f'{name}/{phase}: request JSON bytes differ')
   need(len(req)==len(res)==4 and all(x['max_tokens']==512 and x['temperature']==.7 and x['top_p']==1 for x in req),f'{name}/{phase}: request budget')
   need(all('error' not in x and x['usage']['completion_tokens']==512 and x['finish_reason']=='length' for x in res),f'{name}/{phase}: incomplete request')
   output=sum(x['usage']['completion_tokens'] for x in res);prompt=sum(x['usage']['prompt_tokens'] for x in res)
   need(output==2048 and prompt==prompts and f['tokens_out']-i['tokens_out']==2048 and f['tokens_prompt']-i['tokens_prompt']==prompts,f'{name}/{phase}: actual counters differ')
   need(i['active']==i['queued']==f['active']==f['queued']==0 and f['tokens_cached']-i['tokens_cached']==0,f'{name}/{phase}: idle/cache')
   need(s['all_hit_output_budget'] and s['completed']==4 and s['output_tokens']==s['requested_output_tokens']==s['server_tokens_out_delta']==2048 and s['prompt_tokens']==prompts and s['cached_tokens_delta']==0,f'{name}/{phase}: summary differs')
   wall=max(x['end_monotonic'] for x in res)-min(x['start_monotonic'] for x in res);tps=output/wall
   need(math.isclose(wall,s['wall_s'],rel_tol=1e-9) and math.isclose(tps,s['output_tokens_per_wall_s'],rel_tol=1e-9),f'{name}/{phase}: wall/tps recomputation')
   start=c['started_epoch']+i['uptime_s'];end=c['started_epoch']+f['uptime_s']
   need(guard[0]['epoch']<=start and guard[-1]['epoch']>=end and guardc['until']>=end,f'{name}/{phase}: guard does not bracket entire phase')
   arm['phases'][phase]={'wall_s':wall,'tps':tps,'output_tokens':output,'prompt_tokens':prompt,'cache_tokens':0,'request_sha256':h,'guard_phase_start_epoch_estimate':start,'guard_phase_end_epoch_estimate':end,'median_ttft_s':statistics.median(x['ttft_s'] for x in res)}
  report['arms'][name]=arm
 report['normalized_config']=common;report['request_sha256']=request_hash
 need([report['arms'][x]['started_epoch'] for x in ARMS]==sorted(report['arms'][x]['started_epoch'] for x in ARMS),'start order notA/B/B/A')
 for label,a,b in [('A1_to_B1','hp-cpu-a1','hp-gpu-b1'),('B2_vs_A2','hp-cpu-a2','hp-gpu-b2')]:
  report['pairs'][label]={p:{'CPU_tps':report['arms'][a]['phases'][p]['tps'],'GPU_tps':report['arms'][b]['phases'][p]['tps'],'gain_percent':100*(report['arms'][b]['phases'][p]['tps']/report['arms'][a]['phases'][p]['tps']-1)} for p in ['fixed','mixed']}
 need(all(x['gain_percent']>5 for v in report['pairs'].values() for x in v.values()),'not all swapped phase gains exceed5%')
 report['drift_percent']={p:{'CPU_A2_over_A1':100*(report['arms']['hp-cpu-a2']['phases'][p]['tps']/report['arms']['hp-cpu-a1']['phases'][p]['tps']-1),'GPU_B2_over_B1':100*(report['arms']['hp-gpu-b2']['phases'][p]['tps']/report['arms']['hp-gpu-b1']['phases'][p]['tps']-1)} for p in ['fixed','mixed']}
 report['status']='PASS';report['timing_gate_passed']=True
except Exception as e:
 report['errors'].append(str(e));report['timing_gate_passed']=False
(R/'receipts/hp-gpu-abba-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
if report['status']=='PASS':
 (R/'receipts/HP_GPU_MATCHED_TIMING_GATE_PASSED').write_text(json.dumps({'analysis_sha256':digest('receipts/hp-gpu-abba-analysis.json'),'analysis':'hp-gpu-abba-analysis.json','created_epoch':time.time(),'scope':'Matched HP timing only; not a promotion marker'},indent=2)+'\n')
 print(json.dumps({'status':'PASS','pairs':report['pairs'],'drift_percent':report['drift_percent']}))
else:
 marker=R/'receipts/HP_GPU_MATCHED_TIMING_GATE_PASSED'
 if marker.exists():marker.unlink()
 print(json.dumps(report['errors']));raise SystemExit(1)
