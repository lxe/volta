#!/usr/bin/env python3
"""Evaluate the precommitted threshold gate on actual fresh-server A/B/B/A receipts."""
import pathlib,json,hashlib,shutil,statistics
r=pathlib.Path(__file__).parent
assert (r/'receipts/SPECULATION_ABBA_DONE').exists()
names=['spec-a1','spec-b1','spec-b2','spec-a2'];phases=['fixed','mixed','long']
d={n:{p:json.loads((r/'screens'/n/p/'summary.json').read_text()) for p in phases} for n in names}
configs={n:json.loads((r/'server-runs'/n/'configuration.json').read_text()) for n in names}
assert {c['binary_sha256'] for c in configs.values()}=={'b9506fb2fd48ff71b95c855294b7cc0cab442fd18e0655ee617d35b69d0a61c4'}
for p in phases:
 requests=[json.loads((r/'screens'/n/p/'requests.json').read_text()) for n in names]
 assert all(x==requests[0] for x in requests)
 assert len({d[n][p]['prompt_tokens'] for n in names})==1
 for n in names:
  s=d[n][p];assert s['completed']==4 and s['all_hit_output_budget'] and s['cached_tokens_delta']==0
  assert s['output_tokens']==(4096 if p=='long' else 2048) and s['server_tokens_out_delta']==s['output_tokens']
assert [configs[n]['theta'] for n in names]==[.35,.2,.2,.35]
assert all(c['ctx']==131072 and c['slots']==1 and not c['head_parallel'] and c['spec_k']==5 and not c['profile'] for c in configs.values())
guards={}
for n in names:
 rows=[json.loads(s) for s in (r/'memory-guards'/n/'observations-through-arm.jsonl').read_text().splitlines()]
 assert rows and not any(x['stop'] for x in rows) and all(x['pid']==configs[n]['pid'] for x in rows)
 low=min(g[1] for x in rows for g in x['gpus_index_free_used_mib']);assert low>=192
 guards[n]={'observations':len(rows),'minimum_free_mib':low,'max_observation_gap_s':max((b['epoch']-a['epoch'] for a,b in zip(rows,rows[1:])),default=0)}
pairs=[]
for a,b in [('spec-a1','spec-b1'),('spec-a2','spec-b2')]:
 ratios={p:d[b][p]['output_tokens_per_wall_s']/d[a][p]['output_tokens_per_wall_s'] for p in phases}
 ttft=max(d[b]['mixed']['ttft_s'])/max(d[a]['mixed']['ttft_s'])
 pairs.append({'baseline':a,'candidate':b,'throughput_ratio':ratios,'gain_percent':{p:(v-1)*100 for p,v in ratios.items()},'mixed_worst_ttft_ratio':ttft,'primary_at_least_5_percent':ratios['long']>=1.05,'mixed_within_3_percent':ratios['mixed']>=.97 and ttft<=1.03})
report={'experiment':'Single theta0.2 hypothesis against theta0.35; no source or quantization change','order':names,'throughput_tps':{n:{p:d[n][p]['output_tokens_per_wall_s'] for p in phases} for n in names},'pairs':pairs,'baseline_A2_over_A1_TPS':{p:d['spec-a2'][p]['output_tokens_per_wall_s']/d['spec-a1'][p]['output_tokens_per_wall_s'] for p in phases},'candidate_B2_over_B1_TPS':{p:d['spec-b2'][p]['output_tokens_per_wall_s']/d['spec-b1'][p]['output_tokens_per_wall_s'] for p in phases},'guards':guards,'promotion_gate_passed':all(x['primary_at_least_5_percent'] and x['mixed_within_3_percent'] for x in pairs),'limits':['Prompts, seeds and full output work match; sampled text and routing can differ.','Rolling acceptance summaries do not identify the scheduler EWMA or per-position prefix probabilities.','This gate is configuration-specific; no global optimum is established.']}
(r/'receipts/speculation-abba-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
