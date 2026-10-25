#!/usr/bin/env python3
"""CPU-only actual V3 loop counters aligned to same-process request events."""
import argparse,hashlib,json,re,statistics
from datetime import datetime
from pathlib import Path
ANSI=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
FIELDS=re.compile(r'(\w+)=("[^"]*"|[^ ]+)')

def parse(text,summaries):
 lines=ANSI.sub('',text).splitlines();admissions=[];done={};profiles=[]
 for line in lines:
  if not any(s in line for s in (' admitted ',' done ','loop profile')):continue
  f={k:v.strip('"') for k,v in FIELDS.findall(line)}
  stamp=line.split()[0];datetime.fromisoformat(stamp.replace('Z','+00:00'))
  if ' admitted ' in line:admissions.append((stamp,f))
  elif ' done ' in line:done[f['id']]=(stamp,f)
 assert [int(f['prompt_tokens']) for _,f in admissions]==[39]+[563]*8+[4145]*7+[8242], 'unexpected request sequence'
 assert len(done)==len(admissions)==17,'missing/duplicate completions'
 phaseids={f['id']:('fixed' if 9<=i<13 else 'mixed' if 13<=i<17 else 'prefix') for i,(_,f) in enumerate(admissions)}
 active=set();last_stamp=None
 for line in lines:
  if not any(s in line for s in (' admitted ',' done ','loop profile')):continue
  f={k:v.strip('"') for k,v in FIELDS.findall(line)};stamp=line.split()[0]
  if ' admitted ' in line:
   active.add(f['id'])
  elif ' done ' in line:
   assert f['id'] in active,'completion without active request'
   active.remove(f['id'])
  else:
   assert 'cpu_sample_ms' in f,'not V3 accounting log'
   tags={phaseids[i] for i in active}
   profiles.append({'utc':stamp,'phases':sorted(tags),**{k:float(f[k]) for k in ['iter_ms','spark_ms','step_ms','rewind_ms','cpu_sample_ms','rest_ms']}})
   last_stamp=stamp
 assert not active
 out={}
 for phase in ('fixed','mixed'):
  req=[(s,f) for s,f in admissions if phaseids[f['id']]==phase];rows=[x for x in profiles if phase in x['phases']]
  assert rows and all(x['phases']==[phase] for x in rows),'cross-phase accounting interval'
  s=summaries[phase];assert s['completed']==4 and s['output_tokens']==2048 and s['cached_tokens_delta']==0
  assert sum(int(done[f['id']][1]['generated']) for _,f in req)==2048
  wall=s['wall_s'];cpu=sum(x['cpu_sample_ms'] for x in rows);frac=cpu/(wall*1000)
  out[phase]={'admitted_utc':min(x[0] for x in req),'done_utc':max(done[f['id']][0] for _,f in req),'profile_first_utc':rows[0]['utc'],'profile_last_utc':rows[-1]['utc'],'profile_records':len(rows),'SSE_wall_s':wall,'cpu_sample_sum_ms':cpu,'cpu_sample_wall_fraction':frac,'cpu_sum_rounding_bound_ms':.005*len(rows),'cpu_wall_fraction_rounding_bounds':[(cpu-.005*len(rows))/(wall*1000),(cpu+.005*len(rows))/(wall*1000)],'cpu_distribution_5percent_gate':(cpu-.005*len(rows))/(wall*1000)>=.05,'step_sum_ms':sum(x['step_ms'] for x in rows),'cpu_sample_median_ms':statistics.median(x['cpu_sample_ms'] for x in rows),'max_counter_accounting_rounding_error_ms':max(abs(x['iter_ms']-sum(x[k] for k in ['spark_ms','step_ms','rewind_ms','cpu_sample_ms','rest_ms'])) for x in rows),'output_tokens':s['output_tokens'],'cached_tokens_delta':s['cached_tokens_delta']}
 return {'evidence':'ACTUAL_INSTRUMENTED_V3_HOST_TIMING','phases':out,'limits':['CPU distribution Rayon work is timed once around join; no per-thread sum.','Source logs loop profile before the final retire/done; each record is assigned to request IDs active at that line. No completed phase is carried into the next phase.','iter_ms includes idle queue waits and is never the phase denominator.','step_ms includes HP GPU work/download and host vocabulary merge; cpu_sample_ms excludes that work.','Rounded per-pass ms fields have at most 0.005ms error per record per field.','STEPT synchronization and logging perturb runtime; this is an attribution screen, not primary throughput.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='hp4-loop-prof-001');a=p.parse_args();r=Path(__file__).parent
 assert (r/'receipts/HP_LOOP_PROFILE_DONE').exists(),'actual completion receipt required'
 screen=r/'screens'/a.run;cfgp=r/'server-runs'/a.run/'configuration.json';cfg=json.loads(cfgp.read_text())
 assert cfg['binary_sha256']=='c50eb1df6dc084b116dfe26a07a750ceb9e8b7db3797c38e2f4e956637de5972'
 assert cfg['head_parallel'] and cfg['ctx']==262144 and cfg['slots']==4 and cfg['profile'] and cfg['loop_profile']==1
 paths=[cfgp,screen/'server.log',*[screen/k/'summary.json' for k in ('fixed','mixed')]]
 out=parse((screen/'server.log').read_text(),{k:json.loads((screen/k/'summary.json').read_text()) for k in ('fixed','mixed')});out['configuration']=cfg
 out['input_sha256']={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 dest=r/'receipts'/f'{a.run}-CPU-analysis.json';dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['phases'],indent=2))
if __name__=='__main__':main()
