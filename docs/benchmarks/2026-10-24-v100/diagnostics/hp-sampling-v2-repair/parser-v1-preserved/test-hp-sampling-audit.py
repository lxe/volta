#!/usr/bin/env python3
"""Synthetic format/coverage/error checks; no model/GPU evidence."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys
r=Path(__file__).parent;d=r/'receipts/hp-sampling-parser-synthetic';d.mkdir(exist_ok=True)
def row(rb,nd,greedy,accepted):return {'rb':rb,'nd':nd,'greedy':greedy,'drafts':list(range(nd)),'accepted':accepted,'token':17}
def audit(i,t,rows):return {'audit_id':i,'logit_rows':t,'vocab':129280,'checked_u32':t*129280,'logits_equal':True,'sampler_replay_equal':True,'logits_fnv_u32':f'{i+1:016x}','table_ints_fnv_u32':'0123456789abcdef','table_floats_fnv_u32':'fedcba9876543210','rows':rows}
def dr(rb,n,prefill,greedy,top_p=1.,hes=None):return {'rb':rb,'rows':n,'prefill':prefill,'greedy':greedy,'top_p':top_p,'hesitation':hes is not None,'hesitation_lambda':hes,'draft_bound_ok':prefill or n<=9}
def dispatch(command,rows):return {'command':command,'gpu':command=='StepSampleHp','sampled':command!='StepGreedy','gpu_sample_enabled':True,'step_rows':sum(x['rows'] for x in rows),'table_rows_if_gpu':len(rows) if command=='StepSampleHp' else 0,'rows':rows}
a0=audit(0,64,[row(0,2,True,2),row(63,0,False,0)]);a1=audit(1,12,[row(0,5,False,0),row(6,5,True,5)])
records=[('HP_SAMPLE_DISPATCH',dispatch('StepSampleHp',[dr(0,3,False,True),dr(3,61,True,False)])),('HP_SAMPLE_AUDIT',a0),('HP_SAMPLE_DISPATCH',dispatch('StepSampleHp',[dr(0,6,False,False),dr(6,6,False,True)])),('HP_SAMPLE_AUDIT',a1),('HP_SAMPLE_DISPATCH',dispatch('Step',[dr(0,1,False,False,.9)])),('HP_SAMPLE_DISPATCH',dispatch('Step',[dr(0,1,False,False,hes=.3)])),('HP_SAMPLE_DISPATCH',dispatch('StepGreedy',[dr(0,1,False,True)]))]
def encode(rs):return '\n'.join('\x1b[32m'+tag+' '+json.dumps(data)+'\x1b[0m' for tag,data in rs)+'\n'
cases={'positive':(encode(records),True),'missing_coverage':(encode(records[:2]),False),'malformed_json':('HP_SAMPLE_AUDIT {\n',False),'empty_log':('ordinary startup\n',False)}
for name,field,value in [('bad_checked','checked_u32',1),('false_parity','logits_equal',False),('bad_fingerprint','logits_fnv_u32','xyz')]:
 rs=copy.deepcopy(records);rs[1][1][field]=value;cases[name]=(encode(rs),False)
for name,field,value in [('bad_rb','rb',64),('bad_accepted','accepted',3),('bad_token','token',129280),('bad_nd','nd',9),('bool_as_integer','rb',True)]:
 rs=copy.deepcopy(records);rs[1][1]['rows'][0][field]=value;cases[name]=(encode(rs),False)
rs=copy.deepcopy(records);rs[3][1]['audit_id']=0;cases['duplicate_audit_id']=(encode(rs),False)
rs=copy.deepcopy(records);rs[4][1]['gpu']=True;cases['wrong_fallback_gpu_flag']=(encode(rs),False)
cases['duplicate_json_field']=('HP_SAMPLE_AUDIT {"audit_id":0,"audit_id":0}\n',False)
cases['explicit_runtime_failure']=(encode(records)+'GPU0 HP sampled step failed: HP collected-logit audit failed\n',False)
results=[]
for name,(text,expected) in cases.items():
 log=d/(name+'.log');out=d/(name+'.json');log.write_text(text)
 p=subprocess.run([sys.executable,str(r/'analyze-hp-sampling-audit.py'),'--log',str(log),'--out',str(out)],capture_output=True,text=True)
 need=json.loads(out.read_text());actual=p.returncode==0
 assert actual==expected and need['audit_log_gate_passed']==expected,(name,p.returncode,p.stdout,p.stderr)
 results.append({'case':name,'expected_pass':expected,'exit_code':p.returncode,'status':need['status'],'errors_total':need['errors_total'],'missing':need['missing_required_coverage']})
summary={'evidence':'SYNTHETIC_PARSER_VALIDATION_ONLY','passed':True,'cases':results,'parser_sha256':hashlib.sha256((r/'analyze-hp-sampling-audit.py').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Synthetic parser checks PASS:',len(results),'cases; no actual GPU evidence.')
