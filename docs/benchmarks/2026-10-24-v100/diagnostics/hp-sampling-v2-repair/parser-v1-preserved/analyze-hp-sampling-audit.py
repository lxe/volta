#!/usr/bin/env python3
"""Validate the exact opt-in HP sampling JSON logs; CPU-only, no inferred request coverage."""
import argparse, hashlib, json, math, re, sys
from collections import Counter
from pathlib import Path
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
ERROR = re.compile(r'(?i)(HP[^\n]*(?:audit|sample)[^\n]*(?:failed|mismatch)|GPU\d+[^\n]*(?:step failed|setup failed)|CUDA_ERROR_|illegal memory access|thread .* panicked|collected byte mismatch)')
COVERAGE = ('row_base_63', 'speculative_nd_positive', 'accepted_positive', 'draft_rejection', 'all_drafts_accepted',
            'multiple_tables', 'mixed_greedy_sample_tables', 'matrix_t_lt64', 'cpu_top_p_fallback',
            'cpu_positive_hesitation_fallback', 'pure_greedy_step')

def need(ok, msg):
    if not ok: raise ValueError(msg)
def integer(x, low, high, name):
    need(type(x) is int and low <= x <= high, f'{name}: expected integer {low}..{high}')
def boolean(x, name): need(type(x) is bool, f'{name}: expected bool')
def finite(x, name): need(type(x) in (float, int) and math.isfinite(x), f'{name}: expected finite number')
def strict_object(pairs):
    d = {}
    for k,v in pairs:
        need(k not in d, f'duplicate JSON field {k}'); d[k] = v
    return d

def audit_check(a):
    need(type(a) is dict, 'audit must be an object')
    integer(a['audit_id'], 0, 2**63-1, 'audit_id'); integer(a['logit_rows'], 1, 64, 'logit_rows')
    integer(a['vocab'], 129280, 129280, 'vocab'); integer(a['checked_u32'], 1, 64*129280, 'checked_u32')
    need(a['checked_u32'] == a['logit_rows']*a['vocab'], 'checked_u32 != logit_rows*vocab')
    need(a['logits_equal'] is True and a['sampler_replay_equal'] is True, 'gathered/replay parity is not true')
    for k in ('logits_fnv_u32', 'table_ints_fnv_u32', 'table_floats_fnv_u32'):
        need(type(a[k]) is str and re.fullmatch('[0-9a-f]{16}', a[k]) is not None, f'{k}: invalid u32 fingerprint encoding')
    rows=a['rows'];need(type(rows) is list and 1 <= len(rows) <= 64, 'invalid audit table row count')
    for i,r in enumerate(rows):
        need(type(r) is dict, f'table {i}: not an object')
        integer(r['rb'], 0, 63, 'rb');integer(r['nd'], 0, 8, 'nd');boolean(r['greedy'], 'greedy')
        need(r['rb']+r['nd'] < a['logit_rows'], f'table {i}: rb+nd exceeds matrix')
        integer(r['accepted'], 0, r['nd'], 'accepted');integer(r['token'], 0, a['vocab']-1, 'token')
        need(type(r['drafts']) is list and len(r['drafts'])==r['nd'], f'table {i}: draft count differs from nd')
        for token in r['drafts']: integer(token, 0, a['vocab']-1, 'draft token')
    return {
        'row_base_63':any(r['rb']==63 for r in rows),
        'speculative_nd_positive':any(r['nd']>0 for r in rows),
        'accepted_positive':any(r['accepted']>0 for r in rows),
        'draft_rejection':any(r['nd']>0 and r['accepted']<r['nd'] for r in rows),
        'all_drafts_accepted':any(r['nd']>0 and r['accepted']==r['nd'] for r in rows),
        'multiple_tables':len(rows)>1,
        'mixed_greedy_sample_tables':{r['greedy'] for r in rows}=={False,True},
        'matrix_t_lt64':a['logit_rows']<64,
    }

def dispatch_check(d):
    need(type(d) is dict, 'dispatch must be an object')
    need(d['command'] in ('StepSampleHp','Step','StepGreedy'), 'unknown HP command')
    for k in ('gpu','sampled','gpu_sample_enabled'):boolean(d[k],k)
    need(d['gpu']==(d['command']=='StepSampleHp'), 'GPU flag differs from command')
    need(d['sampled']==(d['command']!='StepGreedy'), 'sampled flag differs from command')
    integer(d['step_rows'],1,64,'step_rows');integer(d['table_rows_if_gpu'],0,64,'table_rows_if_gpu')
    rows=d['rows'];need(type(rows) is list and 1<=len(rows)<=64,'invalid dispatch rows')
    need(d['table_rows_if_gpu']==(len(rows) if d['gpu'] else 0),'dispatch table count mismatch')
    end=0;eligible=True
    for r in rows:
        integer(r['rb'],0,63,'dispatch rb');integer(r['rows'],1,64,'dispatch row extent')
        need(r['rb']==end,'dispatch rows do not form contiguous caller plan');end+=r['rows']
        for k in ('prefill','greedy','hesitation','draft_bound_ok'):boolean(r[k],k)
        finite(r['top_p'],'top_p')
        if r['hesitation_lambda'] is not None:finite(r['hesitation_lambda'],'hesitation_lambda')
        need(r['hesitation']==(r['hesitation_lambda'] is not None),'hesitation/lambda presence mismatch')
        need(r['draft_bound_ok']==(r['prefill'] or r['rows']<=9),'draft bound flag inconsistent')
        eligible &= not r['hesitation'] and (r['greedy'] or r['top_p']>=1) and r['draft_bound_ok']
    need(end==d['step_rows'],'dispatch row extents differ from step_rows')
    need(d['sampled']==any(not r['greedy'] for r in rows),'dispatch sampled flag differs from greedy rows')
    need(d['gpu']==(d['sampled'] and d['gpu_sample_enabled'] and eligible),'dispatch differs from built eligibility predicate')
    return {
        'cpu_top_p_fallback':d['command']=='Step' and any(not r['greedy'] and r['top_p']<1 for r in rows),
        'cpu_positive_hesitation_fallback':d['command']=='Step' and any(r['hesitation'] and r['hesitation_lambda']>0 for r in rows),
        'pure_greedy_step':d['command']=='StepGreedy' and all(r['greedy'] for r in rows),
    }

def analyze(path):
    coverage={k:[] for k in COVERAGE};counts=Counter();errors=[];errors_total=0;row_hist=Counter();nd_hist=Counter();cmd_hist=Counter();checked=0;last_id=-1;hashes=set();h=hashlib.sha256()
    def error(line, detail):
        nonlocal errors_total
        errors_total+=1
        if len(errors)<100:errors.append({'line':line,'error':str(detail)[:512]})
    try:
        with path.open('rb') as stream:
            for lineno,raw in enumerate(stream,1):
                h.update(raw);line=ANSI.sub('',raw.decode('utf-8',errors='replace')).strip()
                if ERROR.search(line):error(lineno,line)
                tags=[x for x in ('HP_SAMPLE_AUDIT','HP_SAMPLE_DISPATCH') if x+' ' in line]
                if not tags:continue
                counts['target_records']+=1
                if len(tags)!=1 or len(line)>65536:error(lineno,'ambiguous/oversize record');continue
                tag=tags[0];counts[tag]+=1
                if counts[tag]>4096:error(lineno,'record cap exceeds built source maximum4096');continue
                try:
                    a=json.loads(line.split(tag+' ',1)[1],object_pairs_hook=strict_object,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON constant '+x)))
                    got=audit_check(a) if tag=='HP_SAMPLE_AUDIT' else dispatch_check(a)
                    if tag=='HP_SAMPLE_AUDIT':
                        need(a['audit_id']==last_id+1,'audit IDs must start at zero and be consecutive in one fresh log');last_id=a['audit_id']
                        row_hist[a['logit_rows']]+=1;checked+=a['checked_u32'];hashes.add(a['logits_fnv_u32'])
                        for row in a['rows']:nd_hist[row['nd']]+=1
                    else:cmd_hist[a['command']]+=1
                    counts[tag+'_valid']+=1
                    for k,hit in got.items():
                        if hit and len(coverage[k])<8:coverage[k].append({'line':lineno,**({'audit_id':a['audit_id']} if tag=='HP_SAMPLE_AUDIT' else {'command':a['command']})})
                except (ValueError,KeyError,TypeError,OverflowError) as e:error(lineno,e)
    except OSError as e:error(0,e)
    if not counts['HP_SAMPLE_AUDIT_valid']:error(0,'at least one valid parity audit is required')
    missing=[k for k,v in coverage.items() if not v]
    return {'status':'PASS' if not errors_total and not missing else 'FAIL','audit_log_gate_passed':not errors_total and not missing,
        'input_log':str(path),'input_sha256':h.hexdigest(),'source_schema_binding':{'engine_commit':'acf798786b1962f5ea45ecbebc6cf114f2126b81','expected_server_sha256':'63f597b080dee291d0f146d88d6cef70c075f546d82dcfec192880c13c7d6820','binary_identity_verified_here':False},'counts':dict(counts),'checked_u32_total':checked,'checked_bytes_total':checked*4,
        'matrix_rows_histogram':dict(sorted(row_hist.items())),'draft_nd_histogram':dict(sorted(nd_hist.items())),
        'command_histogram':dict(cmd_hist),'distinct_logit_fingerprints':len(hashes),'coverage':{k:{'observed':bool(v),'evidence':v} for k,v in coverage.items()},
        'missing_required_coverage':missing,'errors_total':errors_total,'errors_first100':errors,
        'limits':['This gate covers observed audit/dispatch records only; teacher quality, memory, performance and long-context gates remain separate.',
          'Fingerprints are validated for encoding; original full tensors/tables are not retained in logs, so their hashes cannot be independently recomputed here.',
          't<64 establishes a smaller logical matrix, not a prefill/decode phase classification.',
          'Coverage is never inferred from submitted requests; CPU fallback and greedy coverage require dispatch records.',
          'IDs are checked within one fresh source-bound server log; do not concatenate independent runs.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--log',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.log.resolve()==a.out.resolve():p.error('--out must differ from the original input log')
    out=analyze(a.log);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'audits':out['counts'].get('HP_SAMPLE_AUDIT_valid',0),'missing':out['missing_required_coverage'],'errors':out['errors_total'],'out':str(a.out)}))
    return 0 if out['audit_log_gate_passed'] else 1
if __name__=='__main__':sys.exit(main())
