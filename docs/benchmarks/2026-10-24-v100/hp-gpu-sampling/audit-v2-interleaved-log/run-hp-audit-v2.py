#!/usr/bin/env python3
"""Run one source-bound repaired HP audit and retain evidence even when validation fails."""
import pathlib,subprocess,json,hashlib,time
r=pathlib.Path(__file__).parent;w=pathlib.Path('/home/lxe/volta-hp-sampling')
proof=json.loads((r/'hp-sampling-v2-build-proof.json').read_text())
commit=(r/'hp-sampling-v2-source-commit.txt').read_text().strip()
assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
for path,expected_hash in proof['source_receipt']['source_sha256'].items():
 assert hashlib.sha256((w/path).read_bytes()).hexdigest()==expected_hash
 data=subprocess.check_output(['git','-C',str(w),'show',commit+':'+path]);assert hashlib.sha256(data).hexdigest()==expected_hash
assert set(proof['fatbins'].values())=={'4387f0e32b257dd12f30f5bc1d2fad266a0bd2405e53f92d0b54f9d321c3ddf5'}
assert not proof['compiler_processes']
binary='/root/volta-hp-sampling-v2/target-hp-sampling-v2/release/volta-server';expected=proof['binaries_sha256'][binary]
name='hp-gpu-audit2';args=['--engine','hp-sample','--name',name,'--ctx','262144','--slots','4','--head-parallel','--graphs','4','--hp-gpu-sample','--hp-sample-audit','1024']
subprocess.run(['python',str(r/'server-control.py'),*args],check=True)
cfg=json.loads((r/'server-runs'/name/'configuration.json').read_text());assert cfg['binary']==binary and cfg['binary_sha256']==expected
subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--reserve','192','--admit-free','512'],check=True)
(r/'screens'/name).mkdir(exist_ok=False)
client=subprocess.run([str(r/'venv/bin/python'),str(r/'hp-sampling-audit-client.py'),'--name',name])
copy_results=[]
for remote,local in [('server-'+name+'.log',r/'screens'/name/'server.log'),('memory-'+name+'.jsonl',r/'memory-guards'/name/'observations-through-audit.jsonl')]:
 p=subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/'+remote,str(local)],capture_output=True,text=True)
 copy_results.append({'remote_file':remote,'returncode':p.returncode,'stderr':p.stderr})
(r/'screens'/name/'evidence-copy-status.json').write_text(json.dumps(copy_results,indent=2))
assert all(x['returncode']==0 for x in copy_results),'Evidence copy failed; inspect before proceeding'
analysis=subprocess.run(['python',str(r/'analyze-hp-sampling-audit.py'),'--log',str(r/'screens'/name/'server.log'),'--out',str(r/'receipts/hp-gpu-audit2-analysis.json'),'--source-commit',commit,'--expected-server-sha',expected])
guard=[json.loads(x) for x in (r/'memory-guards'/name/'observations-through-audit.jsonl').read_text().splitlines()]
assert guard and not any(x['stop'] for x in guard) and all(x['pid']==cfg['pid'] for x in guard)
low=min(g[1] for row in guard for g in row['gpus_index_free_used_mib']);assert low>=192
receipt={'client_exit_code':client.returncode,'analyzer_exit_code':analysis.returncode,'actual_server_pid':cfg['pid'],'actual_server_binary_sha256':cfg['binary_sha256'],'source_commit':commit,'guard_observations':len(guard),'minimum_free_mib':low,'evidence_copied':copy_results}
(r/'receipts/hp-gpu-audit2-launch-binding.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
assert client.returncode==analysis.returncode==0
(r/'receipts/HP_GPU_AUDIT2_PASSED').write_text(str(time.time())+'\n')
print('HP_GPU_AUDIT2_PASSED',flush=True)
