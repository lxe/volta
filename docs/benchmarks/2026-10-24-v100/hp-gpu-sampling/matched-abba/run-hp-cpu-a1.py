import pathlib,subprocess,json,time
r=pathlib.Path(__file__).parent
subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/memory-hp4-loop-prof-001.jsonl',str(r/'lab/hp-copy-run/model-guard-through-copy.jsonl')],check=True)
name='hp-cpu-a1';args=['--engine','sampler','--name',name,'--ctx','262144','--slots','4','--head-parallel','--graphs','4']
subprocess.run(['python',str(r/'server-control.py'),*args],check=True)
subprocess.run(['python',str(r/'start-memory-guard.py'),'--name',name,'--reserve','192','--admit-free','512'],check=True)
(r/'screens'/name).mkdir(exist_ok=False)
subprocess.run(['python',str(r/'run-arm.py'),*args,'--pair','hp-gpu-01','--resume-existing'],check=True)
for remote,local in [('server-'+name+'.log',r/'screens'/name/'server.log'),('memory-'+name+'.jsonl',r/'memory-guards'/name/'observations-through-arm.jsonl')]:
 subprocess.run(['scp','-l','1900000','-F',str(r/'ssh.conf'),'volta-run:/root/'+remote,str(local)],check=True)
(r/'receipts/HP_CPU_A1_DONE').write_text(str(time.time())+'\n')
print('HP_CPU_A1_DONE',flush=True)
