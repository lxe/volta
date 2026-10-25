# Strict8 cache CPU hypothesis

The strict8resident hypothesis is rejected under both its per-GPU robustness criterion and a separately predeclared all-eight collective-step criterion. No policy was implemented, built or run on GPUs. The actual input traces and failed strict4policies are preserved unchanged in the parent graph/ and profile/ folders from commit60c1cd3.

Actual legacy nominal-four admission retained6/8/8/8graphs on active GPUs in the normal trace and6/9/8/8in the profile. Final free memory was349/319/319/319MiB normal and349/297/319/319MiB profile. A separately written strict8budget allowed45MiB possible steady growth onGPU0 and46MiB transient, leaving at least251MiB by its model versus a192MiBreserve. Graph16residents remained rejected. These are budget estimates; no actual eight-resident implementation/resource trial occurred.

The CPU model changes capacity only, retaining64-step frequency history, minimum8candidate observations, advantage4, at least64steps between replacements and16total trial replacements/GPU. Normal-only per-GPU primary coverage passes, but the independent profile prefix failsGPU1fixed:3.74ppgain against a15ppthreshold. Sixteen is a trial safety quota; a sustained design would need a rolling rate/resource/repayment contract instead of permanently stopping adaptation.

The parent then requested a NEW criterion reflecting the shared collective step. It was written before computing results and preserved the original failure. A joint hit requires ALL EIGHT GPUs, including t0expert/exchange workers, to replay a graph on the same sequence. Captures are not hits. The criterion requires>=15ppjoint gain and<=5% measured-capture-median proxy on each phase of BOTH actual prefixes. It is an opportunity screen, not a throughput model.

| Prefix/phase | Actual all8hits | Strict8all8hits | Steps | Gain |
|---|---:|---:|---:|---:|
| Normal fixed |8|51|308|13.96pp|
| Normal mixed |8|40|363|8.82pp|
| Profile fixed |8|10|294|0.68pp|
| Profile mixed |8|95|375|23.20pp|

Three of four coverage gates fail. Measured-capture-median proxies are1.63-2.64%wall, while the conservative100msstress charges remain6.25-9.24%; neither rescues a failed coverage gate. Mixed statistics concern the visible prefix, with approximately1.3s of final decoder tail absent. Simulations always start at first smoke and retain cache state before selecting each run's own primary interval.

The existing admission issue is real, but these results do not demonstrate that this frequency/rate/capacity combination improves four-agent serving. No gain is booked and no additional threshold change or policy implementation follows automatically. Raw traces, hashes and same-run phase boundaries remain in the parent archive; this folder adds only the new written hypotheses, CPU results, exact CPU source copies and final memory samples. Its own sha256.json binds these additions; the parent manifest remains the original evidence snapshot.
