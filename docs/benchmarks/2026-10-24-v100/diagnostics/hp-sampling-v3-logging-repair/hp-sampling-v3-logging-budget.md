# HP audit v3 record logging repair

V2 completed all API cases and614valid numerical audits (8,796,728,320checkedbytes), but its strict log gate failed because one dispatch JSON record interleaved with a tracing request line. Preserve the raw malformed record and failed gate. Do not stitch it or ignore it to promote v2.

Repair only audit I/O: preformat the entire label+JSON+newline into one String, then call Write::write_all once on locked stderr. Both numerical audit and dispatch records use that helper; propagate write errors through their existing Result paths. Audit-off timing remains unchanged. No copy, numeric, sampler, table, eligibility or kernel change. Add a CPU formatter check that the complete record has exactly one terminating newline and valid JSON.

CPU build budget2jobs/180s, expected35–70s, exact frozen4387fatbin, new /root/volta-hp-sampling-v3/target-hp-sampling-v3. Preserve all v2 paths and hashes. Wait for parent stop/idle confirmation and independently check no live model/GPU compute process immediately before building. No GPU/model launch by this agent. Parent owns fresh strict audit/teacher/timing; successful v2 numerical records do not remove the malformed-log failure. Aim build proof before04:32UTC.
