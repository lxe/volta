# HP sampling audit parser

Run CPU-only after copying the complete fresh candidate server log:

    python3 /home/lxe/volta-sixhour-20261024/analyze-hp-sampling-audit.py --log SERVER.log --out SUMMARY.json

The parser always writes a summary for missing files, malformed records, runtime audit/worker failure or missing coverage, and exits nonzero when the audit log gate fails. The output path must differ from the original log. The parser never starts or accesses a model/GPU. Parent must separately bind the log to the expected controller configuration, serverSHA63f597b080dee291d0f146d88d6cef70c075f546d82dcfec192880c13c7d6820, and sourcecommitacf798786b1962f5ea45ecbebc6cf114f2126b81. A schema binding is not proof that an arbitrary log came from that binary.

Every audit must report true bitwise collection and supplied-table/uniform replay parity; checked_u32 must equal logit_rows*vocab with1..64rows and vocab129280. Each table validates rb+nd<logit_rows,0<=nd<=8,0<=accepted<=nd, draft/token ranges, greedy boolean and all three16hex-digit fingerprints. Audit IDs must start atzero and remain consecutive in this one process. Each record family is capped at4096, matching the source audit maximum; retained error examples and coverage witnesses are bounded. Fingerprint syntax is checked; hashes cannot be recomputed from logs because full tensors/tables are not retained.

The required audit coverage is rb63; speculative nd>0; at least one accepted draft; at least one rejected draft with nd>0; all drafts accepted with nd>0; multiple tables; mixed greedy/sample tables in one audit; and a matrix with t<64. The last condition establishes only a smaller logical matrix, not its prefill/decode phase. Separate dispatch records must demonstrate CPU top-p<1 fallback, CPU positive hesitation fallback, and entirely greedy StepGreedy. Dispatch row extents and the exact built eligibility predicate are validated. Submitted request settings alone never satisfy coverage.

Sixteen synthetic positive/negative fixtures validate ANSI handling, complete coverage, missing categories, malformed/duplicateJSON, missingaudit, parity/count/fingerprint/bounds failures, duplicateauditIDs, wrong dispatch flags and explicitruntimefailure. Their receipts are SYNTHETIC_PARSER_VALIDATION_ONLY; no actual candidate result is inferred. The main summary's audit_log_gate_passed covers this log gate only. Actual teacher quality, continuous memory, matched A/B/B/A performance and long-context task gates remain separate.
