# Recovered Scout expert-kernel research

These CPU-only feasibility notes recover the previously named vlut2 source under /home/lxe/volta-box-archive/scout. They identify exact source hashes and native CB3/TP8 compatibility requirements, correct the shared-expert byte budget, and define a conditional bounded real-data confirmation protocol. No implementation, build or GPU experiment was performed for these notes.

The copied REPORT_EXPERT.md and REPORT_RS.md are historical lab reports from another hardware window. Their timing figures, correctness statements and proposed gains are not current-rental measurements or validated model throughput gains. REPORT_RS records that a promising lab stream reorder actually slowed its engine11.33→11.48ms. The original generated expert test uses real sliced dimensions but generated values and limited routing; it cannot establish the present four-agent behavior.

The current decision is to stop this candidate for now because no exact real consumed FP16 routed-input plus routing snapshot was found, and the five-way atomic sum needs a preagreed whole-output numerical contract. The conditional budget does not authorize a build, benchmark or engine change. The existing SHARED_EARLY flag is HP-only and has no effect on primary DP. No positive gain is booked.

Large kernel sources and binaries are not duplicated here. Their original local paths and SHA256 fingerprints are recorded in the feasibility review, and recovering them does not mean they are active in the current engine. The manifest records every other archived file's SHA256 and byte size.
