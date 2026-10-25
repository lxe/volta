# HP4/256K CPU sampling profile

V3 separates the sampled CPU engine step from subsequent CPU distribution and RNG updates. This actual same-process profile produced2048tokens with zero cache reuse in each primary phase. Fixed CPU sampling was8.90630s/40.858084s=21.7981%; mixed was10.23597s/48.132269s=21.2663%. Median per-pass CPU sampling was17.64/17.34ms. These instrumented measurements pass the five-percent feasibility gate; no optimized implementation or throughput gain is established.

The parser validates V3 binary/config, strips ANSI and assigns each profile record to active request IDs. Source and actual logs show the profile precedes the final retire/done. Its original synthetic fixture assumed reverse order and was superseded after actual data triggered a cross-phase rejection; that receipt is preserved with an invalid-assumption label. Actual totals use the corrected parser. Idle iter_ms is excluded from the denominator. Rayon duration is counted once. Rounding bounds are2.425/2.870ms for complete phase sums.

The verified guard supplied98observations with minimum1745MiB free across8GPUs and no stop. This receipt belongs to this HP profile, not the previous four-long capacity run. The hypothetical31.56MiB gather allocation leaves a calculated1713.44MiB from that minimum; candidate startup and workload still require their own guard.

The source-only HP GPU-sampling proposal and NVIDIA12.9.1 API references are included. No implementation/build/GPU experiment for that proposal occurred. Mega remains rejected based on historical V100 regressions and unchanged source mechanisms. V3 itself compiled03:30:09–03:31:06UTC with the exact frozen fatbin; build proof preserves V1/V2 hashes and confirms no compiler remained.
