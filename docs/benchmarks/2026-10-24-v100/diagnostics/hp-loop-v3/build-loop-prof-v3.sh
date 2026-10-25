#!/bin/bash
set -euo pipefail
# Invoke only in a parent-approved CPU build window. This never launches CUDA/model work.
test "${1:-}" = BUILD_WINDOW_CONFIRMED
exec >/root/build-loop-prof-v3-launch.log 2>&1
trap 'echo GRAPH_DIAG_BUILD_FAILED line=$LINENO' ERR
cd /root/volta-loop-prof-v3
# Verify every kernel source matches frozen source before reusing its exact sm70 fatbin.
python3 - <<'PY'
import pathlib,hashlib
base=pathlib.Path('/root/volta/engine/src/kernels');new=pathlib.Path('/root/volta-loop-prof-v3/engine/src/kernels')
a={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file()}
b={str(p.relative_to(new)):hashlib.sha256(p.read_bytes()).hexdigest() for p in new.rglob('*') if p.is_file()}
assert a==b,'kernel source mismatch'
PY
# Fully separate files/target; reflink is allowed, hardlinks are not used.
test ! -e target-v3
cp -a --reflink=auto /root/volta-graph-diag-v2/target-v2 target-v3
cat >/root/volta-loop-prof-v3/nvcc-reuse-frozen <<'SH'
#!/bin/bash
set -euo pipefail
out=
while (($#)); do
 if test "$1" = -o; then shift; out=$1; break; fi
 shift
done
test -n "$out"
cp /root/volta/target/release/build/dsx-b2944b73405876f5/out/dsx.fatbin "$out"
SH
chmod +x /root/volta-loop-prof-v3/nvcc-reuse-frozen
source /root/.cargo/env
CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=/root/volta-loop-prof-v3/target-v3 NVCC=/root/volta-loop-prof-v3/nvcc-reuse-frozen CUDA_ARCHS=70 timeout 180 cargo build --offline --locked --release -p dsx --features cuda --bin volta-server >/root/build-loop-prof-v3.log 2>&1
sha256sum target-v3/release/volta-server >/root/loop-prof-v3-binary.sha256
find target-v3/release/build -path '*/out/dsx.fatbin' -exec sha256sum '{}' \; >/root/loop-prof-v3-fatbins.sha256
touch /root/LOOP_PROF_V3_BUILD_DONE
