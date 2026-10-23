#!/usr/bin/env bash
# Hydrate a fresh 8x V100 box for volta-server (copy the repo to /root/volta first): CUDA 12.9 (the newest toolkit that targets sm_70), Rust, the
# Hugging Face build (pack in /dev/shm, base on disk), then the release build. Log: /root/hydrate.log
set -uo pipefail
exec > /root/hydrate.log 2>&1
t0=$(date +%s); ts() { echo "[$(( $(date +%s) - t0 ))s] $*"; }
ts start
python3 -m pip install -q -U --break-system-packages "huggingface_hub[hf_xet]" 2>&1 | tail -2 || python3 -m pip install -q -U "huggingface_hub[hf_xet]" 2>&1 | tail -2
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1
( hf download lxe/DeepSeek-V4.1-Flash-1017-Volta --include 'pack-27.5gb-8xv100/*' --local-dir /dev/shm/vw --max-workers 32 > /root/dl_pack.log 2>&1; echo "pack rc=$?" >> /root/dl_pack.log ) &
( hf download lxe/DeepSeek-V4.1-Flash-1017-Volta --include 'base/*' --local-dir /root/vw --max-workers 32 > /root/dl_base.log 2>&1; echo "base rc=$?" >> /root/dl_base.log ) &
ts "downloads started"
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-9 > /root/apt.log 2>&1; ts "cuda-toolkit-12-9 rc=$?"
ls /usr/local/cuda-12.9/bin/nvcc && /usr/local/cuda-12.9/bin/nvcc --version | tail -1
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal > /root/rustup.log 2>&1; ts "rustup rc=$?"
source ~/.cargo/env
# build.rs prefers /usr/local/cuda-13, which no longer targets sm_70: point NVCC at 12.9
cd /root/volta && NVCC=/usr/local/cuda-12.9/bin/nvcc CUDA_ARCHS=70 cargo build --release -p dsx --features cuda --bin volta-server --bin volta 2>&1 | tail -3
ts "build rc=${PIPESTATUS[0]}"
wait
ts "downloads done"; tail -2 /root/dl_pack.log /root/dl_base.log
du -sh /dev/shm/vw /root/vw
ts DONE
