#!/usr/bin/env bash
# volta-server for agents: one session per GPU (SLOTS, CTX, CHUNK, SPECK below), sampling on the GPU,
# API on 127.0.0.1:8090 and the dashboard / stats on 127.0.0.1:8091 (no key: loopback only, reached by ssh tunnel)
cd /root/volta
pkill -f '^./target/release/volta-server' 2>/dev/null
VOLTA_LOOP_PROF=${LOOP_PROF:-0} VOLTA_PACK=/dev/shm/vw/pack-27.5gb-8xv100 nohup ./target/release/volta-server \
  --model /root/vw/base --slots ${SLOTS:-1} --ctx ${CTX:-262144} --chunk ${CHUNK:-32} --spec-k ${SPECK:-5} --host 127.0.0.1 --port 8090 > /root/server.log 2>&1 &
echo "started pid $!"
