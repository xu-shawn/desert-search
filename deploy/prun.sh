#!/bin/bash
# On a box: one gpipe per GPU, threads split by the container's CPU quota. usage: prun.sh RUN_PREFIX START [THREADS_PER_GPU]
# GPU g searches seed indices from START + g * 1e12 (a pipeline walks ~5-20M indices/s; 1e12 lasts > 12 h).
cd "$(dirname "$0")/.."; RUN=$1; S=$2
NG=$(nvidia-smi -L | wc -l)
NC=$(nproc)                                                        # CPU quota: cgroup v2, else v1, else all cores
if [ -f /sys/fs/cgroup/cpu.max ] && [ "$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)" != max ]; then
  NC=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max)
elif [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)" -gt 0 ]; then
  NC=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))
fi
TPG=${3:-$(( (NC - NG) / NG ))}
echo "gpus $NG cores $NC threads/gpu $TPG"
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash deploy/gpipe.sh "${RUN}g$g" $((S + g * 1000000000000)) "$TPG" > /dev/null 2>&1 < /dev/null &
done
