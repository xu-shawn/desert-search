#!/bin/bash
# One GPU: the searcher with the in-process GPU gate, plus an MC-Measure verification of every CANDIDATE
# appended to runs/RUN.verified. usage: gpipe.sh RUN START THREADS [extra key=value ...]
cd "$(dirname "$0")/.."; mkdir -p runs; RUN=$1; S=$2; T=$3; shift 3
./bin/dsearch_desert_gpu "$S" 0 "runs/$RUN" "$T" gate=1 gpu=1 gscan=1 "$@" 2> "runs/$RUN.log" &
sleep 2
tail -n +1 -F "runs/$RUN.log" | grep --line-buffered CANDIDATE | while read -r line; do
  seed=$(echo "$line" | sed -E 's/.*seed=(-?[0-9]+).*/\1/'); x=$(echo "$line" | sed -E 's/.* x=(-?[0-9]+).*/\1/'); z=$(echo "$line" | sed -E 's/.* z=(-?[0-9]+).*/\1/')
  res=$(./bin/mcmeasure "$seed" "$x" "$z" 2>&1 | tr '\n' ' ')
  echo "$(date -u +%FT%TZ) $RUN $line => $res" >> "runs/$RUN.verified"   # one atomic line per candidate
done
