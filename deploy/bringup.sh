#!/bin/bash
# Upload the repo to a rented box, build the GPU searcher there, and start one pipeline per GPU.
# usage: bringup.sh NAME RUN_PREFIX START      (START: seed-index base; give each box its own, e.g. 1e13 apart)
. "$(dirname "$0")/common.sh"
N=$1; RUN=$2; S=$3
[ -z "$S" ] && { echo "usage: bringup.sh NAME RUN_PREFIX START" >&2; exit 2; }
tmp=$(mktemp --suffix=.tgz)
(cd "$REPO" && { git ls-files --cached --others --exclude-standard | grep -v "^MC-Measure$"; git -C MC-Measure ls-files | sed 's|^|MC-Measure/|'; } | grep -v '^results/\|^docs/' | tar czf "$tmp" -T -)
"$DEPLOY/box.sh" "$N" 'mkdir -p /root/desert-search' && "$DEPLOY/box.sh" "$N" put "$tmp" /root/src.tgz || exit 1
rm -f "$tmp"
"$DEPLOY/box.sh" "$N" "cd /root/desert-search && tar xzf ../src.tgz && (command -v make >/dev/null || (apt-get update -qq && apt-get install -y -qq make g++ >/dev/null)) \
  && make -j\$(nproc) gpu 2>&1 | grep -E 'error|Error' ; ls bin/dsearch_desert_gpu bin/mcmeasure >/dev/null && nvidia-smi -L && bash deploy/prun.sh $RUN $S"
