#!/bin/bash
# Poll every box once a minute; exit and print the lines as soon as a verified desert >= MIN million blocks²
# (default 100) appears, or a step-4 CANDIDATE >= MIN-1 that has not been verified yet (step 4 reads up to
# ~5% under exact, so such a candidate may be a hit; once verified below MIN it no longer triggers).
# usage: watch.sh [MIN]
. "$(dirname "$0")/common.sh"
MIN=${1:-100}
REMOTE=$(cat <<'EOF'
cd /root/desert-search/runs 2>/dev/null || exit 0
grep -h 'Blocks^2 Size' *.verified 2>/dev/null | awk -v m=@MIN@ '$NF >= m * 1e6'
grep -h -o 'CANDIDATE step4=[0-9.]*M seed=-\?[0-9]* x=-\?[0-9]* z=-\?[0-9]*' *g[0-9].log 2>/dev/null |
  sed -E 's/CANDIDATE step4=([0-9.]+)M seed=(-?[0-9]+)(.*)/\1 \2\3/' | awk -v m=@MIN@ '$1 >= m - 1' | sort -u |
  while read -r v s rest; do grep -q -- "seed=$s " *.verified 2>/dev/null || echo "unverified CANDIDATE step4=${v}M seed=$s$rest"; done
EOF
)
REMOTE=${REMOTE//@MIN@/$MIN}
while true; do
  for b in $(box_names); do
    hit=$("$DEPLOY/box.sh" "$b" "$REMOTE" 2>/dev/null)
    [ -n "$hit" ] && { echo "$(date -u +%FT%TZ) $b:"; echo "$hit"; exit 0; }
  done
  sleep 60
done
