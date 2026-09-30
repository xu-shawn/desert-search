#!/bin/bash
# Poll every box once a minute; exit and print the lines as soon as a verified desert >= MIN million blocks²
# (default 100) or a step-4 CANDIDATE >= MIN-1 appears. usage: watch.sh [MIN]
. "$(dirname "$0")/common.sh"
MIN=${1:-100}
while true; do
  for b in $(box_names); do
    hit=$("$DEPLOY/box.sh" "$b" "cd /root/desert-search/runs 2>/dev/null && { grep -h 'Blocks^2 Size' *.verified 2>/dev/null | awk -v m=$MIN '\$NF >= m * 1e6'; grep -h -o 'CANDIDATE step4=[0-9.]*M seed=-\?[0-9]* x=-\?[0-9]* z=-\?[0-9]*' *g[0-9].log 2>/dev/null | awk -F'[=M]' -v m=$MIN '\$2 >= m - 1'; }" 2>/dev/null)
    [ -n "$hit" ] && { echo "$(date -u +%FT%TZ) $b:"; echo "$hit"; exit 0; }
  done
  sleep 60
done
