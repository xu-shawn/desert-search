#!/bin/bash
# Pull a compact archive from every box into runs/fleet/NAME.tgz: verified finds, logs, run JSON (written when
# a run exits), the CSV rows that got exact floods, and per-run seed counts (enough for scripts/forecast.py).
# usage: collect.sh        Safe to run while pipelines are still searching.
. "$(dirname "$0")/common.sh"
mkdir -p "$REPO/runs/fleet"
REMOTE='cd /root/desert-search/runs || exit 1
for f in $(ls *.csv 2>/dev/null | grep -v seeds); do awk -F, "NR==1 || \$14>0" "$f"; done | gzip > exact_rows.csv.gz
for f in $(ls *.seeds.csv 2>/dev/null); do echo "$f $(($(wc -l < "$f") - 1))"; done > seed_counts.txt
tar czf /root/compact.tgz $(ls *.verified *.log *.json 2>/dev/null) exact_rows.csv.gz seed_counts.txt'
ok=0
for b in $(box_names); do
  if "$DEPLOY/box.sh" "$b" "$REMOTE" && "$DEPLOY/box.sh" "$b" get /root/compact.tgz "$REPO/runs/fleet/$b.tgz"; then
    echo "$b: $(du -h "$REPO/runs/fleet/$b.tgz" | cut -f1)"
  else echo "$b: COLLECT FAILED" >&2; ok=1; fi
done
exit $ok
