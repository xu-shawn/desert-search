#!/bin/bash
# Stop (not destroy) every ${FLEET_PREFIX:-desert-}* instance if vast credit falls below MIN dollars (default 8),
# so other instances on the account keep running. usage: credit_watchdog.sh [MIN] &
. "$(dirname "$0")/common.sh"
MIN=${1:-8}
while true; do
  c=$(vastai show user --raw | python3 -c "import json,sys; print(json.load(sys.stdin).get('credit', 1e9))")
  if python3 -c "import sys; sys.exit(0 if float('$c') < $MIN else 1)"; then
    echo "$(date -u +%FT%TZ) credit \$$c < \$$MIN: stopping $PREFIX_LABEL* instances"
    vastai show instances --raw | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if (x.get('label') or '').startswith('$PREFIX_LABEL'): print(x['id'])" | while read -r id; do vastai stop instance "$id"; done
    exit 0
  fi
  sleep 300
done
