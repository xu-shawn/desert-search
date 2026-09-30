#!/bin/bash
# Destroy every vast instance whose label starts with ${FLEET_PREFIX:-desert-} (run collect.sh first).
# usage: teardown.sh [--yes]      (without --yes, only lists what would be destroyed)
. "$(dirname "$0")/common.sh"
ids=$(vastai show instances --raw | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if (x.get('label') or '').startswith('$PREFIX_LABEL'): print(x['id'], x['label'])")
echo "$ids"
[ "$1" = --yes ] || { echo "(dry run; pass --yes to destroy these)"; exit 0; }
echo "$ids" | while read -r id label; do [ -n "$id" ] && vastai destroy instance "$id" -y; done
: > "$BOXES"
