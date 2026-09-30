#!/bin/bash
# Rent vast.ai offers (IDs from offers.py) as ${FLEET_PREFIX:-desert-}NAME, wait for all of them to boot, and
# register each in deploy/boxes.txt as soon as its SSH port is up.
# usage: rent.sh OFFER_ID NAME [OFFER_ID NAME ...]
. "$(dirname "$0")/common.sh"
IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
declare -A IID
while [ $# -ge 2 ]; do
  id=$1; name=$2; shift 2
  out=$(vastai create instance "$id" --image "$IMAGE" --disk 30 --label "$PREFIX_LABEL$name" --ssh --direct --raw) || { echo "$name: create failed"; continue; }
  IID[$name]=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['new_contract'])" "$out")
  echo "$name: instance ${IID[$name]}"
done
touch "$BOXES"
for _ in $(seq 90); do                                   # up to 30 minutes
  [ ${#IID[@]} -eq 0 ] && break
  for name in "${!IID[@]}"; do
    read -r st host port <<< "$(vastai show instance "${IID[$name]}" --raw | python3 -c "
import json,sys; x=json.load(sys.stdin); p=(x.get('ports') or {}).get('22/tcp') or [{}]
print(x.get('actual_status'), x.get('public_ipaddr'), p[0].get('HostPort', '-'))")"
    if [ "$st" = running ] && [ "$port" != - ]; then
      sed -i "/^$name /d" "$BOXES"; echo "$name $host $port" >> "$BOXES"; echo "$name: $host:$port registered"
      unset "IID[$name]"
    fi
  done
  [ ${#IID[@]} -gt 0 ] && sleep 20
done
for name in "${!IID[@]}"; do echo "$name: not running after 30 min; destroy it with: vastai destroy instance ${IID[$name]}"; done
