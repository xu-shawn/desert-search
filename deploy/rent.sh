#!/bin/bash
# Rent vast.ai offers (IDs from offers.py) as ${FLEET_PREFIX:-desert-}NAME and register them in boxes.txt once SSH is up.
# usage: rent.sh OFFER_ID NAME [OFFER_ID NAME ...]
. "$(dirname "$0")/common.sh"
IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
while [ $# -ge 2 ]; do
  id=$1; name=$2; shift 2
  out=$(vastai create instance "$id" --image "$IMAGE" --disk 30 --label "$PREFIX_LABEL$name" --ssh --direct --raw) || { echo "$name: create failed"; continue; }
  iid=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['new_contract'])" "$out")
  echo "$name: instance $iid, waiting for it to boot"
  for _ in $(seq 90); do
    read -r st host port <<< "$(vastai show instance "$iid" --raw | python3 -c "
import json,sys; x=json.load(sys.stdin); p=(x.get('ports') or {}).get('22/tcp') or [{}]
print(x.get('actual_status'), x.get('public_ipaddr'), p[0].get('HostPort', '-'))")"
    [ "$st" = running ] && [ "$port" != - ] && break; sleep 20
  done
  [ "$st" = running ] || { echo "$name: not running after 30 min (status $st); destroy it with: vastai destroy instance $iid"; continue; }
  touch "$BOXES"; sed -i "/^$name /d" "$BOXES"; echo "$name $host $port" >> "$BOXES"; echo "$name: $host:$port registered"
done
