#!/bin/bash
# Run a command on a fleet box, or copy files. usage: box.sh NAME CMD... | box.sh NAME put SRC DST | box.sh NAME get SRC [DST]
. "$(dirname "$0")/common.sh"
read -r H P <<< "$(awk -v n="$1" '$1 == n {print $2, $3}' "$BOXES" 2>/dev/null)"; shift
[ -z "$H" ] && { echo "unknown box (see $BOXES)" >&2; exit 2; }
case "$1" in
  put) scp "${SSHOPTS[@]}" -P "$P" "$2" "root@$H:$3" ;;
  get) scp "${SSHOPTS[@]}" -P "$P" "root@$H:$2" "${3:-.}" ;;
  *)   ssh "${SSHOPTS[@]}" -p "$P" "root@$H" "$@" ;;
esac
