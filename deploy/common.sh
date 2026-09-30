# Shared settings for the deploy scripts (sourced).
DEPLOY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DEPLOY")"
PREFIX_LABEL="${FLEET_PREFIX:-desert-}"                 # only instances whose label starts with this are touched
KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_ed25519}"             # private key registered with vast.ai
BOXES="$DEPLOY/boxes.txt"                                # name host port (written by rent.sh)
SSHOPTS=(-o IdentitiesOnly=yes -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes -o LogLevel=ERROR)
box_names() { [ -f "$BOXES" ] && awk '!/^#/ && NF >= 3 {print $1}' "$BOXES"; }
