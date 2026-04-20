#!/usr/bin/env bash
# sync_clusters.sh — Multi-Cluster Sync for OLALA
#
# Usage: ./sync_clusters.sh [bwuni|dws|all] [--setup|--sync|--job <script>]
#
# --setup  Install conda env and create directories on the cluster
# --sync   git pull on the cluster
# --job    sbatch <script> on the cluster

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------
BWUNI_USER="ma_amarkic"
BWUNI_HOST="uc3.scc.kit.edu"
BWUNI_WORKDIR="/pfs/work9/workspace/scratch/ma_amarkic-olala_ws"

DWS_USER="amarkic"
DWS_HOST_PRIMARY="dws-login-01.informatik.uni-mannheim.de"
DWS_HOST_FALLBACK="dws-login-02.informatik.uni-mannheim.de"
DWS_WORKDIR="/work/amarkic/beyondequivalence"

SSH_TIMEOUT=5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [bwuni|dws|all] [--setup|--sync|--job <script>]"
    exit 1
}

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[ERROR] $*" >&2; }

# Test SSH connectivity (timeout seconds, host)
ssh_ok() {
    local host="$1"
    local user="$2"
    ssh -o ConnectTimeout="$SSH_TIMEOUT" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "${user}@${host}" true 2>/dev/null
}

# Resolve DWS login node with automatic fallback
resolve_dws_host() {
    if ssh_ok "$DWS_HOST_PRIMARY" "$DWS_USER"; then
        log "DWS: using primary login node $DWS_HOST_PRIMARY" >&2
        echo "$DWS_HOST_PRIMARY"
    elif ssh_ok "$DWS_HOST_FALLBACK" "$DWS_USER"; then
        log "DWS: primary unreachable, falling back to $DWS_HOST_FALLBACK" >&2
        echo "$DWS_HOST_FALLBACK"
    else
        err "DWS: both login nodes unreachable ($DWS_HOST_PRIMARY, $DWS_HOST_FALLBACK)"
        return 1
    fi
}

# Run a command on a remote host via SSH
remote_run() {
    local user="$1"; local host="$2"; shift 2
    ssh -o ConnectTimeout="$SSH_TIMEOUT" \
        -o StrictHostKeyChecking=accept-new \
        "${user}@${host}" "$@"
}

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
do_setup() {
    local user="$1" host="$2" workdir="$3"
    log "[$host] Setting up environment in $workdir ..."
    remote_run "$user" "$host" bash -lc "
        set -euo pipefail
        mkdir -p '$workdir'
        cd '$workdir'
        mkdir -p models results logs
        if conda env list | grep -q 'melt-olala'; then
            echo 'Conda env melt-olala already exists, updating ...'
            conda env update -n melt-olala -f environment.yml --prune
        else
            echo 'Creating conda env melt-olala ...'
            conda env create -f environment.yml
        fi
        echo 'Setup complete.'
    "
}

do_sync() {
    local user="$1" host="$2" workdir="$3"
    log "[$host] Syncing (git pull) in $workdir ..."
    remote_run "$user" "$host" bash -lc "
        set -euo pipefail
        cd '$workdir'
        git pull
        echo 'Sync complete.'
    "
}

do_job() {
    local user="$1" host="$2" workdir="$3" script="$4"
    log "[$host] Submitting job: sbatch $script in $workdir ..."
    remote_run "$user" "$host" bash -lc "
        set -euo pipefail
        cd '$workdir'
        sbatch '$script'
    "
}

# ---------------------------------------------------------------------------
# Per-cluster dispatch
# ---------------------------------------------------------------------------
run_bwuni() {
    local action="$1"; shift
    if ! ssh_ok "$BWUNI_HOST" "$BWUNI_USER"; then
        err "bwUniCluster: cannot reach $BWUNI_HOST — check VPN / SSH key"
        return 1
    fi
    case "$action" in
        --setup) do_setup "$BWUNI_USER" "$BWUNI_HOST" "$BWUNI_WORKDIR" ;;
        --sync)  do_sync  "$BWUNI_USER" "$BWUNI_HOST" "$BWUNI_WORKDIR" ;;
        --job)   do_job   "$BWUNI_USER" "$BWUNI_HOST" "$BWUNI_WORKDIR" "$1" ;;
        *)       usage ;;
    esac
}

run_dws() {
    local action="$1"; shift
    local dws_host
    dws_host="$(resolve_dws_host)" || return 1
    case "$action" in
        --setup) do_setup "$DWS_USER" "$dws_host" "$DWS_WORKDIR" ;;
        --sync)  do_sync  "$DWS_USER" "$dws_host" "$DWS_WORKDIR" ;;
        --job)   do_job   "$DWS_USER" "$dws_host" "$DWS_WORKDIR" "$1" ;;
        *)       usage ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
[[ $# -lt 2 ]] && usage

TARGET="$1"; shift
ACTION="$1"; shift
JOB_SCRIPT="${1:-}"

case "$TARGET" in
    bwuni)
        run_bwuni "$ACTION" "$JOB_SCRIPT"
        ;;
    dws)
        run_dws "$ACTION" "$JOB_SCRIPT"
        ;;
    all)
        run_bwuni "$ACTION" "$JOB_SCRIPT"
        run_dws   "$ACTION" "$JOB_SCRIPT"
        ;;
    *)
        usage
        ;;
esac
