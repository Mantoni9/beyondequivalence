#!/bin/bash
# Diagnose: how many results/subB_*g7-literature* dirs exist for sbert/sym
# and qwen3/sym, and what wandb_group they belong to. Used to figure out
# why analyze_sweep reported 600 / 630 with these two cells empty.

set -u

GROUP="${SUBB_WANDB_GROUP:-}"
echo "Current SUBB_WANDB_GROUP: $GROUP"
echo

extract_group() {
    # $1 = path to config.json. Print the wandb_group field, or <no-group>.
    grep -o '"wandb_group"[[:space:]]*:[[:space:]]*"[^"]*"' "$1" 2>/dev/null \
        | head -1 \
        | sed -E 's/.*"wandb_group"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/'
}

count_in_group() {
    local pattern="$1"
    local match=0 total=0
    for d in $pattern; do
        [ -d "$d" ] || continue
        total=$((total + 1))
        local cfg="$d/config.json"
        [ -f "$cfg" ] || continue
        local g
        g=$(extract_group "$cfg")
        if [ "$g" = "$GROUP" ]; then
            match=$((match + 1))
        fi
    done
    echo "$match / $total"
}

echo "sbert sym g7-lit (in current group / total dirs):"
count_in_group "results/subB_sbert_sym_*g7-literature*"

echo "qwen3 sym g7-lit (in current group / total dirs):"
count_in_group "results/subB_qwen3-embedding-8b_sym_*g7-literature*"

echo
echo "Distinct wandb_groups present in those dirs (count + group):"
{
    for d in results/subB_sbert_sym_*g7-literature* results/subB_qwen3-embedding-8b_sym_*g7-literature*; do
        [ -d "$d" ] || continue
        cfg="$d/config.json"
        [ -f "$cfg" ] || { echo "<no-config>"; continue; }
        g=$(extract_group "$cfg")
        echo "${g:-<empty>}"
    done
} | sort | uniq -c | sort -rn
