#!/bin/bash
set -eux

TB=$1
shift

STOPTIME=${STOPTIME:-1us}

SRC=()
for arg in "$@" ; do
    arg=$(readlink -f "$arg")
    SRC+=("$arg")
done

mkdir -p .cache
cd .cache || exit 1

OPTS=()

ghdl -i "${OPTS[@]}" "${SRC[@]}"
ghdl -s "${OPTS[@]}" "${SRC[@]}"
ghdl -m "${OPTS[@]}" "$TB"
ghdl -e "${OPTS[@]}" "$TB"
ghdl -r "${OPTS[@]}" "$TB" --stop-time="$STOPTIME" --vcd="$TB.vcd" --wave="$TB.ghw"

gtkwave "$TB.ghw"
