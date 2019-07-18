#!/bin/bash
set -eux

TB=$1
shift

STOPTIME=1100ns

SRC=()
for arg in "$@" ; do
    arg=$(readlink -f "$arg")
    SRC+=("$arg")
done

mkdir -p .cache
cd .cache || exit 1

ghdl -i "${SRC[@]}"
ghdl -s "${SRC[@]}"
ghdl -m "$TB"
ghdl -e "$TB"
ghdl -r "$TB" --stop-time="$STOPTIME" --vcd="$TB.vcd" --wave="$TB.ghw"

gtkwave "$TB.ghw"
