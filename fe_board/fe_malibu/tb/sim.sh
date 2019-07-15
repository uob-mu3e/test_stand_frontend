#!/bin/sh
set -eux

TB=$1
shift

STOPTIME=1100ns

#mkdir -p .cache
#cd .cache || exit 1

#DIR=..
SRC="$@"

ghdl -i $SRC
ghdl -s $SRC
ghdl -m "$TB"
ghdl -e "$TB"
ghdl -r "$TB" --stop-time="$STOPTIME" --vcd="$TB.vcd" --wave="$TB.ghw"

gtkwave "$TB.ghw"
