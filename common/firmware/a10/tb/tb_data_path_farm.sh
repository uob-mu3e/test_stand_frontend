#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=500ns
#2us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd"  *.vhd ../*.vhd ../../util/*.vhd ../../registers/*.vhd ../../s4/ip_ram_1_port.vhd ../../s4/ip_ram.vhd ../../s4/ip_scfifo.vhd ../../s4/ip_dcfifo.vhd ../../s4/ip_dcfifo_mixed_widths.vhd
