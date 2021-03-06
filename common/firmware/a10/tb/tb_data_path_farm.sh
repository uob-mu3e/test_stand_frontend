#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=8000ns

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" \
    *.vhd ../*.vhd ../../util/*.vhd ../../util/quartus/*.vhd \
    ../../util/altera/*.vhd ../../registers/*.vhd ../ddr/*.vhd \
    ../swb/*.vhd ../farm/*.vhd ../ddr/tb/*.vhd
