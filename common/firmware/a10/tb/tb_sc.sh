#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=10us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" *.vhd ../../a10/sc_secondary.vhd ../../a10/sc_main.vhd ../../../../../common/firmware/s4/ip_ram.vhd ../../../../../fe_board/fe/sc_rx.vhd
