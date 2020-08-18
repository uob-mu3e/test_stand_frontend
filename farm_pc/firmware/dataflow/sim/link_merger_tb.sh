#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=2us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" *.vhd ../link_merger.vhd ../dataflow_components.vhd ../../../../common/firmware/s4/ip_dcfifo.vhd   ../../a10/data_generator_a10.vhd ../../a10/linear_shift.vhd ../../util/link_to_fifo.vhd ../../a10/time_merger.vhd ../../../../common/firmware/s4/ip_scfifo.vhd
