#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=2us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" *.vhd ../data_flow.vhd ../dataflow_components.vhd ../../../../common/firmware/s4/ip_dcfifo.vhd ../../../../common/firmware/s4/ip_ram_1_port.vhd ../../../../common/firmware/s4/ip_dcfifo_mixed_widths.vhd ../../a10/data_generator_merged_data.vhd ../../util/linear_shift.vhd ../../util/counter.vhd ../../util/link_to_fifo.vhd ../../../../common/firmware/s4/ip_ram.vhd
