#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=4us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" ../swb_data_path.vhd ../../assignments/mudaq_registers.vhd ../../assignments/dataflow_components.vhd ../../util/*.vhd 
