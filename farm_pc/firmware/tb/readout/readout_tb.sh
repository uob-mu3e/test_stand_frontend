#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=4us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" *.vhd ../../midas_event_builder.vhd ../../midas_bank_builder.vhd ../../../../common/firmware/s4/ip_ram.vhd ../../../../common/firmware/s4/ip_dcfifo.vhd ../../../../common/firmware/s4/ip_scfifo.vhd ../../../fe_board/firmware/FEB_common/daq_constants.vhd
