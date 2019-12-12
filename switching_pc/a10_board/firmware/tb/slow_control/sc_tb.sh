#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=10us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" *.vhd ../../sc_slave.vhd ../../sc_master.vhd ../../../../../common/firmware/s4/ip_ram.vhd
