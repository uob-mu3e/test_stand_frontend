#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=4us

./util/sim.sh tb_sc_ram \
    tb_sc_ram.vhd sc_ram.vhd \
    util/util_pkg.vhd
