#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

./util/sim.sh tb_sc_rx \
    tb_sc_rx.vhd sc_rx.vhd \
    s4/ip_scfifo.vhd
