#!/bin/sh

unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

util/sim.sh tb_sc_rx \
    tb_sc_rx.vhd sc_rx.vhd \
    util/scfifo.vhd ip_scfifo.vhd \
    util/util_pkg.vhd
