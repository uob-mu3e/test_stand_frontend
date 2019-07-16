#!/bin/sh

unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

./sim.sh tb_sc_s4 tb_sc_s4.vhd ../sc_s4.vhd
