#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=4us

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" \
    *.vhd ../*.vhd ../../util/*.vhd ../../util/altera/*.vhd \
    ../../registers/*.vhd f0_sim.vhd f1_sim.vhd ../../../../fe_board/fe_mupix/mupix_block/sorter/mp_sorter_datagen.vhd
