#!/bin/sh
set -eu
IFS="$(printf '\n\t')"
unset CDPATH
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1

export STOPTIME=20ns

entity=$(basename "$0" .sh)

../../util/sim.sh "$entity" "$entity.vhd" \
    *.vhd ../*.vhd ../../util/*.vhd ../../util/altera/*.vhd \
    ../../registers/*.vhd
