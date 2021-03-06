#!/bin/sh
set -euf

# find '*.cmp' files
# and generate 'components' package

if [ $# -ge 1 ] ; then
    cd -- "$1" || exit 1
fi

cat << EOF
library ieee;
use ieee.std_logic_1164.all;

package cmp is

    constant GIT_HEAD : std_logic_vector(16*4-1 downto 0) := X"$(git rev-parse --short=16 HEAD | rev)";

EOF

find -L . -name '*.cmp' -exec echo "--" '{}' ';' -exec cat '{}' ';'

cat << EOF

end package;
EOF
