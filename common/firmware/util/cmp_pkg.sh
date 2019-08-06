#!/bin/sh
set -eu

# find '*.cmp' files
# and make 'cmp' package

(
cat << EOF
library ieee;
use ieee.std_logic_1164.all;
package cmp is
EOF

find -L -name "*.cmp" -exec cat {} \;

cat << EOF
end package;
EOF
) > cmp_pkg.vhd
