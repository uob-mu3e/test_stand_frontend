#!/bin/sh
set -eu

(
cat << EOF
library ieee;
use ieee.std_logic_1164.all;
package cmp is
EOF

find -name "*.cmp" -exec cat {} \;

cat << EOF
end package;
EOF
) > cmp_pkg.vhd
