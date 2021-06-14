#!/bin/bash

#
# should be used on generated `altera_pcie_express.sdc`
#
# `query_collection` returns string (space separated list of objects)
# which must not be treated as tcl list,
# otherwise list access procedures (`lindex`, `foreach`, etc.) may corrupt objects
# (e.g. by stripping `\` symbol)
#

sed -e 's/foreach clk $clk_prefix/foreach clk [split $clk_prefix " "]/' -i -- "$1"
sed -e 's/lindex \[query_collection $byte_ser_clk_pins\]/lindex [split [query_collection $byte_ser_clk_pins] " "]/' -i -- "$1"
sed -e 's/foreach clk $byte_ser_clk_pin0/foreach clk [split $byte_ser_clk_pin0 " "]/' -i -- "$1"
