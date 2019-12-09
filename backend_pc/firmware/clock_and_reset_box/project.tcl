#!/bin/sh
# \
unset CDPATH ; \
cd "$(dirname -- "$(readlink -e -- "$0")")" || exit 1 ; \
exec vivado -mode tcl -source "$0" -tclargs "$@"

set part "xc7k325tffg900-2"

create_project -in_memory -part $part

set_property -name "part" -value "xc7k325tffg900-2" -objects [ current_project ]
set_property -name "enable_vhdl_2008" -value "1" -objects [ current_project ]
set_property -name "target_language" -value "VHDL" -objects [ current_project ]

add_files src
set_property is_enabled false [ get_files "src/top_genesys2.vhd" ]
read_xdc "src/constrs_1/new/genesys_master.xdc"
set_property top mu3e_top [ current_fileset ]
set_property file_type {VHDL 2008} [ get_files -filter {FILE_TYPE == VHDL} ]

file mkdir ".cache"
source "ip/gtx_reset_firefly.tcl"
source "ip/ila_0.tcl"
source "ip/mac_fifo_axi4.tcl"
source "ip/temac_gbe_v9_0_rgmii.tcl"
