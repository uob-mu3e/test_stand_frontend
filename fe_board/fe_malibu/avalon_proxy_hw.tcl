package require qsys

set name "avalon_proxy"

set_module_property NAME $name
set_module_property GROUP {mu3e}

set_module_property VERSION 1.0
set_module_property DESCRIPTION ""
set_module_property AUTHOR akozlins

#
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL $name
add_fileset_file $name.vhd VHDL PATH $name.vhd TOP_LEVEL_FILE

#
add_parameter addr_width INTEGER 8

set_module_property ELABORATION_CALLBACK elaborate
proc elaborate {} {
    set addr_width [get_parameter_value addr_width]

    # clock
    add_interface clk clock end
    set_interface_property clk clockRate 0
    add_interface_port clk clk clk Input 1
    # reset
    add_interface reset reset end
    set_interface_property reset associatedClock clk
    add_interface_port reset reset reset Input 1

    # avalon slave
    set name {slave}
    add_interface $name avalon slave
    set_interface_property $name associatedClock clk
    set_interface_property $name associatedReset reset
    set_interface_property $name addressUnits WORDS

    set prefix {avs}
    add_interface_port $name ${prefix}_address address Input $addr_width
    add_interface_port $name ${prefix}_read read Input 1
    add_interface_port $name ${prefix}_readdata readdata Output 32
    add_interface_port $name ${prefix}_write write Input 1
    add_interface_port $name ${prefix}_writedata writedata Input 32
    add_interface_port $name ${prefix}_waitrequest waitrequest Output 1

    # conduit master
    set name {master}
    add_interface $name avalon master
    set_interface_property $name associatedClock clk
    set_interface_property $name associatedReset reset
    set_interface_property $name addressUnits WORDS

    set prefix {avm}
    add_interface_port $name ${prefix}_address address Output $addr_width
    add_interface_port $name ${prefix}_read read Output 1
    add_interface_port $name ${prefix}_readdata readdata Input 32
    add_interface_port $name ${prefix}_write write Output 1
    add_interface_port $name ${prefix}_writedata writedata Output 32
    add_interface_port $name ${prefix}_waitrequest waitrequest Input 1
}
