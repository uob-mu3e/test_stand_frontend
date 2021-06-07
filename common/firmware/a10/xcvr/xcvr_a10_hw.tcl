package require qsys


# 
# module xcvr_a10
# 
set_module_property NAME xcvr_a10
set_module_property DISPLAY_NAME xcvr_a10
set_module_property GROUP {mu3e}

set_module_property VERSION 1.0
set_module_property DESCRIPTION ""
set_module_property AUTHOR "Alexandr Kozlinskiy"


# 
# file sets
# 
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL xcvr_a10
add_fileset_file xcvr_a10.vhd VHDL PATH xcvr_a10.vhd TOP_LEVEL_FILE



# 
# connection point clk
# 
add_interface clk clock end
set_interface_property clk clockRate 0

add_interface_port clk clk clk Input 1


# 
# connection point reset
# 
add_interface reset reset end
set_interface_property reset associatedClock clk
set_interface_property reset synchronousEdges DEASSERT

add_interface_port reset reset reset Input 1



foreach { name addr_width } {
    avs 14
} {
    add_interface $name avalon end
    set_interface_property $name addressUnits WORDS
    set_interface_property $name associatedClock clk
    set_interface_property $name associatedReset reset
    set_interface_property $name bitsPerSymbol 8

    add_interface_port $name ${name}_address address Input $addr_width
    add_interface_port $name ${name}_read read Input 1
    add_interface_port $name ${name}_readdata readdata Output 32
    add_interface_port $name ${name}_write write Input 1
    add_interface_port $name ${name}_writedata writedata Input 32
    add_interface_port $name ${name}_waitrequest waitrequest Output 1
}



# 
# connection point cdr_refclk
# 
add_interface cdr_refclk clock end
set_interface_property cdr_refclk clockRate 0

add_interface_port cdr_refclk cdr_refclk clk Input 1


# 
# connection point pll_refclk
# 
add_interface pll_refclk clock end
set_interface_property pll_refclk clockRate 0

add_interface_port pll_refclk pll_refclk clk Input 1


# 
# connection point xcvr_tx0/tx1/tx2/tx3
# 
foreach { tx } { tx0 tx1 tx2 tx3 } {
    add_interface xcvr_${tx} conduit end
    set_interface_property xcvr_${tx} associatedClock ""
    set_interface_property xcvr_${tx} associatedReset ""

    add_interface_port xcvr_${tx} ${tx}_data data Input 32
    add_interface_port xcvr_${tx} ${tx}_datak datak Input 4
}


# 
# connection point xcvr_rx0/rx1/rx2/rx3
# 
foreach { rx } { rx0 rx1 rx2 rx3 } {
    add_interface xcvr_${rx} conduit end
    set_interface_property xcvr_${rx} associatedClock ""
    set_interface_property xcvr_${rx} associatedReset ""

    add_interface_port xcvr_${rx} ${rx}_data data Output 32
    add_interface_port xcvr_${rx} ${rx}_datak datak Output 4
}


# 
# connection point qsfp
# 
add_interface qsfp conduit end
set_interface_property qsfp associatedClock ""
set_interface_property qsfp associatedReset ""

add_interface_port qsfp tx_p tx_p Output 4
add_interface_port qsfp rx_p rx_p Input 4


# 
# 
# 
add_interface tx_clkout conduit end
add_interface_port tx_clkout tx_clkout clk Output 4
add_interface tx_clkin conduit end
add_interface_port tx_clkin tx_clkin clk Input 4
add_interface rx_clkout conduit end
add_interface_port rx_clkout rx_clkout clk Output 4
add_interface rx_clkin conduit end
add_interface_port rx_clkin rx_clkin clk Input 4
