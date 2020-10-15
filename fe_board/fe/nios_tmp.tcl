#

add_instance temp altera_temp_sense
add_connection clk.clk temp.clk
set_instance_parameter_value temp {CLK_FREQUENCY} 50
set_instance_parameter_value temp {CLOCK_DIVIDER_VALUE} 80
set_instance_parameter_value temp {CE_CHECK} 1
set_instance_parameter_value temp {CLR_CHECK} 1

set_interface_property temp_clr EXPORTOF temp.clr
set_interface_property temp_ce EXPORTOF temp.ce
set_interface_property temp_done EXPORTOF temp.tsdcaldone
set_interface_property temp EXPORTOF temp.tsdcalo
