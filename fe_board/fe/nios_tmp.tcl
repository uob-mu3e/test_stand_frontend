#

add_instance temp altera_temp_sense
add_connection clk.clk temp.clk
set_instance_parameter_value temp {CLK_FREQUENCY} 50
set_instance_parameter_value temp {CLOCK_DIVIDER_VALUE} 80
set_interface_property temp EXPORTOF temp.tsdcalo
