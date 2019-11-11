#

# FE board leds are active low
set_location_assignment PIN_D23 -to led_n[0]
set_location_assignment PIN_D24 -to led_n[1]
set_location_assignment PIN_D25 -to led_n[2]
set_location_assignment PIN_D26 -to led_n[3]
set_location_assignment PIN_D27 -to led_n[4]
set_location_assignment PIN_E23 -to led_n[5]
set_location_assignment PIN_E24 -to led_n[6]
set_location_assignment PIN_F21 -to led_n[7]
set_location_assignment PIN_F22 -to led_n[8]
set_location_assignment PIN_F23 -to led_n[9]
set_location_assignment PIN_F25 -to led_n[10]
set_location_assignment PIN_G20 -to led_n[11]
set_location_assignment PIN_G22 -to led_n[12]
set_location_assignment PIN_G23 -to led_n[13]
set_location_assignment PIN_G24 -to led_n[14]
set_location_assignment PIN_G25 -to led_n[15]



# clk_aux SMA input
set_location_assignment PIN_B18 -to clk_aux
set_instance_assignment -name IO_STANDARD LVDS -to clk_aux



# Si5342 clocks
set_instance_assignment -name IO_STANDARD LVDS -to si42_clk_40
set_location_assignment PIN_AN15 -to si42_clk_40
set_instance_assignment -name IO_STANDARD LVDS -to si42_clk_80
set_location_assignment PIN_B15 -to si42_clk_80



# QSFP pins
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_tx[3]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_tx[2]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_tx[1]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_tx[0]
set_location_assignment PIN_U31 -to qsfp_tx[3]
set_location_assignment PIN_W31 -to qsfp_tx[2]
set_location_assignment PIN_AE31 -to qsfp_tx[1]
set_location_assignment PIN_AG31 -to qsfp_tx[0]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_rx[3]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_rx[2]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_rx[1]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to qsfp_rx[0]
set_location_assignment PIN_V33 -to qsfp_rx[3]
set_location_assignment PIN_Y33 -to qsfp_rx[2]
set_location_assignment PIN_AF33 -to qsfp_rx[1]
set_location_assignment PIN_AH33 -to qsfp_rx[0]

# QSFP clock from Si5345 chip
set_instance_assignment -name IO_STANDARD LVDS -to qsfp_pll_clk
set_location_assignment PIN_AB33 -to qsfp_pll_clk



# POD pins
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_tx[3]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_tx[2]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_tx[1]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_tx[0]
set_location_assignment PIN_U4 -to pod_tx[3]
set_location_assignment PIN_W4 -to pod_tx[2]
set_location_assignment PIN_AE4 -to pod_tx[1]
set_location_assignment PIN_AG4 -to pod_tx[0]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_rx[3]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_rx[2]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_rx[1]
set_instance_assignment -name IO_STANDARD "1.5-V PCML" -to pod_rx[0]
set_location_assignment PIN_V2 -to pod_rx[3]
set_location_assignment PIN_Y2 -to pod_rx[2]
set_location_assignment PIN_AF2 -to pod_rx[1]
set_location_assignment PIN_AH2 -to pod_rx[0]

set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to pod_tx_reset_n
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to pod_rx_reset_n
set_location_assignment PIN_AE18 -to pod_tx_reset_n
set_location_assignment PIN_AP18 -to pod_rx_reset_n

# POD clock from Si5345 chip
set_instance_assignment -name IO_STANDARD LVDS -to pod_pll_clk
set_location_assignment PIN_K2 -to pod_pll_clk
#set_instance_assignment -name IO_STANDARD LVDS -to pod_pll_clk_right
#set_location_assignment PIN_K33 -to pod_pll_clk_right



# "Altera Quartus Settings File Reference Manual - Advanced I/O Timing Assignments"
set_instance_assignment -name GXB_0PPM_CORE_CLOCK ON -from qsfp_pll_clk -to qsfp_tx[*]
set_instance_assignment -name GXB_0PPM_CORE_CLOCK ON -from qsfp_pll_clk -to qsfp_rx[*]
set_instance_assignment -name GXB_0PPM_CORE_CLOCK ON -from pod_pll_clk -to pod_tx[*]
set_instance_assignment -name GXB_0PPM_CORE_CLOCK ON -from pod_pll_clk -to pod_rx[*]



# ...
set_instance_assignment -name IO_STANDARD LVDS -to clk_625
set_location_assignment PIN_B17 -to clk_625
