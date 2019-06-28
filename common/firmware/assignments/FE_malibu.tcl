# base frontend-to-malibu assignments

set_location_assignment PIN_AA26 -to malibu_ck_fpga_0
set_instance_assignment -name IO_STANDARD LVDS -to malibu_ck_fpga_0
set_location_assignment PIN_M26 -to malibu_ck_fpga_1
set_instance_assignment -name IO_STANDARD LVDS -to malibu_ck_fpga_1

set_location_assignment PIN_N25 -to malibu_pll_test
set_instance_assignment -name IO_STANDARD LVDS -to malibu_pll_test
set_location_assignment PIN_W24 -to malibu_pll_reset
set_instance_assignment -name IO_STANDARD LVDS -to malibu_pll_reset

set_location_assignment PIN_AP10 -to malibu_chip_reset
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_chip_reset

set_location_assignment PIN_AP29 -to malibu_i2c_scl
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_i2c_scl
set_instance_assignment -name WEAK_PULL_UP_RESISTOR ON -to malibu_i2c_scl
set_location_assignment PIN_AP30 -to malibu_i2c_sda
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_i2c_sda
set_instance_assignment -name WEAK_PULL_UP_RESISTOR ON -to malibu_i2c_sda
set_location_assignment PIN_AP31 -to malibu_i2c_int_n
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_i2c_int_n

set_location_assignment PIN_AH25 -to malibu_spi_sdi
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_spi_sdi
set_location_assignment PIN_AL27 -to malibu_spi_sdo
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_spi_sdo
set_location_assignment PIN_AP25 -to malibu_spi_sck
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_spi_sck
set_location_assignment PIN_AL22 -to malibu_spi_sdo_cec
set_instance_assignment -name IO_STANDARD "1.8 V" -to malibu_spi_sdo_cec

set_location_assignment PIN_AE28 -to malibu_data[0]
set_location_assignment PIN_AF28 -to malibu_data[1]
set_location_assignment PIN_L26 -to malibu_data[2]
set_location_assignment PIN_AG29 -to malibu_data[3]
set_location_assignment PIN_AB27 -to malibu_data[4]
set_location_assignment PIN_AD26 -to malibu_data[5]
set_location_assignment PIN_AC26 -to malibu_data[6]
set_location_assignment PIN_AH28 -to malibu_data[7]
set_location_assignment PIN_J28 -to malibu_data[8]
set_location_assignment PIN_H29 -to malibu_data[9]
set_location_assignment PIN_L29 -to malibu_data[10]
set_location_assignment PIN_K28 -to malibu_data[11]
set_location_assignment PIN_L30 -to malibu_data[12]
set_location_assignment PIN_G28 -to malibu_data[13]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[0]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[1]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[2]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[3]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[4]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[5]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[6]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[7]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[8]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[9]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[10]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[11]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[12]
set_instance_assignment -name IO_STANDARD LVDS -to malibu_data[13]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[0]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[1]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[2]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[3]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[4]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[5]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[6]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[7]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[8]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[9]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[10]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[11]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[12]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[13]
