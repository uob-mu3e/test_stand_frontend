#
set_location_assignment PIN_B10  -to tile_spi_scl
set_location_assignment PIN_F17  -to tile_spi_mosi
set_location_assignment PIN_D8   -to tile_spi_miso
set_location_assignment PIN_E12  -to tile_chip_reset
set_location_assignment PIN_A14  -to tile_din[13]
set_location_assignment PIN_G15  -to tile_din[12]
set_location_assignment PIN_B12  -to tile_cec
set_location_assignment PIN_A15  -to tile_din[11]
set_location_assignment PIN_K11  -to tile_din[1]
set_location_assignment PIN_C14  -to tile_din[10]
set_location_assignment PIN_D13  -to tile_din[2]
set_location_assignment PIN_J12  -to tile_din[8]
set_location_assignment PIN_J14  -to tile_i2c_sda
set_location_assignment PIN_D9   -to tile_din[9]
set_location_assignment PIN_AC9  -to tile_pll_test
set_location_assignment PIN_AC16 -to tile_pll_reset
set_location_assignment PIN_AB15 -to tile_i2c_scl
set_location_assignment PIN_AA6  -to tile_din[3]
set_location_assignment PIN_AH4  -to tile_din[4]
set_location_assignment PIN_AJ16 -to tile_din[5]
set_location_assignment PIN_AJ3  -to tile_din[7]
set_location_assignment PIN_AJ1  -to tile_din[6]
set_location_assignment PIN_AH1  -to tile_i2c_int

set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 2.5-V SSTL CLASS II" -to tile_i2c_sda
set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 2.5-V SSTL CLASS II" -to tile_i2c_scl
set_instance_assignment -name IO_STANDARD LVDS -to tile_pll_test
set_instance_assignment -name IO_STANDARD LVDS -to tile_chip_reset
set_instance_assignment -name IO_STANDARD LVDS -to tile_cec
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[13]
set_instance_assignment -name IO_STANDARD LVDS -to tile_spi_miso
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[11]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[12]
set_instance_assignment -name IO_STANDARD LVDS -to tile_i2c_int
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[10]
set_instance_assignment -name IO_STANDARD LVDS -to tile_pll_reset
set_instance_assignment -name IO_STANDARD LVDS -to tile_spi_scl
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[9]
set_instance_assignment -name IO_STANDARD LVDS -to tile_spi_mosi
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[8]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[1]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[5]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[2]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[7]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[3]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[6]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[4]

set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_cec
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[13]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_spi_miso
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[11]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[12]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_i2c_int
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[10]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[9]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[8]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[1]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[5]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[2]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[7]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[3]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[6]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[4]
