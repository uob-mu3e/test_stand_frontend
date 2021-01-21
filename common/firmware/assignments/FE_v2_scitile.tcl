#
set_location_assignment PIN_AG16 -to tile_pll_test
set_location_assignment PIN_AD15 -to tile_chip_reset
set_location_assignment PIN_AE11 -to tile_i2c_sda
set_location_assignment PIN_AB18 -to tile_i2c_scl
set_location_assignment PIN_AB11 -to tile_cec
set_location_assignment PIN_AC12 -to tile_din[13]
set_location_assignment PIN_AH11 -to tile_spi_miso
set_location_assignment PIN_AF13 -to tile_din[11]
set_location_assignment PIN_AH12 -to tile_din[12]
set_location_assignment PIN_AK10 -to tile_i2c_int
set_location_assignment PIN_AJ13 -to tile_din[10]
set_location_assignment PIN_AD22 -to tile_pll_reset
set_location_assignment PIN_AC21 -to tile_spi_scl
set_location_assignment PIN_AB17 -to tile_din[9]
set_location_assignment PIN_AK19 -to tile_spi_mosi
set_location_assignment PIN_AH18 -to tile_din[8]
set_location_assignment PIN_AB22 -to tile_din[1]
set_location_assignment PIN_AH20 -to tile_din[5]
set_location_assignment PIN_AH23 -to tile_din[2]
set_location_assignment PIN_AK21 -to tile_din[7]
set_location_assignment PIN_AG22 -to tile_din[3]
set_location_assignment PIN_AK22 -to tile_din[6]
set_location_assignment PIN_AG21 -to tile_din[4]

set_instance_assignment -name IO_STANDARD LVDS -to tile_pll_test
set_instance_assignment -name IO_STANDARD LVDS -to tile_chip_reset
set_instance_assignment -name IO_STANDARD LVDS -to tile_i2c_sda
set_instance_assignment -name IO_STANDARD LVDS -to tile_i2c_scl
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
