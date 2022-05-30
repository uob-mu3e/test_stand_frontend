#
# CON3 = tileB, CON2 = tileA

set_location_assignment PIN_AG16 -to tileB_spi_sclk
set_location_assignment PIN_B10  -to tileA_spi_sclk_n

set_location_assignment PIN_AD15 -to tileA_spi_mosi_n
set_location_assignment PIN_F17  -to tileB_spi_mosi_n

set_location_assignment PIN_A5   -to tileA_spi_miso_n
set_location_assignment PIN_C8   -to tileB_spi_miso_n

set_location_assignment PIN_AD22 -to tileA_pll_reset
set_location_assignment PIN_AH8  -to tileB_pll_reset

set_location_assignment PIN_AC21 -to tileA_pll_test
set_location_assignment PIN_AD9  -to tileB_pll_test

set_location_assignment PIN_AH19 -to tileA_i2c_sda_io
set_location_assignment PIN_AG7  -to tileB_i2c_sda_io

set_location_assignment PIN_AH18 -to tileA_i2c_scl_io
set_location_assignment PIN_AH7  -to tileB_i2c_scl_io

set_location_assignment PIN_A14  -to tile_din[13]
set_location_assignment PIN_F15  -to tile_din[12]
set_location_assignment PIN_A15  -to tile_din[11]
set_location_assignment PIN_J11  -to tile_din[1]
set_location_assignment PIN_C14  -to tile_din[10]
set_location_assignment PIN_C13  -to tile_din[2]
set_location_assignment PIN_J12  -to tile_din[8]
set_location_assignment PIN_D9   -to tile_din[9]
set_location_assignment PIN_AB6  -to tile_din[3]
set_location_assignment PIN_AJ4  -to tile_din[4]
set_location_assignment PIN_AK16 -to tile_din[5]
set_location_assignment PIN_AK3  -to tile_din[7]
set_location_assignment PIN_AK2  -to tile_din[6]

#set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 2.5-V SSTL CLASS II" -to tile_i2c_sda
#set_instance_assignment -name IO_STANDARD "DIFFERENTIAL 2.5-V SSTL CLASS II" -to tile_i2c_scl
set_instance_assignment -name IO_STANDARD "2.5-V" -to tileA_i2c_sda_io
set_instance_assignment -name IO_STANDARD "2.5-V" -to tileA_i2c_scl_io
set_instance_assignment -name IO_STANDARD "2.5-V" -to tileB_i2c_sda_io
set_instance_assignment -name IO_STANDARD "2.5-V" -to tileB_i2c_scl_io
set_instance_assignment -name IO_STANDARD LVDS -to tileA_spi_sclk_n
set_instance_assignment -name IO_STANDARD LVDS -to tileB_spi_sclk
set_instance_assignment -name IO_STANDARD LVDS -to tileA_pll_test
set_instance_assignment -name IO_STANDARD LVDS -to tileB_pll_test
set_instance_assignment -name IO_STANDARD LVDS -to tileA_spi_miso_n
set_instance_assignment -name IO_STANDARD LVDS -to tileB_spi_miso_n
set_instance_assignment -name IO_STANDARD LVDS -to tileA_pll_reset
set_instance_assignment -name IO_STANDARD LVDS -to tileB_pll_reset
set_instance_assignment -name IO_STANDARD LVDS -to tileA_spi_mosi_n
set_instance_assignment -name IO_STANDARD LVDS -to tileB_spi_mosi_n

set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tileA_spi_miso_n
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tileB_spi_miso_n

set_instance_assignment -name IO_STANDARD LVDS -to tile_din[13]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[11]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[12]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[10]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[9]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[8]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[1]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[5]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[2]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[7]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[3]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[6]
set_instance_assignment -name IO_STANDARD LVDS -to tile_din[4]

set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[13]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[11]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to tile_din[12]
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
