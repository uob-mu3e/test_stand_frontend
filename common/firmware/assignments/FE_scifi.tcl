
#rx inputs port A
#rx[0:3]={data_in_A_2_0,data_in_A_2_1,data_in_A_2_2,data_in_A_1_2}
for {set i 0} {$i < 4} {incr i} {
    set_instance_assignment -name IO_STANDARD LVDS -to i_fee_rxd[$i]
    set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to i_fee_rxd[$i]
}
set_location_assignment PIN_AH28 -to i_fee_rxd[0]
set_location_assignment PIN_AJ26 -to i_fee_rxd[1]
set_location_assignment PIN_AH26 -to i_fee_rxd[2]
set_location_assignment PIN_AE28 -to i_fee_rxd[3]

#slow control port A

#o_fee_spi_CSn[0:3]={monitor_A_1_p,CLOCK_A_p,data_in_A_1_1_p,data_in_A_1_0}
for {set i 0} {$i < 4} {incr i} {
    set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_CSn[$i]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to malibu_data[0]
}
set_location_assignment PIN_AL28 -to o_fee_spi_CSn[0]
set_location_assignment PIN_AA26 -to o_fee_spi_CSn[1]
set_location_assignment PIN_AF28 -to o_fee_spi_CSn[2]
set_location_assignment PIN_AG29 -to o_fee_spi_CSn[3]

#o_fee_spi_MOSI=SPI_load_A_0
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI
set_location_assignment PIN_AL27 -to o_fee_spi_MOSI

#i_fee_spi_MISO=SPI_SDI_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO
set_location_assignment PIN_AM26 -to o_fee_spi_MISO

#o_fee_spi_SCK=SPI_SCK_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK
set_location_assignment PIN_AL26 -to o_fee_spi_SCK

#o_fee_ext_trig=chip_reset_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig
set_location_assignment PIN_AP10 -to o_fee_ext_trig

#o_fee_chip_rst=fast_reset_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_chip_rst_p
set_location_assignment PIN_W24 -to o_fee_chip_rst

