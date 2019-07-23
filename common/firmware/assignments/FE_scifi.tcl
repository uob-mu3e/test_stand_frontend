#----------------------------------------------------------------------------
#PORT A ---------------------------------------------------------------------
#rx inputs
#rx[0:3]={data_in_A_2_0,data_in_A_2_1,data_in_A_2_2,data_in_A_1_2}
for {set i 0} {$i < 4} {incr i} {
    set_instance_assignment -name IO_STANDARD LVDS -to i_fee_rxd[$i]
    set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to i_fee_rxd[$i]
}
set_location_assignment PIN_AH28 -to i_fee_rxd[0]
set_location_assignment PIN_AJ26 -to i_fee_rxd[1]
set_location_assignment PIN_AH26 -to i_fee_rxd[2]
set_location_assignment PIN_AE28 -to i_fee_rxd[3]

#slow control
#o_fee_spi_CSn[0:3]={monitor_A_1_p,CLOCK_A_p,data_in_A_1_1_p,data_in_A_1_0}
for {set i 0} {$i < 4} {incr i} {
    set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[$i]
}
set_location_assignment PIN_AL28 -to o_fee_spi_CSn[0]
set_location_assignment PIN_AA26 -to o_fee_spi_CSn[1]
set_location_assignment PIN_AF28 -to o_fee_spi_CSn[2]
set_location_assignment PIN_AG29 -to o_fee_spi_CSn[3]

#o_fee_spi_MOSI=SPI_load_A_0
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI
set_location_assignment PIN_AL27 -to o_fee_spi_MOSI[0]

#i_fee_spi_MISO=SPI_SDI_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO
set_location_assignment PIN_AM26 -to o_fee_spi_MISO[0]

#o_fee_spi_SCK=SPI_SCK_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK
set_location_assignment PIN_AL26 -to o_fee_spi_SCK[0]

#o_fee_ext_trig=chip_reset_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig
set_location_assignment PIN_AP10 -to o_fee_ext_trig[0]

#o_fee_chip_rst=fast_reset_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_chip_rst_p
set_location_assignment PIN_W24 -to o_fee_chip_rst[0]

#----------------------------------------------------------------------------
#PORT B ---------------------------------------------------------------------

#rx inputs
#rx[0:3]={data_B_2_0,data_B_2_1,data_B_2_2,data_B_1_2}
for {set i 5} {$i < 8} {incr i} {
    set_instance_assignment -name IO_STANDARD LVDS -to i_fee_rxd[$i]
    set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to i_fee_rxd[$i]
}
set_location_assignment PIN_G28 -to i_fee_rxd[4]
set_location_assignment PIN_F26 -to i_fee_rxd[5]
set_location_assignment PIN_G26 -to i_fee_rxd[6]
set_location_assignment PIN_L26 -to i_fee_rxd[7]

#slow control

#o_fee_spi_CSn[0:3]={monitor_B_1_p,CLOCK_B_p,data_B_1_1_p,data_B_1_0}
for {set i 0} {$i < 4} {incr i} {
    set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[$i]
}
set_location_assignment PIN_D28 -to o_fee_spi_CSn[0]
set_location_assignment PIN_M26 -to o_fee_spi_CSn[1]
set_location_assignment PIN_J28 -to o_fee_spi_CSn[2]
set_location_assignment PIN_H29 -to o_fee_spi_CSn[3]

#o_fee_spi_MOSI=SPI_load_B_0
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI
set_location_assignment PIN_AN30 -to o_fee_spi_MOSI

#i_fee_spi_MISO=SPI_SDI_B
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO
set_location_assignment PIN_AP31 -to o_fee_spi_MISO

#o_fee_spi_SCK=SPI_SCK_B
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK
set_location_assignment PIN_AP30 -to o_fee_spi_SCK

#o_fee_ext_trig=chip_reset_B
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig
set_location_assignment PIN_AL22 -to o_fee_ext_trig

#o_fee_chip_rst=fast_reset_B
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_chip_rst_p
set_location_assignment PIN_N26 -to o_fee_chip_rst

set_instance_assignment -name IO_STANDARD "2.5 V" -to mscb_data_in
set_instance_assignment -name IO_STANDARD "2.5 V" -to mscb_data_out
set_instance_assignment -name IO_STANDARD "2.5 V" -to mscb_oe
set_location_assignment PIN_A23 -to mscb_data_in
set_location_assignment PIN_A24 -to mscb_data_out
set_location_assignment PIN_A25 -to mscb_oe
