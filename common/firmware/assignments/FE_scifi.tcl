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
#o_fee_spi_CSn[0:3]={monitor_A_1_p,CLOCK_A_p,data_in_A_1_1_p,data_in_A_1_0_p}
for {set i 0} {$i < 4} {incr i} {
    set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[$i]
}
set_location_assignment PIN_AL28 -to o_fee_spi_CSn[0]
set_location_assignment PIN_AA26 -to o_fee_spi_CSn[1]
set_location_assignment PIN_AF28 -to o_fee_spi_CSn[2]
set_location_assignment PIN_AG29 -to o_fee_spi_CSn[3]

#o_fee_spi_MOSI=SPI_load_A_0
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI[0]
set_location_assignment PIN_AL27 -to o_fee_spi_MOSI[0]

#i_fee_spi_MISO=SPI_load_A_2
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO[0]
set_location_assignment PIN_AH25 -to o_fee_spi_MISO[0]

#o_fee_spi_SCK=SPI_SDO_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK[0]
set_location_assignment PIN_AP25 -to o_fee_spi_SCK[0]

#o_fee_ext_trig=chip_reset_A
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig[0]
set_location_assignment PIN_AP10 -to o_fee_ext_trig[0]

#o_fee_chip_rst=fast_reset_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_chip_rst_p[0]
set_location_assignment PIN_W24 -to o_fee_chip_rst[0]

#----------------------------------------------------------------------------
#PORT B ---------------------------------------------------------------------

#rx inputs
#rx[0:3]={data_B_2_0,data_B_2_1,data_B_2_2,data_B_1_2}
for {set i 4} {$i < 8} {incr i} {
    set_instance_assignment -name IO_STANDARD LVDS -to i_fee_rxd[$i]
    set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to i_fee_rxd[$i]
}
set_location_assignment PIN_G28 -to i_fee_rxd[4]
set_location_assignment PIN_F26 -to i_fee_rxd[5]
set_location_assignment PIN_G26 -to i_fee_rxd[6]
set_location_assignment PIN_L26 -to i_fee_rxd[7]

#slow control
#o_fee_spi_CSn[0:3]={monitor_B_1_p,CLOCK_B_p,data_B_1_1_p,data_B_1_0_p}
for {set i 4} {$i < 8} {incr i} {
    set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[$i]
}
set_location_assignment PIN_D28 -to o_fee_spi_CSn[4]
set_location_assignment PIN_M26 -to o_fee_spi_CSn[5]
set_location_assignment PIN_J28 -to o_fee_spi_CSn[6]
set_location_assignment PIN_H29 -to o_fee_spi_CSn[7]


#o_fee_spi_MOSI=SPI_load_B_0
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI[1]
set_location_assignment PIN_AN30 -to o_fee_spi_MOSI[1]

#i_fee_spi_MISO=SPI_load_B_2
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO[1]
set_location_assignment PIN_AN29 -to o_fee_spi_MISO[1]

#o_fee_spi_SCK=SPI_SDO_B
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK[1]
set_location_assignment PIN_AP29 -to o_fee_spi_SCK[1]

#o_fee_ext_trig=chip_reset_B
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig[1]
set_location_assignment PIN_AL22 -to o_fee_ext_trig[1]

#o_fee_chip_rst=fast_reset_B_p
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_chip_rst_p[1]
set_location_assignment PIN_N25 -to o_fee_chip_rst[1]

#----------------------------------------------------------------------------
#PORT C ---------------------------------------------------------------------
#NOTE: bank 7A used on that port have different voltages!

#rx inputs
#rx[0:3]={data_C_2_0,data_C_2_1,data_C_2_2,data_C_1_2}
for {set i 8} {$i < 12} {incr i} {
    set_instance_assignment -name IO_STANDARD LVDS -to i_fee_rxd[$i]
    set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to i_fee_rxd[$i]
}
set_location_assignment PIN_J5 -to i_fee_rxd[8]
set_location_assignment PIN_J7 -to i_fee_rxd[9]
set_location_assignment PIN_G9 -to i_fee_rxd[10]
set_location_assignment PIN_F7 -to i_fee_rxd[11]

#slow control
#o_fee_spi_CSn[0:3]={monitor_C_1_p,CLOCK_C_p,data_C_1_1_p,data_C_1_0_p}
for {set i 8} {$i < 12} {incr i} {
    set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[$i]
}
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_CSn[8]
set_location_assignment PIN_G11 -to o_fee_spi_CSn[8]
set_location_assignment PIN_H9 -to o_fee_spi_CSn[9]
set_location_assignment PIN_G7 -to o_fee_spi_CSn[10]
set_location_assignment PIN_F9 -to o_fee_spi_CSn[11]

#o_fee_spi_MOSI=SPI_load_C_0
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI[2]
set_location_assignment PIN_AP5 -to o_fee_spi_MOSI[2]

#i_fee_spi_MISO=SPI_load_C_2
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO[2]
set_location_assignment PIN_AP6 -to o_fee_spi_MISO[2]

#o_fee_spi_SCK=SPI_SDO_C
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK[2]
set_location_assignment PIN_AP7 -to o_fee_spi_SCK[2]

#o_fee_ext_trig=chip_reset_C
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig[2]
set_location_assignment PIN_AP8 -to o_fee_ext_trig[2]

#o_fee_chip_rst=fast_reset_C
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_chip_rst_p[2]
set_location_assignment PIN_N12 -to o_fee_chip_rst[2]

#----#----------------------------------------------------------------------------
#----#PORT D --- NOT WORKING, NO LVDS RX -----------------------------------------
#----
#----#rx inputs
#----#rx[0:3]={data_D_2_0,data_D_2_1,data_D_2_2,data_D_1_2}
#----for {set i 12} {$i < 16} {incr i} {
#----    set_instance_assignment -name IO_STANDARD LVDS -to i_fee_rxd[$i]
#----    set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to i_fee_rxd[$i]
#----}
#----set_location_assignment PIN_AN24 -to i_fee_rxd[12]
#----set_location_assignment PIN_AN26 -to i_fee_rxd[13]
#----set_location_assignment PIN_AN27 -to i_fee_rxd[14]
#----set_location_assignment PIN_AN23 -to i_fee_rxd[15]
#----
#----#slow control
#----#o_fee_spi_CSn[0:3]={monitor_D_1_p,CLOCK_D_p,data_D_1_1_p,data_D_1_0_p}
#----for {set i 8} {$i < 12} {incr i} {
#----    set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[$i]
#----}
#----set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_CSn[12]
#----set_location_assignment PIN_AN11 -to o_fee_spi_CSn[12]
#----set_location_assignment PIN_R11 -to o_fee_spi_CSn[13]
#----set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_CSn[14]
#----set_location_assignment PIN_AL25 -to o_fee_spi_CSn[14]
#----set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_CSn[15]
#----set_location_assignment PIN_AL23 -to o_fee_spi_CSn[15]
#----
#----#o_fee_spi_MOSI=SPI_load_D_0
#----set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI[3]
#----set_location_assignment PIN_AF20 -to o_fee_spi_MOSI[3]
#----
#----#i_fee_spi_MISO=SPI_load_D_2
#----set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO[3]
#----set_location_assignment PIN_AP22 -to o_fee_spi_MISO[3]
#----
#----#o_fee_spi_SCK=SPI_SDO_D
#----set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK[3]
#----set_location_assignment PIN_AH22 -to o_fee_spi_SCK[3]
#----
#----#o_fee_ext_trig=chip_reset_D
#----set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig[3]
#----set_location_assignment PIN_AJ21 -to o_fee_ext_trig[3]
#----
#----#o_fee_chip_rst=fast_reset_D
#----set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_chip_rst_p[3]
#----set_location_assignment PIN_N10 -to o_fee_chip_rst[3]

#----------------------------------------------------------------------------
#PORT E ---------------------------------------------------------------------

#rx inputs
#rx[0:3]={data_E_2_0,data_E_2_1,data_E_2_2,data_E_1_2}
for {set i 12} {$i < 16} {incr i} {
    set_instance_assignment -name IO_STANDARD LVDS -to i_fee_rxd[$i]
    set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to i_fee_rxd[$i]
}
set_location_assignment PIN_AM7 -to i_fee_rxd[12]
set_location_assignment PIN_AM6 -to i_fee_rxd[13]
set_location_assignment PIN_AK6 -to i_fee_rxd[14]
set_location_assignment PIN_AJ7 -to i_fee_rxd[15]

#slow control
#o_fee_spi_CSn[0:3]={monitor_E_1_p,CLOCK_E_p,data_E_1_1_p,data_E_1_0_p}
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_CSn[12]
set_location_assignment PIN_AH14 -to o_fee_spi_CSn[12]
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[13]
set_location_assignment PIN_AA11 -to o_fee_spi_CSn[13]
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[14]
set_location_assignment PIN_AJ8 -to o_fee_spi_CSn[14]
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_spi_CSn[15]
set_location_assignment PIN_AK8 -to o_fee_spi_CSn[15]

#o_fee_spi_MOSI=SPI_load_E_0
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MOSI[3]
set_location_assignment PIN_AJ12 -to o_fee_spi_MOSI[3]

#i_fee_spi_MISO=SPI_load_E_2
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_MISO[3]
set_location_assignment PIN_AH11 -to o_fee_spi_MISO[3]

#o_fee_spi_SCK=SPI_SDO_E
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_spi_SCK[3]
set_location_assignment PIN_AL10 -to o_fee_spi_SCK[3]

#o_fee_ext_trig=chip_reset_E
set_instance_assignment -name IO_STANDARD "1.8 V" -to o_fee_ext_trig[3]
set_location_assignment PIN_AF14 -to o_fee_ext_trig[3]

#o_fee_chip_rst=fast_reset_E_p
set_instance_assignment -name IO_STANDARD "2.5 V" -to o_fee_chip_rst_p[3]
set_location_assignment PIN_AA9 -to o_fee_chip_rst[3]



set_instance_assignment -name IO_STANDARD "2.5 V" -to mscb_data_in
set_instance_assignment -name IO_STANDARD "2.5 V" -to mscb_data_out
set_instance_assignment -name IO_STANDARD "2.5 V" -to mscb_oe
set_location_assignment PIN_A23 -to mscb_data_in
set_location_assignment PIN_A24 -to mscb_data_out
set_location_assignment PIN_A25 -to mscb_oe
