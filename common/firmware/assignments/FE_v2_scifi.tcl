# CON2
set_location_assignment PIN_AG16 -to scifi_csn[0]
set_location_assignment PIN_AG6  -to scifi_csn[1]
set_location_assignment PIN_AE11 -to scifi_csn[2]
set_location_assignment PIN_AA10 -to scifi_csn[3]
set_location_assignment PIN_A5   -to scifi_cec_csn[0]
set_location_assignment PIN_AC12 -to scifi_cec_csn[1]
set_location_assignment PIN_AC15 -to scifi_cec_csn[2]
set_location_assignment PIN_AD15 -to scifi_cec_csn[3]
set_location_assignment PIN_AB12 -to scifi_cec_miso
set_location_assignment PIN_A4   -to scifi_fifo_ext
set_location_assignment PIN_AB18 -to scifi_inject
set_location_assignment PIN_AD22 -to scifi_syncres
set_location_assignment PIN_AC21 -to scifi_spi_sclk
set_location_assignment PIN_AB17 -to scifi_spi_miso
set_location_assignment PIN_AK19 -to scifi_spi_mosi
set_location_assignment PIN_AB22 -to scifi_din[0]
set_location_assignment PIN_AH23 -to scifi_din[1]
set_location_assignment PIN_AG22 -to scifi_din[2]
set_location_assignment PIN_AG21 -to scifi_din[3]
set_location_assignment PIN_AJ21 -to scifi_temp_mutrig
set_location_assignment PIN_AK22 -to scifi_temp_sipm

set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[0]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[1]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[2]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[3]

set_instance_assignment -name IO_STANDARD LVDS -to scifi_inject
set_instance_assignment -name IO_STANDARD LVDS -to scifi_syncres
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_sclk
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_miso
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_mosi
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[0]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[1]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[2]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[3]

set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_spi_miso
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[0]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[1]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[2]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[3]


# CON3
set_location_assignment PIN_B10  -to scifi_csn[4]
set_location_assignment PIN_B7   -to scifi_csn[5]
set_location_assignment PIN_D12  -to scifi_csn[6]
set_location_assignment PIN_D15  -to scifi_csn[7]
set_location_assignment PIN_C8   -to scifi_cec_csn[4]
set_location_assignment PIN_F15  -to scifi_cec_csn[5]
set_location_assignment PIN_G17  -to scifi_cec_csn[6]
set_location_assignment PIN_F17  -to scifi_cec_csn[7]
set_location_assignment PIN_G15  -to scifi_cec_miso2
set_location_assignment PIN_D8   -to scifi_fifo_ext2
set_location_assignment PIN_D19  -to scifi_inject2
set_location_assignment PIN_AH8  -to scifi_syncres2
set_location_assignment PIN_AD9  -to scifi_spi_sclk2
set_location_assignment PIN_AF6  -to scifi_spi_miso2
set_location_assignment PIN_AD16 -to scifi_spi_mosi2
set_location_assignment PIN_AB15 -to scifi_din[4]
set_location_assignment PIN_AB6  -to scifi_din[5]
set_location_assignment PIN_AK16 -to scifi_din[6]
set_location_assignment PIN_AK2  -to scifi_din[7]
set_location_assignment PIN_AH2  -to scifi_temp_mutrig2
set_location_assignment PIN_AH1  -to scifi_temp_sipm2

set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[4]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[5]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[6]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[7]

set_instance_assignment -name IO_STANDARD LVDS -to scifi_inject2
set_instance_assignment -name IO_STANDARD LVDS -to scifi_syncres2
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_sclk2
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_miso2
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_mosi2
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[4]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[5]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[6]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[7]

set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_spi_miso2
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[5]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[6]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[7]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[8]
