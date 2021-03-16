#
set_location_assignment PIN_AG16 -to scifi_csn[1]
set_location_assignment PIN_AG6  -to scifi_csn[2]
set_location_assignment PIN_AE11 -to scifi_csn[3]
set_location_assignment PIN_AA10 -to scifi_csn[4]
set_location_assignment PIN_AD15 -to scifi_cec_cs[1]
set_location_assignment PIN_AC15 -to scifi_cec_cs[2]
set_location_assignment PIN_AB11 -to scifi_cec_cs[3]
set_location_assignment PIN_AB12 -to scifi_cec_cs[4]
set_location_assignment PIN_AA12 -to scifi_cec_sdo
set_location_assignment PIN_AC12 -to scifi_fifo_ext
set_location_assignment PIN_AB18 -to scifi_inject
set_location_assignment PIN_AD22 -to scifi_syncres
set_location_assignment PIN_AC21 -to scifi_spi_sclk
set_location_assignment PIN_AB17 -to scifi_spi_miso
set_location_assignment PIN_AK19 -to scifi_spi_mosi
set_location_assignment PIN_AB22 -to scifi_din[1]
set_location_assignment PIN_AH23 -to scifi_din[2]
set_location_assignment PIN_AG22 -to scifi_din[3]
set_location_assignment PIN_AG21 -to scifi_din[4]
set_location_assignment PIN_AH20 -to scifi_bidir_test

set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[1]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[2]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[3]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_csn[4]

set_instance_assignment -name IO_STANDARD LVDS -to scifi_inject
set_instance_assignment -name IO_STANDARD LVDS -to scifi_syncres
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_sclk
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_miso
set_instance_assignment -name IO_STANDARD LVDS -to scifi_spi_mosi
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[1]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[2]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[3]
set_instance_assignment -name IO_STANDARD LVDS -to scifi_din[4]

set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_spi_miso
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[1]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[2]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[3]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to scifi_din[4]
