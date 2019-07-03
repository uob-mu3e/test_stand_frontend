
set_instance_assignment -name MEMORY_INTERFACE_DATA_PIN_GROUP 4 -from clock_A -to clock_A -disable
set_instance_assignment -name IO_STANDARD LVDS -to clock_A
set_location_assignment PIN_AA26 -to clock_A
set_location_assignment PIN_AH28 -to data_in_A_0[3]
set_location_assignment PIN_AB27 -to data_in_A_0[2]
set_location_assignment PIN_AD26 -to data_in_A_0[1]
set_location_assignment PIN_AC26 -to data_in_A_0[0]
set_location_assignment PIN_AJ26 -to data_in_A_1[3]
set_location_assignment PIN_AE28 -to data_in_A_1[2]
set_location_assignment PIN_AF28 -to data_in_A_1[1]
set_location_assignment PIN_AG29 -to data_in_A_1[0]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_0[0]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_0[1]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_0[2]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_0[3]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_1[0]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_1[1]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_1[2]
set_instance_assignment -name INPUT_TERMINATION DIFFERENTIAL -to data_in_A_1[3]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_0[0]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_0[1]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_0[2]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_0[3]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_1[0]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_1[1]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_1[2]
set_instance_assignment -name IO_STANDARD LVDS -to data_in_A_1[3]
set_instance_assignment -name IO_STANDARD LVDS -to fast_reset_A
set_location_assignment PIN_W24 -to fast_reset_A
set_location_assignment PIN_AP14 -to test_pulse_A
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to test_pulse_A
set_location_assignment PIN_AP25 -to CTRL_SDO_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to CTRL_SDO_A
set_location_assignment PIN_AM26 -to CTRL_SDI_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to CTRL_SDI_A
set_location_assignment PIN_AL27 -to CTRL_SCK1_A
set_location_assignment PIN_AJ25 -to CTRL_SCK2_A
set_location_assignment PIN_AH25 -to CTRL_Load_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to CTRL_SCK1_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to CTRL_SCK2_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to CTRL_Load_A
set_location_assignment PIN_AL26 -to CTRL_RB_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to CTRL_RB_A
set_location_assignment PIN_AP10 -to chip_reset_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to chip_reset_A
set_location_assignment PIN_AJ29 -to SPI_DIN0_A
set_location_assignment PIN_AM28 -to SPI_DIN1_A
set_location_assignment PIN_AJ28 -to SPI_CLK_A
set_location_assignment PIN_AL28 -to SPI_LD_DAC_A
set_location_assignment PIN_AK29 -to SPI_LD_ADC_A
set_location_assignment PIN_AL29 -to SPI_LD_TEMP_DAC_A
set_location_assignment PIN_AH26 -to SPI_DOUT_ADC_0_A
set_location_assignment PIN_AJ27 -to SPI_DOUT_ADC_1_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_DIN0_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_DIN1_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_CLK_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_LD_DAC_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_LD_ADC_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_LD_TEMP_DAC_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_DOUT_ADC_0_A
set_instance_assignment -name IO_STANDARD "2.5 V" -to SPI_DOUT_ADC_1_A
