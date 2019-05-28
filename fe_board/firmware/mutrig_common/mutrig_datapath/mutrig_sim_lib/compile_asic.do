vlib mutrig_sim

################################################################################
############## these modules are common for all the simulations ################
################################################################################
# add the customized modules and surrounding modules
#
## data type definition and functions
vcom -work mutrig_sim -2008 $1/datapath_defs/source/rtl/vhdl/datapath_types.vhd
vcom -work mutrig_sim -2008 $1/datapath_defs/source/rtl/vhdl/datapath_helpers.vhd
vcom -work mutrig_sim -2008 $1/datapath_defs/source/rtl/vhdl/txt_util.vhd
vcom -work mutrig_sim -2008 $1/datapath_defs/source/rtl/vhdl/serial_comm_defs.vhd
#
## analog emulators
vcom -work mutrig_sim -2008 $1/analog_macro_emu/source/rtl/vhdl/ANALOG_CHANNEL_ISOLATED.vhd
vcom -work mutrig_sim -2008 $1/analog_macro_emu/source/rtl/vhdl/TDC_Channel.vhd
vcom -work mutrig_sim -2008 $1/analog_macro_emu/source/rtl/vhdl/TimeBase.vhd
vcom -work mutrig_sim -2008 $1/analog_macro_emu/source/rtl/vhdl/LVDS_RX_top.vhd
vcom -work mutrig_sim -2008 $1/analog_macro_emu/source/rtl/vhdl/LVDS_TX_top.vhd
#
## deserializer and frame_rcv (with CRC check)
#vcom -work mutrig_sim -2008 $1/deserializer/source/rtl/vhdl/bclock_gen.vhd
#vcom -work mutrig_sim -2008 $1/deserializer/source/rtl/vhdl/dec_8b10b.vhd
#vcom -work mutrig_sim -2008 $1/deserializer/source/rtl/vhdl/deserializer.vhd
#vcom -work mutrig_sim -2008 $1/deserializer/source/rtl/vhdl/crc16_calc.vhd
#vcom -work mutrig_sim -2008 $1/frame_rcv/source/rtl/vhdl/frame_rcv.vhd
#vcom -work mutrig_sim -2008 $1/block_deser_frame_rcv/source/rtl/vhdl/block_deser_frame_rcv.vhd


################################################################################
############## these modules are DUTs in vhdl for behaviour simulations ########
################################################################################
# ## hdlcore_lib modules
vcom -work mutrig_sim -2008 $1/hdlcore_lib/generic_arbitration/units/arb_selection/source/rtl/vhdl/arb_selection_alter.vhd
vcom -work mutrig_sim -2008 $1/hdlcore_lib/generic_arbitration/units/generic_mux_chnumappend/source/rtl/vhdl/generic_mux.vhd
#vcom -work mutrig_sim -2008 $1/hdlcore_lib/generic_memory/generic_dp_ram/source/rtl/vhdl/generic_dp_ram.vhd
vcom -work mutrig_sim -2008 $1/hdlcore_lib/generic_memory/fifo_wtrig/source/rtl/vhdl/fifo_wtrig_entity.vhd
vcom -work mutrig_sim -2008 $1/hdlcore_lib/generic_memory/fifo_wtrig/source/rtl/vhdl/fifo_wtrig_arch_generic_ram.vhd
vcom -work mutrig_sim -2008 $1/hdlcore_lib/generic_memory/generic_dp_fifo/source/rtl/vhdl/generic_dp_fifo.vhd
#
## SRAM wrapper
vcom -work mutrig_sim -2008 $1/SRAM/memaker_output/256X78/SZ180_256X78X1CM2.vhd
vcom -work mutrig_sim -2008 $1/SRAM/memaker_output/256X48/SZ180_256X48X1CM4.vhd
vcom -work mutrig_sim -2008 $1/SRAM/source/rtl/vhdl/generic_dp_ram.vhd
#
## clk divider and rst_gen
vcom -work mutrig_sim -2008 $1/clock_divider/source/rtl/vhdl/clock_divider_sreg_counter_longedge.vhd
vcom -work mutrig_sim -2008 $1/reset_generator/source/rtl/vhdl/reset_generator_single_clk.vhd
vcom -work mutrig_sim -2008 $1/synchronizer/source/rtl/vhdl/synchronizer.vhd
#
## L1
vcom -work mutrig_sim -2008 $1/ch_event_counter/source/rtl/vhdl/ch_event_counter.vhd
vcom -work mutrig_sim -2008 $1/therm_decode/source/rtl/vhdl/therm_decode.vhd
vcom -work mutrig_sim -2008 $1/ch_hit_receiver/source/rtl/vhdl/ch_hit_receiver.vhdl
vcom -work mutrig_sim -2008 $1/L1_arbitration/source/rtl/vhdl/L1_arbitration.vhd
vcom -work mutrig_sim -2008 $1/group_buffer/source/rtl/vhdl/group_buffer.vhd
#
## L2, MSselection
vcom -work mutrig_sim -2008 $1/MS_select/source/rtl/vhdl/MS_select.vhd
vcom -work mutrig_sim -2008 $1/group_select/source/rtl/vhdl/group_select.vhd
vcom -work mutrig_sim -2008 $1/GroupMasterSelect/source/rtl/vhdl/GroupMasterSelect.vhd
#
## frame_gen, 8b/10b, serializer
vcom -work mutrig_sim -2008 $1/frame_generator/source/rtl/vhdl/crc16_8.vhd
vcom -work mutrig_sim -2008 $1/frame_generator/source/rtl/vhdl/frame_generator.vhd
vcom -work mutrig_sim -2008 $1/8b10b_encoder/source/rtl/vhdl/8b10_enc.vhd
vcom -work mutrig_sim -2008 $1/8b10b_encoder/source/rtl/vhdl/encoder_module.vhd
vcom -work mutrig_sim -2008 $1/dual_edge_flipflop/source/rtl/vhdl/dual_edge_flipflop.vhd
vcom -work mutrig_sim -2008 $1/dual_edge_serializer/source/rtl/vhdl/dual_edge_serializer.vhd
vcom -work mutrig_sim -2008 $1/init_transmission/source/rtl/vhdl/init_transmission.vhd
vcom -work mutrig_sim -2008 $1/prbs_gen/source/rtl/vhdl/prbs_gen48.vhd
vcom -work mutrig_sim -2008 $1/block_frame_gen_ser/source/rtl/vhdl/block_frame_gen_ser.vhd
#
## SPI
vcom -work mutrig_sim -2008 $1/spi_slave/source/rtl/vhdl/spi_slave.vhdl
vcom -work mutrig_sim -2008 $1/spi_master_ch_ent_cnt/source/rtl/vhdl/spi_master_ch_ent_cnt.vhd
#
## digital_all
vcom -work mutrig_sim -2008 $1/digital_all/source/rtl/vhdl/digital_all.vhdl
#
## DUT
vcom -work mutrig_sim -2008 $1/stic3_top/source/rtl/vhdl/stic3_top.vhd

################################################################################
########### these modules are gate level DUTs in verilog from dc_shell #########
################################################################################
## FARADAY RAM cells
# vcom -work mutrig_sim -2008 $1/SRAM/memaker_output/256X78/SZ180_256X78X1CM2.v
# vcom -work mutrig_sim -2008 $1/SRAM/memaker_output/256X48/SZ180_256X48X1CM4.v
# ## top level cell
# vcom -work mutrig_sim -2008 $1l2gds/dc_shell/results/stic3_top.v


################################################################################
########## these modules are gate level DUTs in verilog from encounter #########
################################################################################
## FARADAY RAM cells
#vcom -work mutrig_sim -2008 $1/SRAM/memaker_output/256X78/SZ180_256X78X1CM2.v
#vcom -work mutrig_sim -2008 $1/SRAM/memaker_output/256X48/SZ180_256X48X1CM4.v
## top level cell
#vcom -work mutrig_sim -2008 $1l2gds/encounter/results/stic3_top.v


################################################################################
########################## this is the test bench ##############################
################################################################################
#vcom -work mutrig_sim -2008 $1/stic3_top/source/tb/vhdl/tb_stic3_top.vhd
