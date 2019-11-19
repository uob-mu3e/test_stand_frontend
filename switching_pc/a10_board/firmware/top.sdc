#**************************************************************
# This .sdc file is created by Terasic Tool.
# Users are recommended to modify this file to match users logic.
#**************************************************************

#**************************************************************
# Create Clock
#**************************************************************
create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2J]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2L]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B3D]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B3F]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B3H]
#create_clock -period "100.000000 MHz" [get_ports CLK_100_B3D]
#create_clock -period "100.000000 MHz" [get_ports CLKUSR_100]
#create_clock -period "266.667000 MHz" [get_ports DDR3A_REFCLK_p]
#create_clock -period "266.667000 MHz" [get_ports DDR3B_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIIA_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIIB_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIIC_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIID_REFCLK_p]
create_clock -period "125.000000 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]
#create_clock -period "156.250000 MHz" [get_ports refclk1_qr0_p]
#create_clock -period "125.000000 MHz" [get_ports refclk2_qr1_p]
#create_clock -period "250.000000 MHz" [get_ports QSFPB_REFCLK_p]
#create_clock -period "644.531250 MHz" [get_ports QSFPC_REFCLK_p]
#create_clock -period "644.531250 MHz" [get_ports QSFPD_REFCLK_p]
#create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]
#create_clock -period "50 MHz" [get_ports SMA_CLKIN]

#**************************************************************
# Create Generated Clock
#**************************************************************
derive_pll_clocks -create_base_clocks


#**************************************************************
# Set Clock Latency
#**************************************************************


#**************************************************************
# Set Clock Uncertainty
#**************************************************************
derive_clock_uncertainty


#**************************************************************
# Set Input Delay
#**************************************************************



#**************************************************************
# Set Output Delay
#**************************************************************



#**************************************************************
# Set Clock Groups
#**************************************************************



#**************************************************************
# Set False Path
#**************************************************************
set_false_path -to {clk_sync}


#**************************************************************
# Set Multicycle Path
#**************************************************************



#**************************************************************
# Set Maximum Delay
#**************************************************************
set_max_delay -to {xcvr_a10:*|av_ctrl.readdata[*]} 100
set_max_delay -to {readregs[*]} 100
set_max_delay -to {writeregs_slow[*]} 100


#**************************************************************
# Set Minimum Delay
#**************************************************************
set_min_delay -to {xcvr_a10:*|av_ctrl.readdata[*]} -100
set_min_delay -to {readregs[*]} -100
set_min_delay -to {writeregs_slow[*]} -100


#**************************************************************
# Set Skew
#**************************************************************



#**************************************************************
# Set Input Transition
#**************************************************************



#**************************************************************
# Set Load
#**************************************************************


