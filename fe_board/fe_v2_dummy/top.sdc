# global clock constrains for feb_v2
# M. Mueller, September 2020

# create all the clocks coming from Si-Chip or oscillator
create_clock -period "125 MHz" [ get_ports LVDS_clk_si1_fpga_A ]
create_clock -period "125 MHz" [ get_ports LVDS_clk_si1_fpga_B ]
create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[0] ]
create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[1] ]
create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[2] ]
create_clock -period "125 MHz" [ get_ports lvds_firefly_clk ]
create_clock -period "50 MHz" [ get_ports systemclock ]
create_clock -period "50 MHz" [ get_ports systemclock_bottom ]
create_clock -period "125 MHz" [ get_ports clk_125_top ]
create_clock -period "125 MHz" [ get_ports clk_125_bottom ]
create_clock -period "50 MHz" [ get_ports spare_clk_osc ]

# By default, Quartus assumes that all of these clocks are related
# (for example: systemclock and spare_clk_osc are identical for Quartus right now)
# So we need to group clocks and specify which of them are asynchronous
# (i'm not sure if we need to divide the first 3 groups or just spare_clk_osc, however .. i would go for 4 groups here and assume them to be independent)

set_clock_groups -asynchronous -group { LVDS_clk_si1_fpga_A LVDS_clk_si1_fpga_B lvds_firefly_clk clk_125_top clk_125_bottom}
set_clock_groups -asynchronous -group { transceiver_pll_clock[0] transceiver_pll_clock[1] transceiver_pll_clock[2] }
set_clock_groups -asynchronous -group { systemclock systemclock_bottom }
set_clock_groups -asynchronous -group { spare_clk_osc }

# set_clock_groups is basically a set_false_path to all regs driven by a different group.
# This means that the timing analyser will not find ANY timing violations between these groups, only in the group itself.
# and it also means that quartus will not try to optimise paths between these groups.

# Now in theory this is fine because we can not produce time violations between clock domains
# if everything is properly synchronised by a dc-Fifo or sync-chain.

# Problem: How do we ensure that?  
# (.. i don't know yet, Quartus reports clock domain crossings, but i can only find absolute numers right now and not individual crossings)

# In the Stratix IV version (and also everywhere else) we did not use set_clock_groups and got no timing violations.
# If we don't use set_clock_groups I believe this is what happens:
#   - Quartus assumes all clocks to be related
#   - Quartus might be able to arrange a save transition between related 50 Mhz and 125 Mhz clocks without a sync chain
#   - Maybe also between other domains (156.25 - 125 etc., there is a relatively small least common multiple)
#   - This is just wrong in some cases (spare_clk_osc)
#   - If quartus did not manage to do that we get a timing violation (If it does manage we get none, even if the transition is unsafe)
#   - When we got a timing violation we looked at it, spotted the missing sync chain, inserted a sync chain
#   - Quartus: Has now 5 registers of the sync chain to play with in placement --> thinks it has done a  "safe" transition between related 50 Mhz and 125 Mhz clocks
#   - --> transition is save now, but for the wrong reasons
#   - --> some transitions are not detected at all because quartus thinks it has done that safely


# https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/manual/mnl_sdctmq.pdf
# https://www.youtube.com/watch?v=hfaiPxl9Z9A

# derive pll clocks from base clocks
derive_pll_clocks -create_base_clocks
derive_clock_uncertainty