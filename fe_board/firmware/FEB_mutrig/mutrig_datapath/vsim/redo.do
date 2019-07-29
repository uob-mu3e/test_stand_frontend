do compile.do
restart -force
do ../mutrig_sim_lib/force_asic.do   testbench/gen_asic(0)/asic 0
do ../mutrig_sim_lib/force_asic.do   testbench/gen_asic(1)/asic 1
do ../mutrig_sim_lib/force_asic.do   testbench/gen_asic(2)/asic 2
do ../mutrig_sim_lib/force_config.do testbench/gen_asic(0)/asic 0
do ../mutrig_sim_lib/force_config.do testbench/gen_asic(1)/asic 1
do ../mutrig_sim_lib/force_config.do testbench/gen_asic(2)/asic 2
run 450 ns
