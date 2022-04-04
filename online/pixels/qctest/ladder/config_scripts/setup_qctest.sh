# Author J. Guzman-Funck, March 2022, Cosmic Run. jose.guzman-funck19@imperial.ac.uk || pepe.guzmanfunck@gmail.com

# Power suppliers configuration. TODO: add this to all available hamegs
odbedit -d /Equipment/HAMEG0/Common/ -c "set Period 500"


odbedit -d /Sequencer/State -c "set Path ${PWD\%\/\*}"

odbedit -d /Equipment/Mupix/ -c "mkdir QCTests"
odbedit -d /Equipment/Mupix/QCTests -c "mkdir Ladder"
odbedit -d /Equipment/Mupix/QCTests/Ladder -c "mkdir Eval"
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c  "create INT32 current_ladder_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c  "create INT32 current_hameg_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c  "create INT32 current_channel"

#Test 1 : IV curve
odbedit -d /Equipment/Mupix/QCTests/Ladder -c "mkdir IV"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV -c "mkdir Output"
# IDs
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV -c "create INT32 half_ladder_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV -c "create INT32 hameg_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV -c "create INT32 channel"
#Input parameters
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "create DOUBLE hv_current_limit"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "set hv_current_limit 0.00001"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "create DOUBLE start_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "set start_voltage 0"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "create DOUBLE step_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "set step_voltage 5"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "create DOUBLE fine_step_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "set fine_step_voltage 1"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "create DOUBLE stop_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "set stop_voltage 10"
#Output
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Output -c "create DOUBLE Voltage[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Output -c "create DOUBLE Current[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Output -c "create DOUBLE V"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Output -c "set V 0"


# Test 2 : LV power ON test
odbedit -d /Equipment/Mupix/QCTests/Ladder -c "mkdir LVPowerOn"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LVPowerOn -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LVPowerOn -c "mkdir Output"
# IDs
odbedit -d /Equipment/Mupix/QCTests/Ladder/LVPowerOn -c "create INT32 half_ladder_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LVPowerOn -c "create INT32 hameg_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LVPowerOn -c "create INT32 channel"
# Output
odbedit -d /Equipment/Mupix/QCTests/Ladder/LVPowerOn/Output -c "create DOUBLE current_increase[3]"


# Test 3 : DAC Scan test
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c "mkdir DACScan"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan -c "mkdir Output"
# IDs
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan -c "create INT32 half_ladder_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan -c "create INT32 hameg_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan -c "create INT32 channel"
# Input
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input -c "create DOUBLE HV_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input -c "create DOUBLE HV_curr_limit"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input -c "mkdir VPDAC"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input/VPDAC -c "create INT32 start_value"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input/VPDAC -c "create INT32 step"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input/VPDAC -c "create INT32 stop_value"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input -c "mkdir ref_Vss"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input/ref_Vss -c "create INT32 start_value"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input/ref_Vss -c "create INT32 step"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Input/ref_Vss -c "create INT32 stop_value"
# Output 
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output -c "mkdir VPDAC"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC -c "mkdir 0"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC -c "mkdir 1"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC -c "mkdir 2"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC/0 -c "create DOUBLE VPDAC_values[10]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC/0 -c "create DOUBLE VPDAC_current[10]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC/1 -c "create DOUBLE VPDAC_values[10]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC/1 -c "create DOUBLE VPDAC_current[10]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC/2 -c "create DOUBLE VPDAC_values[10]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC/2 -c "create DOUBLE VPDAC_current[10]"

odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output -c "mkdir ref_Vss"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss -c "mkdir 0"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss -c "mkdir 1"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss -c "mkdir 2"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss/0 -c "create DOUBLE ref_Vss_values[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss/0 -c "create DOUBLE ref_Vss_current[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss/1 -c "create DOUBLE ref_Vss_values[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss/1 -c "create DOUBLE ref_Vss_current[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss/2 -c "create DOUBLE ref_Vss_values[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss/2 -c "create DOUBLE ref_Vss_current[32]"


# Test 4: link quality test 
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c "mkdir LINKQUALIcheck"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck -c "mkdir Output"
# IDs
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck -c "create INT32 half_ladder_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck -c "create INT32 hameg_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck -c "create INT32 channel"
# Input
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Input/ -c "create DOUBLE HV_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Input/ -c "create DOUBLE HV_curr_limit"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Input/ -c "create DOUBLE start_magic_value_VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Input/ -c "create DOUBLE stop_magic_value_VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Input/ -c "create DOUBLE waiting_time"
# Output
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 0"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 1"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 2"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0 -c "mkdir Scan"
# TODO think of a good strategy for saving VNVCO as well instead of reconstructing it from vpvco
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0/Scan -c "create DOUBLE VPVCO[23]" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0/Scan -c "create DOUBLE error_rate[23]" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0 -c "create DOUBLE opt_VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0 -c "create DOUBLE opt_VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0 -c "create DOUBLE ErrorRate"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1 -c "mkdir Scan"
# TODO think of a good strategy for saving VNVCO as well instead of reconstructing it from vpvco
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1/Scan -c "create DOUBLE VPVCO[23]" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1/Scan -c "create DOUBLE error_rate[23]" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1 -c "create DOUBLE opt_VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1 -c "create DOUBLE opt_VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1 -c "create DOUBLE ErrorRate"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2 -c "mkdir Scan"
# TODO think of a good strategy for saving VNVCO as well instead of reconstructing it from vpvco
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2/Scan -c "create DOUBLE VPVCO[23]" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2/Scan -c "create DOUBLE error_rate[23]" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2 -c "create DOUBLE opt_VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2 -c "create DOUBLE opt_VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2 -c "create DOUBLE ErrorRate"


# Eval
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "mkdir Insight"

odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/Insight -c "mkdir IV_curve"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/Insight -c "mkdir LV_power_ON"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/Insight -c "mkdir DAC_scan"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/Insight -c "mkdir link_quality"

odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "create INT32 half_ladder_id"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "create STRING grade"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "create BOOL IV_curve"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "create BOOL LV_power_ON[3]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "create BOOL DAC_scan_VPDAC[3]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "create BOOL DAC_scan_ref_Vss[3]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Eval/ -c "create BOOL link_quality[3]"


# Default inputs
# This configuration will not be modified by the scripts. Only accessed. It is not recommended to be changed
# unless you know what you're doing
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c "mkdir Default_config"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config -c "mkdir IV"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config -c "mkdir LVPowerOn"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config -c "mkdir DACScan"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config -c "mkdir LINKQUALIcheck"


odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "create DOUBLE HV_current_limit"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "set HV_current_limit 0.00001"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "create DOUBLE start_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "set start_voltage 0"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "create DOUBLE step_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "set step_voltage 5"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "create DOUBLE fine_step_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "set fine_step_voltage 1" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "create DOUBLE stop_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/IV/ -c "set stop_voltage 30"


odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan -c "create DOUBLE HV_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan -c "set HV_voltage -5"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan -c "create DOUBLE HV_curr_limit"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan -c "set HV_curr_limit 0.00001"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan -c "mkdir VPDAC"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/VPDAC -c "create INT32 start_value"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/VPDAC -c "set start_value 0"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/VPDAC -c "create INT32 step"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/VPDAC -c "set step 1"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/VPDAC -c "create INT32 stop_value"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/VPDAC -c "set stop_value 6"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan -c "mkdir ref_Vss"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/ref_Vss -c "create INT32 start_value"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/ref_Vss -c "set start_value 160"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/ref_Vss -c "create INT32 step"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/ref_Vss -c "set step 2"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/ref_Vss -c "create INT32 stop_value"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/DACScan/ref_Vss -c "set stop_value 220"


odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "create DOUBLE HV_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "set HV_voltage -5" 
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "create DOUBLE HV_curr_limit"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "set HV_curr_limit 0.00001"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "create DOUBLE start_magic_value_VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "set start_magic_value_VPVCO 7"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "create DOUBLE stop_magic_value_VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "set stop_magic_value_VPVCO 29"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "create DOUBLE waiting_time"
odbedit -d /Equipment/Mupix/QCTests/Ladder/Default_config/LINKQUALIcheck/ -c "set waiting_time 3" 
