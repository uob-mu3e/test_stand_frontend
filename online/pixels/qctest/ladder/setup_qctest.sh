# Power suppliers configuration
odbedit -d /Equipment/HAMEG0/Common/ -c "set Period 500"


odbedit -d /Sequencer/State -c "set Path ${PWD}"

odbedit -d /Equipment/Mupix/ -c "mkdir QCTests"
odbedit -d /Equipment/Mupix/QCTests -c "mkdir Ladder"
odbedit -d /Equipment/Mupix/QCTests/Ladder -c "mkdir Eval"


#Test 1 : IV curve
odbedit -d /Equipment/Mupix/QCTests/Ladder -c "mkdir IV"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV -c "mkdir Output"
#Input parameters
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "create DOUBLE current_limit"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "set start_voltage 0"
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
# Output
odbedit -d /Equipment/Mupix/QCTests/Ladder/LVPowerOn/Output -c "create DOUBLE current_increase[3]"


# Test 3 : DAC Scan test
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c "mkdir DACScan"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan -c "mkdir Output"
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
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC -c "create DOUBLE VPDAC_values[10]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/VPDAC -c "create DOUBLE VPDAC_current[10]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output -c "mkdir ref_Vss"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss -c "create DOUBLE ref_Vss_values[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/DACScan/Output/ref_Vss -c "create DOUBLE ref_Vss_current[32]"


# Test 4: link quality test 
odbedit -d /Equipment/Mupix/QCTests/Ladder/ -c "mkdir LINKQUALIcheck"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck -c "mkdir Output"
# Input
odbedit /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/ -c "mkdir Input"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Input/ -c "create DOUBLE HV_voltage"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Input/ -c "create DOUBLE HV_curr_limit"
# Output 
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 0"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 1"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 2"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 3"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 4"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output -c "mkdir 5"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0 -c "create DOUBLE VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0 -c "create DOUBLE VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/0 -c "create DOUBLE ErrorRate"


odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1 -c "create DOUBLE VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1 -c "create DOUBLE VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/1 -c "create DOUBLE ErrorRate"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2 -c "create DOUBLE VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2 -c "create DOUBLE VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/2 -c "create DOUBLE ErrorRate"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/3 -c "create DOUBLE VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/3 -c "create DOUBLE VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/3 -c "create DOUBLE ErrorRate"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/4 -c "create DOUBLE VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/4 -c "create DOUBLE VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/4 -c "create DOUBLE ErrorRate"

odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/5 -c "create DOUBLE VPVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/5 -c "create DOUBLE VNVCO"
odbedit -d /Equipment/Mupix/QCTests/Ladder/LINKQUALIcheck/Output/5 -c "create DOUBLE ErrorRate"