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
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "create DOUBLE V"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Input -c "set V 0"

#Output variables
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Output -c "create DOUBLE Voltage[32]"
odbedit -d /Equipment/Mupix/QCTests/Ladder/IV/Output -c "create DOUBLE Current[32]"
