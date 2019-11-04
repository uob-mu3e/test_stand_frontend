#!/bin/bash
#./killdaq.sh
source ../install/set_env.sh
#sleep 2

odbedit -c "set '/Experiment/Security/Enable non-localhost RPC'  y"
#odbedit -c "set '/Experiment/Security/RPC hosts/Allowed hosts'  '10.20.212.100 10.20.212.13'"

odbedit -c 'mkdir Custom'
odbedit -d Custom -c 'create STRING Path'
odbedit -c "set Custom/Path $SOURCE/mhttpd/custom"

odbedit -c 'mkdir rootana'
odbedit -d rootana -c 'create STRING cors'
odbedit -c 'set rootana/cors *'

#create custom pages for DAQ slow control and Clock/Reset system
odbedit -d Custom -c 'create STRING SC'
odbedit -c 'set Custom/SC sc.html'
odbedit -d Custom -c 'create STRING CR'
odbedit -c 'set Custom/CR cr.html'


#custom pages for SciFi
odbedit -d Custom -c 'create STRING SciFi-ASICs'
odbedit -c 'set Custom/SciFi-ASICs mutrigTdc.html'

odbedit -d Custom -c 'create STRING MutrigAna'
odbedit -c 'set Custom/MutrigAna ana_mutrig/mutrig/mutrigana.html'


tmux new-session -s "DAQ" -d 'bash -i'


tmux rename-window 'Midas' 

tmux split-window -v 'source ../install/set_env.sh; echo "---- MHTTPD ----" ; mhttpd; bash -i'
sleep 1
tmux split-window -v 'source ../install/set_env.sh; echo "---- MSERVER ----" ; mserver; bash -i'
tmux split-window -v 'source ../install/set_env.sh; echo "---- MHIST ----" ; mhist; bash -i'
sleep 1
sleep 1

tmux select-layout tiled

tmux new-window
tmux rename-window 'Frontends' 

tmux split-window -v 'source ../install/set_env.sh; echo "---- SWITCH ----" ; switch_fe_scifi; bash -i'
sleep 1
tmux split-window -v 'source ../install/set_env.sh; echo "---- FARM ----" ; farm_fe; bash -i'
sleep 1
tmux split-window -v 'source ../install/set_env.sh; echo "---- CRFE ----" ; crfe; bash -i'
tmux select-layout tiled



###############################################
## Slow control frontend setup / to be called once, frontends are started on rack pc, not this pc
#########

##set up chiller
#odbedit -c "set '/Equipment/Chiller/Settings/Devices/Huber/BD/RS232 Port' /dev/ttyS0"
#odbedit -c "set '/Equipment/Chiller/Settings/Devices/Huber/BD/Baud' 9600"
#
##set up stage controller
#odbedit -c "set '/Equipment/Stages/Settings/Devices/Stages/BD/RS232 Port' /dev/ttyACM0"
#odbedit -c "set '/Equipment/Stages/Settings/Devices/Stages/BD/Baud' 9600"
#
##set up LV supplies
#odbedit -c "set 'Equipment/LVSupplies/Settings/Devices/LV1/BD/Host' 192.168.0.100"
#odbedit -c "set 'Equipment/LVSupplies/Settings/Devices/LV1/BD/Port' 9221"
#
#odbedit -c "set 'Equipment/LVSupplies/Settings/Devices/LV2/BD/Host' 192.168.0.101"
#odbedit -c "set 'Equipment/LVSupplies/Settings/Devices/LV2/BD/Port' 9221"
#
##set up HV supplies
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV1/BD/RS232' /tmp/keithley1"
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV1/BD/Baud' 9600"
#
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV2/BD/RS232' /tmp/keithley2"
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV2/BD/Baud' 9600"
#
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV3/BD/RS232' /tmp/keithley3"
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV3/BD/Baud' 9600"
#
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV4/BD/RS232' /tmp/keithley4"
#odbedit -c "set 'Equipment/HVSupplies/Settings/Devices/HV4/BD/Baud' 9600"
#tmux new-window -n 'SCFE' "scfe_chiller ; bash -i"
#tmux split-window -h "scfe_stages ; bash -i"
#tmux split-window -h "scfe_VSources ; bash -i"
tmux select-window 0
tmux attach-session
