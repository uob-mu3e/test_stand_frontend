#!/bin/bash

source ../install/set_env.sh

odbedit -c "set '/Experiment/Security/Enable non-localhost RPC'  y"
#odbedit -c "set '/Experiment/Security/RPC hosts/Allowed hosts'  '10.20.212.100 10.20.212.13'"

odbedit -c 'mkdir Custom'
odbedit -d Custom -c 'create STRING Path'
odbedit -c "set Custom/Path $SOURCE/mhttpd/custom"

odbedit -c 'mkdir rootana'
odbedit -d rootana -c 'create STRING cors'
odbedit -c 'set rootana/cors *'

# create custom pages for DAQ slow control
odbedit -d Custom -c 'create STRING SC'
odbedit -c 'set Custom/SC sc.html'
odbedit -d Custom -c 'create STRING SciFi-ASICs'
odbedit -c 'set Custom/SciFi-ASICs mutrigTdc.html'
#odbedit -d Custom -c 'create STRING MALIBU'
#odbedit -c 'set Custom/MALIBU TileGUI/MALIBUMonitor/Monitor.html'
odbedit -d Custom -c 'create STRING MuPix'
odbedit -c 'set Custom/MuPix MuPixTB.html'

# mutrigana
odbedit -d Custom -c 'create STRING MutrigAna'
odbedit -c 'set Custom/MutrigAna ana_mutrig/mutrig/mutrigana.html'

# set mhttpd to run http
odbedit -d Experiment -c 'create BOOL "http redirect to https"'
odbedit -c 'set "Experiment/http redirect to https" "n"'
