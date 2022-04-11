#!/bin/bash
# clean.sh

killall crfe
killall mlogger
killall mserver
killall mhttpd
killall malibu_fe
killall msysmon
killall msysmon-nvidia
killall msequencer

#soemwhat rash...
killall xterm

#Go to build dir
cd "/home/labor/Desktop/Pepe/online/build"
