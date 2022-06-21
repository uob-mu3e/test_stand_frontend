#!/bin/sh

LOCATION="Equipment/ArduinoTestStation/Variables"
setpointExist=$(odbedit -c "ls $LOCATION" | grep "_S_")

if [ -z "$setpointExist" ]
then
    echo "Setpoint doesn't exist, creating setpoint, default value 30."
    odbedit -c "create FLOAT $LOCATION/_S_" > /dev/null
    odbedit -c "set $LOCATION/_S_ 30" > /dev/null
fi
