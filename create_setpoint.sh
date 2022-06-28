#!/bin/sh

DEFAULT_TEMP=25    # Edit this variable to set default setpoint temperature in deg Celsius

LOCATION="Equipment/ArduinoTestStation/Variables"
setpointExist=$(odbedit -c "ls $LOCATION" | grep "_S_")

if [ -z "$setpointExist" ]
then
    echo "Setpoint doesn't exist, creating setpoint, default value $DEFAULT_TEMP."
    odbedit -c "create FLOAT $LOCATION/_S_" > /dev/null
fi

odbedit -c "set $LOCATION/_S_ $DEFAULT_TEMP" > /dev/null
