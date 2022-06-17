#!/bin/bash

source midas.sh
CUR_DIR=$(pwd -P)
MIDAS_ROOT=$(echo $MIDASSYS | cut -d\/ -f1-7)
if [ "$CUR_DIR" != "$MIDAS_ROOT" ]; then
    echo "Please run this inside the working directory of MIDAS."
else
    mlogger -D
    mhttpd -D
    msequencer -D
    cd online/build 
    ./frontend
    cd "$CUR_DIR"
    killall mhttpd
    killall mlogger
    killall msequencer
fi
