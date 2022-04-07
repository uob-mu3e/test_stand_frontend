#!/bin/bash

STOP_TIME_US=1 \
../sim.sh \
    fifo_sync_tb \
    *.vhd \
    ../*.vhd
