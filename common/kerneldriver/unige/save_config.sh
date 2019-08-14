#!/bin/sh

device=$(lspci -nn | grep "\[1172:0004\]" | cut -d" " -f1)
file=~/.arria10_pcie_config


if [[ ! -z $device  ]]
then
	echo Using PCIe Device @$device 
else
	echo "Did not find an altera pcie device... "
	exit
fi

echo /sys/bus/pci/devices/0000\:$device/config $file
echo "Stor configuration to file $file"
