#!/bin/sh
# try to recover stuck fpga board (e.g. after configuration) - copy pcie config header from working state
# to set up the required file, load the driver after powering up the pc, then do
# cp /sys/bus/pci/devices/0000\:01\:00.0/config config_reboot_dload
 
#../online/kill_daq.sh
#rmmod mudaq
#echo 1 > /sys/bus/pci/devices/0000\:01\:00.0/remove
#echo 1 > /sys/bus/pci/rescan 
device=$(lspci -nn | grep "\[1172:0004\]" | cut -d" " -f1)
file=~/.arria10_pcie_config

if [[ -f $file ]]
then
	echo "Using configuration file $file"
else
	echo "Configuration file $file does not exist, run save_config.sh just after reboot to generate it!"
	exit
fi

if [[ ! -z $device  ]]
then
	echo Using PCIe Device @$device 
else
	echo "Did not find an altera pcie device... "
	exit
fi
if [[ ! -f /sys/bus/pci/devices/0000\:$device/config ]]
then
	echo "PCIE configuration device file $file does not exist, something is wrong"
	exit
fi


echo cp $file /sys/bus/pci/devices/0000\:$device/config
sudo cp $file /sys/bus/pci/devices/0000\:$device/config
sudo ./load_mudaq.sh
