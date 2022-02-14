#!/bin/sh
#set -euf

SHA=$(git rev-parse HEAD | cut -c 1-8)
echo "GIT_SHA: $SHA"

if [ ! -e "output_files/top_$SHA.sof" ] ; then
#    exit 1
    echo "copy top_$SHA.sof"
    cp "output_files/top.sof" "output_files/top_$SHA.sof"
fi

echo "pcie: remove"
echo 1 | sudo tee "/sys/bus/pci/devices/0000:02:00.0/remove"
sleep 1
make SOF="output_files/top_$SHA.sof" pgm
sleep 1
echo "pcie: rescan"
echo 1 | sudo tee "/sys/bus/pci/rescan"
sleep 1

#../../cmake-build/farm_pc/tools/swb_dmatest 3 0 0 0x1 1
../../cmake-build/farm_pc/tools/swb_dmatest 2 0 0 0x1 0
head --lines=1000 memory_content.txt > "swb_dmatest.out"
