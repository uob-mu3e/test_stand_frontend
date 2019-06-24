Labor setup for Arria10 and FEB Communication
===============================================

this branch contains the firmware for the FEB and the Arria10 development board for sending slow control from the
Arria10 board to the FEB in a simple labor setup. One can also send data from the FEB to the Arria10 board and read it
out via PCIe and DMA.

## Compiling the firmware and setting up NIOS for the FEB
```console
$ cd fe_board/fe_malibu
$ make flow
$ make pgm
$ make app_upload
$ make terminal
```
#### NIOS FEB

## Compiling the firmware and setting up NIOS for the Arria10
```console
$ cd switching_pc/a10_board/firmware
$ make nios.sopcinfo
$ make ip/ip_xcvr_fpll.sopcinfo
$ make ip/ip_xcvr_reset.sopcinfo
$ make ip/ip_xcvr_phy.sopcinfo
$ make flow
$ make pgm
$ make app_upload
$ make terminal
```

#### NIOS Arria10
The NIOS terminal will look like this. For setting up the transceivers one need to reset the pll first. Then one can monitor the input via "xcvr qsfp" with 0,1,2,3 one can change the channels. The software for this is located in software/app_src.
```console
'Arria 10' NIOS Menu
  [0] => spi si chip
  [1] => i2c fan
  [2] => flash
  [3] => xcvr qsfp
  [r] => reset pll
Select entry ...
```

## Compiling and starting MIDAS
```console
$ cd modules
$ git submodule init
$ git submodule update
$ cd midas
$ git pull origin develop
$ git submodule update --init
$ cd ../../
$ mdkir build
$ cd build
$ cmake ..
$ make
$ make install
$ source set_env.sh
$ start_daq.sh
```
If you do it the first time you also need to generate a SSL certificate and a password. This you need to do after source set_env.sh.
```console
$ openssl req -new -nodes -newkey rsa:2048 -sha256 -out ssl_cert.csr -keyout ssl_cert.key
$ openssl x509 -req -days 365 -sha256 -in ssl_cert.csr -signkey ssl_cert.key -out ssl_cert.pem
$ cat ssl_cert.key >> ssl_cert.pem
$ touch htpasswd.txt
$ htdigest htpasswd.txt Default midas
```
Now you need to load / make the switching board driver. You also need to remove and rescan your pci devices if you uploaded the firmware to the arria10 after starting up the pc. Be careful here that you make sure no program is using the arria10. This can lead to problems if you want to load the driver again. You also need to check which device number is the arria10. You can do this by calling sudo lspci.
```console
$ cd ../common/kerneldriver
$ make
$ su
$ echo 1 > /sys/bus/pci/devices/0000:01:00.0/remove
$ echo 1 > /sys/bus/pci/rescan
$ ./load_mudaq.sh
```
Now you can start the switch_fe and the Switching equipment should show up on your midas status page.


![Switching Page] https://bitbucket.org/mu3e/online/src/lab_setup_arria10/lab_setup.pdf