Switching Board Firmware
======================


Installation
----------------------
1. go to the nios folder and run 

```
$ qsys-script --script=nios.tcl
```
2. go into quartus and generate the nios files form your new created nios.qsys file
3. copy the nios.sopcinfo file from the nios folder into the top folder
4. run
```
$ make flow
```

