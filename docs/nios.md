# NIOS software

### Arria10
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
