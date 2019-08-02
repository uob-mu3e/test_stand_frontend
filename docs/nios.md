# NIOS software

The nios software is compiled in several steps:

- compile `nios.sopcinfo`
- make Board Support Package (BSP)
- compile main `elf` file

## Arria10

The NIOS terminal has a menu like structure:

```
'Arria 10' NIOS Menu
  [0] => spi si chip
  [1] => i2c fan
  [2] => flash
  [3] => xcvr qsfp
  [r] => reset pll
Select entry ...
```

For setting up the transceivers one need to reset the pll first.
Then one can monitor the input via "xcvr qsfp".
With 0, 1, 2, 3 one can change the channels.
The software for this is located in software/app_src.
