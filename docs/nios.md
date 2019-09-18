# NIOS software

The nios software is compiled in several steps:

- compile `nios.sopcinfo`
- make Board Support Package (BSP)
- compile main `elf` file

## Arria10

The NIOS terminal has a menu like structure:

```
'Arria 10' NIOS Menu
  [1] => flash
  [2] => xcvr
Select entry ...
```

For setting up the transceivers one need to reset the pll first.
Then one can monitor the input via "xcvr qsfp".
With 0, 1, 2, 3 one can change the channels.
The software for this is located in software/app_src.
