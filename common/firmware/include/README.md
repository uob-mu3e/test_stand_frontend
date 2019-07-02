# NIOS software

 - `i2c.h` - I2C controller
 - `si.h` - SI chip controller

## `xcvr` menu

```
#    +--------------------- device
#    |        +------------ channel
#    |        |          +- loopback
#    |        |          |
xcvr[A].ch[0x00], lpbk = 1
#                 +-- digital reset
#                 |+- analog reset
#                 ||
#                 ||   +---- lock status
#                 ||   |+--- sync status
#                 ||   || +- ref.lock status
#                 ||   || |
#                 ||   || |    +--- fifo overflow
#                 ||   || |    |+-- 8b10 disparity error
#                 ||   || |    ||+- 8b10 error
#                 ||   || |    |||
                R_DA S_LS_R E__FDE
  tx    :   OK  0x00 0x0001 0x0000
  rx    :   OK  0x00 0x1F07 0x0000
#                     +- loss of lock counter
#                     |
#                     |            +- 8b10b error counter
#                     |            |
        :   LoL_cnt = 1, err_cnt = 65535
#                    +- data
#                    |     +- control symbols
#                    |     |
  data  :   0x000000BC / 0x1
```

commands:

 - `r` - reset
 - `l` - loopback
 - `q` - exit
