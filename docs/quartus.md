# Quartus project

## Structure

- `top.qsf` - project file
- `top.vhd` - top entity
- `top.sdc` - constraints
- `assignments/` - link to assignments directory
- `software/` - nios software
    - `hal_bsp.tcl` - link to base "Board Support Package" file
    - `app_src/` - sources
    - `include/` - link to common software
- `util/` - link to common firmware
- `s4/` - link to Stratix IV common firmware
- `a10/` - link to Arria 10 common firmware

## Compiling the firmware and setting up NIOS

```
$ cd "$project_dir"
$ make flow
$ make pgm
$ make app_upload
$ make terminal
```

### Arria 10

```console
$ cd "$project_dir"
$ make nios.sopcinfo
$ make ip/ip_xcvr_fpll.sopcinfo
$ make ip/ip_xcvr_reset.sopcinfo
$ make ip/ip_xcvr_phy.sopcinfo
$ make flow
$ make pgm
$ make app_upload
$ make terminal
```
