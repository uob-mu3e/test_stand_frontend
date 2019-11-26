# Quartus

## Setting up Quartus

- install [Quartus Prime Standard Edition](http://fpgasoftware.intel.com/?edition=standard)
- modify (set `QUARTUS_ROOTDIR` and `ALTERAD_LICENSE_FILE`) and source env.file `common/firmware/util/quartus.sh`
- common problems:
    - udev version - _TODO_
    - jtagd service - _TODO_

## Project structure

- `top.qpf`
- `top.qsf` - main project file
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
cd "$project_dir"
make flow
# program fpga
make pgm
# compile nios software
make app
# upload nios software
make app_upload
# connect to nios terminal through jtag
make terminal
```

### Arria 10

NOTE: in most cases `make flow` will also compile ip components (`qsys` and `sopcinfo` files).

```console
cd "$project_dir"
make
make flow
make pgm
make app_upload
make terminal
```
