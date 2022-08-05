<!-- vim:set ft=markdown: -->
## Notes/knowledge base

### Permissions

It's probably best to set everything to 777. Additional permission should be
given to users accessing the Arduino device file (location `/dev/ttyACM1`).
Important point, otherwise test stand stuff won't work.

Root permission also needed to
load the mudaq, this file should be executed once (???) and is found in
`./common/kerneldriver/load_mudaq.sh`


### Pull and update from latest branch `online`

```bash
git pull bitbucket online
```

### Compiling first time

```bash
source midas_env.sh
mkdir build && cd build
cmake3 ..
make -j12
```

All compiled binaries are stored in the `build` directory, within their
respective subdirectories. All shell scripts stored in the top-level directory
(with `*.sh.in` extension) are copied into `build` with extensionl `*.sh` with
appropriate execute permissions.

To recompile, make sure to source `midas_env.sh` and run `make -j12` again in
the `build` directory. One can also add to their `~/.bashrc` the contents of
`midas_env.sh` to have all the required environmental variables set at log in.

### Kernel version

Tested on latest long-term kernel from elrepo `kernel-lt` and `kernel-lt-devel`.
Update/install will require `yum` with elevated permissions. Useful commands:

#### Installing latest long-term support kernel

```bash
sudo yum --enablerepo=elrepo-kernel install kernel-lt
```

Change default kernel at launch in `/etc/default/grub`. Lower number means more
recently installed kernel, so

```cfg
GRUB_DEFAULT=0
```

would mean the most recently installed kernel.

### GCC & Boost

The GCC version that comes with CentOS 7 is too old, and so is the C++
Boost library ([needs to be >= v1.66](https://github.com/uob-mu3e/test_stand_frontend/commit/de07630bb2f03efc5c5c43f933865650ec2c32f2)).

Workaround: Get the latest compiler from
[CernVM-FS](https://cernvm.cern.ch/fs/). Install it (need root
permission, but it is already installed). Then for Boost, ensure this in the top-level `CMakeLists.txt`:

```cmake
set(Boost_NO_SYSTEM_PATHS TRUE)
if (Boost_NO_SYSTEM_PATHS)
    set(BOOST_ROOT "${CVMFS_ROOT}/sft.cern.ch/lcg/releases/LCG_102/Boost/1.78.0/x86_64-centos7-gcc11-opt")
    set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
    set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
endif (Boost_NO_SYSTEM_PATHS)
include_directories(${BOOST_INCLUDE_DIRS})
```

For GCC, place this in `~/.bashrc` (these lines don't go into `set_env.sh` because they are our machine-specific things),

```bash
export CVMFS_ROOT="/cvmfs"
source ${CVMFS_ROOT}/sft.cern.ch/lcg/releases/gcc/11.1.0/x86_64-centos7/setup.sh
export GCC_HOME="${CVMFS_ROOT}/sft.cern.ch/lcg/releases/LCG_102/gcc/11/x86_64-centos7"
```

where `CVMFS_ROOT` is the root directory for cvmfs. Ensure that the right
architecture is selected, and the Boost version is compatible with the GCC and
LCG version. The current paths should work.

### Running the software

```bash
cd build
./start_daq.sh
```

There should be many `xterm` windows appearing, each monitoring a different
process. To see the web application, navigate to `localhost:8008` in a browser
running on the same machine.

### SSH tunnelling

To access the web interface through a local web browser (if accessing the Mu3e
machine remotely) using an SSH tunnel, run `ssh` with the following arguments:

```bash
ssh -XYC -L 8007:localhost:8008 user@remote
```

- Then navigate to a local web browser and visit `localhost:8007`
- The port `8007` can be subsituted with any free port. 

### PSU

When running the software, expect Channel 1 of the PSU to switch on (this
controls the fan), the current limit set to 3 A, and the voltage to 12 V. It
should be in constant voltage (CV) mode, where the indicator light on the panel
is green. If it is red, it is in constant current (CC) mode.

In theory, this shouldn't happen, the current limit is set first then constant
voltage is set after. But in the case it goes into CC mode, execute the
following in a shell:

```bash
cat > /dev/ttyACM1
v12
```

Then Ctrl-C to escape. This should force the voltage to 12 V and put the PSU in
CV mode.

The manual for the power supply (HMP4040) is [linked](https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_common_library/dl_manuals/gb_1/h/hmp_serie/HMPSeries_UserManual_en_02.pdf).

### Test stand variables

In the MIDAS ODB, under `Equipments/ArduinoTestStation/Variables/`, there are three
variables that send commands to the Arduino. These are read periodic so if
edited, the changes will be reflected in the next loop.

- `_S_`: temperature setpoint
- `_V_`: voltage constant/limit
- `_C_`: current constant/limit

### Python analyser

Slow control variables can be plotted using `./online/myAnalyser.py`


### Setting up A10 & FEB

Main instruction found on [bitbucket](https://bitbucket.org/mu3e/online/wiki/FEB%20to%20A10%20Dev%20Board%20Lab%20Setup). Here's a quick version:

#### Connect hardware

- Signal generator goes into CON4 and CON5 on the FEB (golden/round, top right), each signal running 125 MHz but one needs to be inverse of the other.
- JTAG connection goes into CON7.
- Firefly optical cable goes into the port right above CON7.
- TODO: Clock signal should go into A10 as well.
- TODO: Change optical link (4 different ports) (4 -> 1, or 1 -> 4),

#### Compiling FPGA firmware

- Navigate to `./fe_board/fe_v2_mupix` and run `make flow` (this can take a while).
- Caveat: for some reason, our custom CVMFS `gcc` environment takes precedence over the sourced `nios2-elf-gcc` binary, even with the correct PATH hierachy set up. (Workaround: remove the sourcing of `gcc` from CVMFS in `~/.bashrc` temporarily)
    - More detail: Go into `~/.bashrc` and comment out these lines so that it looks like the section below, and then log back in and out or open a new shell, and source only `./build/set_env.sh`. After compilation is successful, uncomment those lines.

```bash
# export CVMFS_ROOT="/cvmfs"
# source ${CVMFS_ROOT}/sft.cern.ch/lcg/releases/gcc/11.1.0/x86_64-centos7/setup.sh
# export GCC_HOME="${CVMFS_ROOT}/sft.cern.ch/lcg/releases/LCG_102/gcc/11/x86_64-centos7"
```

- Now, run `make app`, if there are more errors to do with C++11 std, then proceed to the following workaround:
    - Navigate to `./generated/software/hal_bsp/public.mk`.
    - Add in `ALT_CFLAGS += -std=c++11` on a newline anywhere in the file, save and close.
    - Recompile. The `.hex` file should be built and ready to be uploaded onto the board.

#### JTAG chain

The JTAG chain is difficult to set up. The correct order of operation is to

1. Connect the USB Blaster into the board (and the PC)
2. Power the FEB (15V 0.8A).
3. Run `jtagconfig` and check for detection.

If the error "...broken device chain" comes up, repeat 1 & 3 cyclicly until `jtagconfig` shows something like this

```bash
1) USB-Blaster [3-3]
  02A020DD   5AGT(FC7H3|MC7G3)/5AGXBA7D4/..
```

#### Uploading

- Run `make CABLE=1 plm`
- Run `make CABLE=1 app_upload`
- No issue reported with these commands so far

## WIP & TODOs
- [x] Get JTAG connection working at least once
- [x] Compile FPGA and upload onto A5
- [ ] Get reading from A10 (switch-fe) should produce values
- [x] Custom MIDAS page for teststand
- [x] Arduino features (working decently for now), the drop in shell with `cat > /dev/ttyACM1` too slow/finicky. 
- [ ] Just an idea: have a shell script to perform the Arduino commands (which takes in flags/commands/values). Maybe that's not neccessary, but it's less annoying than opening a second tty just to pipe commands into the device file.
- [ ] Toggle on/off the power output within frontend
- [ ] Change PSU channel within frontend
- [ ] Ability to set current limit without leaving CV mode
- [ ] Plot voltage and current using `myAnalyser.py`
- [ ] Documentation on git usage and whatnot
