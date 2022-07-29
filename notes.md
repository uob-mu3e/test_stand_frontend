<!-- vim:set ft=markdown: -->
## Notes/knowledge base

### Permissions

It's probably best to set everything to 777. Additional permission should be
given to users accessing the Arduino device file (location `/dev/ttyACM0`).
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
cat > /dev/ttyACM0
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
