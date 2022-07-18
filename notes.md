## Commands to keep in mind
- Pull and update from latest branch `online`
```bash
git pull bitbucket online
```

- Compiling first time
```bash
source set_env.sh.in
mkdir build && cd build
cmake3 ..
make -j12
```
    - To recompile, run `make -j12` again in the `build` directory

- Kernel version:
    - Tested on latest long-term kernel from elrepo `kernel-lt` and `kernel-lt-devel`

- Running the software
```bash
cd `build`
./start_daq.sh
``` 

- To access the web interface through local web browser (if accessing the Mu3e machine remotely) using an SSH tunnel if, run `ssh` with the following arguments:
```bash
ssh -XYC -L 8007:localhost:8008 user@remote
```
    - Then navigate to a locally installed web browser and visit `localhost:8007`
    - The port `8007` can be subsituted with any free port.
