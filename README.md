# Mu3e online repository

- for latest version use tag `release/22.03`
- current development branch is `dev`
- Documentation and manual are under construction in the wiki associated to this repo

## Quickstart

- `mkdir build && cd build`
- `cmake .. && make -j8 install`
- `source set_env.sh`
- start frontend `febcratefe`
- start frontend `crfe`
- start frontend `mhttpd`
- start other frontends you need

## Structure

- `backend_pc` - software for the backend pc, plus timing and reset system firmware
- `common` - common firmware and software
- `farm_pc` - _TODO_
- `fe_board` - front-end board firmware and nios-software for all subdetectors
- `frontends` - _TODO_
- `switching_pc` - Switching board firmware and nios-software (A10 devboard and LHCb board) and slow control software

## documentation

- read documentation (see "links" sections below)
- put docs to appropriate folders
- link to local and external docs from "links" section

## Links (docs, etc.)

- [quartus project](docs/quartus.md)
- [tests](docs/tests.md)
- [nios software](docs/nios.md)
- [compiling and starting midas](docs/midas.md)
- [setup #1](docs/setup1.md)
- [code style](docs/style.md)
- [git faq](docs/git.md)
- [arduino firmware repository](https://github.com/uob-hep-cad/mu3e)
