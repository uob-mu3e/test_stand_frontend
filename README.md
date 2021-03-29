# Mu3e online repository

- current development branch is `v0.11_dev`
- Documentation and manual are under construction in the wiki associated to this repo

## Current "fixes" with MIDAS tag ced039e, cuda 10 or cuda 8

- cmake include funtion in examples/experiment/CMakeLists.txt:99 -> comment out
- in progs/msysmon.cxx not all cases are in all cuda / nvidia versions -> comment out 
- do cmake .. -DCUDA_HOST_COMPILER='/usr/bin/gcc-7' to link a different compiler to nvcc
- if cuda 8 and glibc 2.26: vim floatn.h --> define __HAVE_FLOAT128 0
- install cuda under tumbleweed https://www.tobiasbartsch.com/installing-cuda-and-cudnn-on-opensuse-tumbleweed/

## Current "fixes" with ROOT not compiled with c++-XX

- online/CMakeLists.txt -> replace CXX_STANDARD XX
- online/modules/midas/CMakeLists.txt -> replace CXX_STANDARD xx
- online/modules/mutrigana-base/CMakeLists.txt -> replace CXX_STANDARD xx
- online/modules/mutrigana-base/calibration/CMakeLists.txt -> replace CXX_STANDARD xx
- online/modules/mutrigana-base/online/CMakeLists.txt -> replace CXX_STANDARD xx

## Raspberry Pi USB Server

- Follow the setup in https://wiki.ubuntuusers.de/USBIP/

## Structure

- `backend_pc` - software for the backend pc, plus timing and reset system firmware
- `common` - common firmware and software
- `farm_px` - _TODO_
- `fe_board` - front-end board firmware (si, fb and tl readout)
- `frontends` - _TODO_
- `switching_pc` - _TODO_

## documentation

- read docmentation (see "links" sections below)
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

- [Transceiver Architecture in Stratix IV Devices](https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/stratix-iv/stx4_siv52001.pdf)
