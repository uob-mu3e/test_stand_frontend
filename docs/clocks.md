# Clocks

The global clock is 125 MHz and is distributed from clock and reset board.

## FEB (frontend board, Stratix4 prototype board)

Two Si chips:
- AUX input drives nios (TODO: use 5342)
- 5345 generates 156.25 for data path, global 125 (zero delay) and lvds clocks

156.25 MHz clock:
- QSFP
- fe_block
- data path for malibu, scifi and mupix

125 MHz global clock:
- timestamps
- reset is synchronized from POD reciver
  with phase measurement relative to recovered clock

nios:
- TODO: external memory interfaces use either 156.25 or 125 clocks

## SB (switching board, Arria10 dev.board)

156.25 MHz clock:
- QSFP from FEB
- data path
- PCIe registers
- read/write side of PCIe dpRAM

125 MHz global clock:
- QSFP to FPC

250 MHz PCIe clock:
- write/read side of PCIe dpRAM

## FPC (farm pc, Arria10 dev.board)

125 MHz global clock:
- data path
