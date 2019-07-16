# Mu3e online repository

## git howto

- do development on your branch
- make pull request to latest dev branch (v0.1_dev)
- merge dev branch to your branch as often as possible

## documentation

- put docs to appropriate folders
- link to local and external docs from "links" section below

## Code style

prefixes/sufixes:

- `e_` / `_id` - entity/component instance
- `g_` - generate statement
- `_v` - variable
- `_t` - type
- UPPERCASE `_g` - generic
- UPPERCASE `_c` - constant
- `i/o_` - input/output ports
- `_n` - active low

...

- ports : `std_logic` and `std_logic_vector`
- use `downto`
- avoid `std_logic_unsigned`, etc.

file names:

- `entityname.vhd`
- `tb_entityname.vhd` or `testbench_entityname.vhd`
- `packagename_pkg.vhd`

spaces:

- tab = 4 spaces
- ascii

## Links (docs, etc.)

- [quartus project](docs/quartus.md)
- [nios software](docs/nios.md)
- [compiling and starting midas](docs/midas.md)
- [setup #1](docs/setup1.md)

- [Transceiver Architecture in Stratix IV Devices](https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/stratix-iv/stx4_siv52001.pdf)
