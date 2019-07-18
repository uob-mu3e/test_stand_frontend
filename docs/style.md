# Code style

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
