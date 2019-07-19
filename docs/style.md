# Code style

variables/objects/entities prefixes/sufixes:

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
- avoid synopsys packages (`std_logic_unsigned`, etc.)

file names:

- `entityname.vhd`
- `tb_entityname.vhd` or `testbench_entityname.vhd`
- `packagename_pkg.vhd`

code identation/alignment:

- tab = 4 spaces
    - *alignment* (inside code) have to be done with spaces
    - *identation* can be done with tabs, but do not mix spaces and tabs for identation
- use unix line ending (`\n`), prefer ascii char set
