# Code style

## VHDL

variables/objects/entities prefixes/sufixes:

- `e_` / `_id` - entity/component instance
- `generate_` - generate statement
- `_v` - variable
- `_t` - type
- `_g` UPPERCASE - generic
- UPPERCASE `_c` - constant
- `i_`, `o_` and `io_` - input, output and input/output ports
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

misc:

- Note that most vhdl language blocks do not require name
  and can be finalized with, e.g. `end process;`.
  The only block that requre name are `generate` and `block` blocks.
- Avoid collisions between names of global objects,
  e.g. same constant and port names, etc.

### Examples

package `example_pkg.vhd`:
```
library ieee;
use ieee.std_logic_1164.all;

package example is

    subtype slv4_t is std_logic_vector(3 downto 0);
    type slv4_array_t is array ( natural range <> ) of slv4_t;

    function max (
        l, r : integer--;
    ) return integer;

-- use `end package;`
end package;

package body util is

    function max (
        l, r : integer
    ) return integer is
    begin
        if ( l > r ) then
            return l;
        else
            return r;
        end if;
    -- use `end function;`
    end function;

-- use `end package body;`
end package body;
```

entity `example.vhd`:
```
library ieee;
use ieee.std_logic_1164.all;

entity example is
generic (
    -- `g_` prefix for generic parameter
    g_W : positive := 8--;
);
port (
    -- `o_` prefix for output port
    o_data      : out   std_logic_vector(g_W-1 downto 0);
    -- `i_' prefix for input port, `_n` suffix for active low signal
    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
-- use `end entity`
end entity;

architecture arch of example is

    signal data : std_logic_vector(o_data'range);

begin

    -- `generate_` prefix for generate block
    generate_data : for i in data'range generate
    begin
        o_data(i) <= data(i);
    -- use `end generate;`
    end generate;

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        data <= (others => '0');
    elsif rising_edge(i_clk) then
        data <= (others => '1');
    -- use `end if;`
    end if;
    -- use `end process;`
    end process;

-- use `end architecture;`
end architecture;
```



## C++

Use <https://google.github.io/styleguide/cppguide.html>
as a baseline for C++ code style.

tools:

- C++17
- cmake

file names:

- `name.h`, `name.cpp`
- `name_test.cpp` or `test/name.cpp` - test code
- `name.inc` - use as *direct* include in cpp files

code identation/alignment:

- tab = 4 spaces
    - use spaces for both *alignment* and *identation*
    - be consistent
- use unix line ending (`\n`) and prefer ascii char set
- do not leave trailing spaces

misc:

- try to use namespaces to group common functionality
- avoid excesive use of getters/setter (make the variable public or use struct)
- avoid use of global variables

### Examples

File `dir/foo.h` in mu3e/online repository:
```
#ifndef MU3E_ONLINE_DIR_FOO_H_
#define MU3E_ONLINE_DIR_FOO_H_

namespace mu3e { namespace online {

inline
int bar1(int a, int b) {
    return a + b;
}

int bar2(int a, int b);

struct foo_t {
};

class Foo {
private:
    int foo_var;

public;
    Foo() {
    }
};

} } // namespace mu3e::online

#endif // MU3E_ONLINE_DIR_FOO_H_
```

File `dir/foo.cpp`:
```
#include "foo.h"

// local includes
#include "bar.h"

// C++ headers
#include <vector>

// C headers
#include <unistd.h>

namespace mu3e { namespace online {

static
int bar3(int a, int b) {
//    return a * b;
    return a - b;
}

int bar2(int a, int b) {
    // comment
    return bar3(a, b);
}

} } // namespace mu3e::online
```
