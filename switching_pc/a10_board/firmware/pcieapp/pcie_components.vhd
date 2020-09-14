library ieee;
use ieee.std_logic_1164.all;

package pcie_components is

subtype reg32           is std_logic_vector(31 downto 0);
type reg32array is array (63 downto 0) of reg32;

end package;
