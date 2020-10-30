-- Types for use in mupix telescope

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;

package mupix_types is

subtype reg32 		is std_logic_vector(31 downto 0);
--type reg32array	is array (64-1 downto 0) of reg32;
type reg32array_128	is array (128-1 downto 0) of reg32;

subtype reg64 		is std_logic_vector(63 downto 0);
subtype REG64_TOP_RANGE is integer range 63 downto 32;
subtype REG64_BOTTOM_RANGE is integer range 31 downto 0;
subtype reg62 		is std_logic_vector(61 downto 0);
subtype REG62_TOP_RANGE is integer range 61 downto 31;
subtype REG62_BOTTOM_RANGE is integer range 30 downto 0;

type reg64array 	is array (NCHIPS-1 downto 0) of reg64;
type reg48array 	is array (NCHIPS-1 downto 0) of std_logic_vector(47 downto 0);

type hitsorter_debug_array	is array (7 downto 0) of reg32;

subtype readmemaddrtype 		is std_logic_vector(15 downto 0);

subtype chipmarkertype 			is std_logic_vector(7 downto 0);-- for pseudo data generator
type NumCOL_array is array (2 downto 0) of integer range 0 to 200;
type MatrixSEL_array is array (2 downto 0) of std_logic_vector(1 downto 0);

end package mupix_types;