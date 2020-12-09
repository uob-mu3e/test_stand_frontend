library ieee;
use ieee.std_logic_1164.all;

package dataflow_components is

	subtype tsrange_type is std_logic_vector(15 downto 0);
	subtype tsupper is natural range 15 downto 8;-- 31 downto 16;
	subtype tslower is natural range 7 downto 0; --15 downto 0;
	
	constant tsone : tsrange_type := (others => '1');
	constant tszero : tsrange_type := (others => '1');

	subtype dataplusts_type is std_logic_vector(271 downto 0);
	
	type data_array is array (natural range <>) of std_logic_vector(37 downto 0);
	type hit_array_t is array (7 downto 0) of std_logic_vector(31 downto 0);

	type fifo_array_2 is array(natural range <>) of std_logic_vector(1 downto 0);
	type fifo_array_4 is array(natural range <>) of std_logic_vector(3 downto 0);
	type fifo_array_32 is array(natural range <>) of std_logic_vector(31 downto 0);
    type fifo_array_64 is array(natural range <>) of std_logic_vector(63 downto 0);
    type fifo_array_66 is array(natural range <>) of std_logic_vector(65 downto 0);
	
end package dataflow_components;
