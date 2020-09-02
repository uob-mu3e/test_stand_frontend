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
	
end package dataflow_components;
