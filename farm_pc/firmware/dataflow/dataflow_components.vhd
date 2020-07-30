library ieee;
use ieee.std_logic_1164.all;

package dataflow_components is

	subtype tsrange_type is std_logic_vector(15 downto 0);
	subtype tsupper is natural range 31 downto 16;
	subtype tslower is natural range 15 downto 0;
	
	constant tsone : tsrange_type := (others => '1');
	constant tszero : tsrange_type := (others => '1');

	subtype dataplusts_type is std_logic_vector(271 downto 0);
	
end package dataflow_components;