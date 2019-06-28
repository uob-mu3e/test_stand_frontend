-- Basic constants for DAQ communication
-- K. Briggl, April 2019 : stripped from mupix8_daq repository / mupix_constants.vhd

library ieee;
use ieee.std_logic_1164.all;

package daq_constants is
subtype links_reg32 is std_logic_vector(31 downto 0);
subtype reg32 is std_logic_vector(31 downto 0);
type reg64b_array_t is array (natural range <>) of std_logic_vector(63 downto 0);

constant TIME_125MHz_1s 	: STD_LOGIC_VECTOR(27 DOWNTO 0) := x"7735940";
constant TIME_125MHz_1ms 	: STD_LOGIC_VECTOR(27 DOWNTO 0) := x"001E848";
constant TIME_125MHz_2s		: STD_LOGIC_VECTOR(27 DOWNTO 0) := x"EE6B280";
constant HUNDRED_MILLION: 		std_logic_vector(27 downto 0) := x"5F5E100"; 
constant HUNDRED_MILLION32:	std_logic_vector(31 downto 0) := x"05F5E100";


-- 8bit/10bit encoding
constant k28_0	:	std_logic_vector(7 downto 0)	:= X"1C";	-- used in MuPix
constant k28_1	:	std_logic_vector(7 downto 0)	:= X"3C";	-- used in data alignment (transceiver)
constant k28_2 : 	std_logic_vector(7 downto 0)	:= X"5C";
constant k28_3 : 	std_logic_vector(7 downto 0)	:= X"7C";
constant k28_4 : 	std_logic_vector(7 downto 0)	:= X"9C";
constant k28_5	: 	std_logic_vector(7 downto 0)	:= X"BC";	-- used in MuPix
constant k28_6 : 	std_logic_vector(7 downto 0)	:= X"DC";
constant k28_7 : 	std_logic_vector(7 downto 0)	:= X"FC";	-- not used, comma symbol with harder constraints!
constant k23_7 : 	std_logic_vector(7 downto 0)	:= X"F7";	-- used as "empty" data (transceiver)
constant k27_7 : 	std_logic_vector(7 downto 0)	:= X"FB";
constant k29_7 : 	std_logic_vector(7 downto 0)	:= X"FD";
constant k30_7 : 	std_logic_vector(7 downto 0)	:= X"FE";

end package daq_constants;
