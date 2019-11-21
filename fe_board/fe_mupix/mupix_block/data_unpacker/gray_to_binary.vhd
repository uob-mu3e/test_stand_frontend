-- Convert gray code to a binary code
-- Sebastian Dittmeier 
-- September 2017
-- dittmeier@physi.uni-heidelberg.de
-- based on code by Niklaus Berger
--


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity gray_to_binary is 
	generic(NBITS : integer :=10);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		gray_in				: in std_logic_vector (NBITS-1 DOWNTO 0);
		bin_out				: out std_logic_vector (NBITS-1 DOWNTO 0)
		);
end gray_to_binary;

architecture rtl of gray_to_binary is

begin

process(reset_n, clk)
	variable decoding: std_logic_vector(NBITS-1 downto 0);
begin
	if(reset_n = '0') then
		bin_out <= (others => '0');
	elsif(clk'event and clk = '1') then
		decoding(NBITS-1) := gray_in(NBITS-1);
		for i in NBITS-2 downto 0 loop
			decoding(i)	:= gray_in(i) xor decoding(i+1);
		end loop;
		bin_out 	<= decoding;
	end if;
end process;

end rtl;