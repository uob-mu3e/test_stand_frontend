----------------------------------------------------------------------------
-- Reset logic
--
-- Niklaus Berger, Heidelberg University
-- nberger@physi.uni-heidelberg.de
--
-- 
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mudaq_registers.all;

entity reset_logic is
	port(
		clk:						in 			std_logic;
		rst_n:					in				std_logic;
		
		reset_register:		in				std_logic_vector(31 downto 0); 
		--reset_reg_written:   in				std_logic;
		
		resets:					out			std_logic_vector(31 downto 0);
		resets_n:				out			std_logic_vector(31 downto 0)																		
		);		
end entity reset_logic;



architecture rtl of reset_logic is

signal resets_reg:    std_logic_vector(31 downto 0); 
signal resets_del0 : std_logic_vector(31 downto 0); 
signal resets_del1 : std_logic_vector(31 downto 0); 
signal resets_del2 : std_logic_vector(31 downto 0); 
signal resets_del3 : std_logic_vector(31 downto 0); 
signal resets_del4 : std_logic_vector(31 downto 0); 

begin


resets	<= resets_reg;
resets_n <= not resets_reg;

process(clk, rst_n)
begin
if(rst_n = '0') then
	resets_reg 		<= (others => '1');
	resets_del0		<= (others => '0');
	resets_del1		<= (others => '0');
	resets_del2		<= (others => '0');
	resets_del3		<= (others => '0');
	resets_del4		<= (others => '0');
	
elsif(clk'event and clk = '1') then

	resets_reg <= resets_del0 or resets_del1 or resets_del2 or resets_del3 or resets_del4;
	
	resets_del0		<= (others => '0');
	resets_del1 <= resets_del0;
	resets_del2 <= resets_del1;
	resets_del3 <= resets_del2;
	resets_del4 <= resets_del3;
	
	
	
	--if(reset_reg_written <= '1' and reset_register(RESET_BIT_ALL) = '1') then
	if(reset_register(RESET_BIT_ALL) = '1') then
		resets_del0 <= (others => '1');
	end if;
	
	for i in 31 downto 0 loop
		--if(reset_reg_written <= '1' and reset_register(i) = '1') then
		if(reset_register(i) = '1') then
			resets_del0(i) <= '1';
		end if;
	end loop;
	
end if;
end process;
end rtl;