-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Wrapper for open core 8bit/10 bit
-- Niklaus Berger, Feb 2014
-- 
-- nberger@physi.uni-heidelberg.de
--
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;




entity decode8b10b is 
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		input					: IN STD_LOGIC_VECTOR (9 DOWNTO 0);
		output				: OUT STD_LOGIC_VECTOR (7 DOWNTO 0);
		k						: OUT std_logic
		);
end decode8b10b;

architecture RTL of decode8b10b is

signal reset : std_logic;

begin

reset <= not reset_n;

oc8b10b : entity work.dec_8b10b 	
    port map(
		RESET 		=> reset,
		RBYTECLK 	=> clk,
		AI				=> input(0), 
		BI				=> input(1), 
		CI				=> input(2), 
		DI				=> input(3), 
		EI				=> input(4), 
		II				=> input(5), 
		FI				=> input(6), 
		GI				=> input(7), 
		HI				=> input(8), 
		JI 			=> input(9),		
		KO 			=> k,
		HO				=> output(7), 
		GO				=> output(6), 
		FO				=> output(5), 
		EO				=> output(4), 
		DO				=> output(3), 
		CO				=> output(2), 
		BO				=> output(1), 
		AO 			=> output(0)
	    );



end RTL;
