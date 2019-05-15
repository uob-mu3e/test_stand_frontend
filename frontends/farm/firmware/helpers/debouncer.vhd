----------------------------------------------------------------------------
-- debouncer for push buttons
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

---use work.mupix_components.all;

entity debouncer is
	port(
		clk:			in std_logic;
		din:			in std_logic;
		dout:			out std_logic
		);		
end entity debouncer;



architecture rtl of debouncer is	

signal insample: std_logic_vector(4 downto 0);
signal counter: std_logic_vector(11 downto 0);
signal dout_last: std_logic;

begin

process(clk)

begin
if(clk'event and clk = '1') then
	counter <= counter + '1';
	if(counter = "000000000000") then
		insample <= din & insample(4 downto 1);
	end if;
	if(insample = "00000") then
		dout <= '0';
		dout_last <= '0';
	elsif(insample = "11111") then
		dout <= '1';
		dout_last <= '1';
	else
		dout <= dout_last;
	end if;
end if;
end process;

end rtl;