-- simple counter for pcie test
-- counter_test.vhd

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_unsigned.all;
use IEEE.numeric_std.all;

entity counter is
	Port(
		clk 				: 	in std_logic;
		reset_n			:	in std_logic;
		enable			:	in std_logic;
		data_en			:	out std_logic;
		time_counter 	: 	out std_logic_vector (255 downto 0)
		);
end counter;

architecture arch of counter is

	signal timer 		: 	std_logic_vector (255 downto 0);
	
begin

process(clk,reset_n)
begin
if(reset_n = '0') then
	data_en 			<= '0';
	timer				<= (others => '0');
	time_counter   <= (others => '0');
-- clk'event and clk = '1' 
-- (The condition above will be true only on rising edge of the CLK signal,
-- i.e. when the actual value of the signal is '1' and there was an event on it (the value changed recently))
-- is it equal to rising_edge(clk)
elsif(clk'event and clk = '1') then
	if(enable = '0') then
		data_en 			<= '0';
		timer				<= (others => '0');
		time_counter   <= (others => '0');
	else
		data_en 		<= '1';
		timer 		<= timer+'1';
		time_counter 	<= timer;
	end if;
end if;
end process;

end arch;		
