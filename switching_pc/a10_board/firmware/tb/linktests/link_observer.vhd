-- link observer for BERT
-- Marius Koeppel, August 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity link_observer is
    port(
		clk:               in std_logic;
		reset_n:           in std_logic;
		rx_data:           in std_logic_vector (31 downto 0);
		rx_datak:          in std_logic_vector (3 downto 0);
		error_counts:      out std_logic_vector (31 downto 0);
		bit_counts:        out std_logic_vector (31 downto 0);
		state_out:         out std_logic_vector(3 downto 0)--;
);
end entity link_observer;

architecture rtl of link_observer is
	
	signal rx_counter    : std_logic_vector(31 downto 0);
	signal error_counter : std_logic_vector(31 downto 0);
	signal bit_counter   : std_logic_vector(31 downto 0);

begin

	error_counts 	<=	error_counter;
	bit_counts 		<=	bit_counter;

process(clk, reset_n)
begin
	if(reset_n = '0') then
		rx_counter		<= (others => '0');
		error_counter 	<= (others => '0');
		bit_counter 	<= (others => '0');
	elsif(rising_edge(clk)) then
		if (rx_data = x"000000BC" and rx_datak = "0001") then
         	-- idle
        elsif (rx_datak = "0000") then
			rx_counter <= rx_counter + '1';
			if(rx_data = rx_counter) then
				bit_counter 	<= bit_counter + '1';
			else
				error_counter 	<= error_counter + '1';
			end if;
      end if;
	end if;
end process;


end rtl;