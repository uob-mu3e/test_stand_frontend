-- link observer for BERT
-- Marius Koeppel, August 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity link_observer is
  	generic (
        g_m             : integer           := 7;
        g_poly          : std_logic_vector  := "1100000" -- x^7+x^6+1 
    );
    port(
		clk:               in std_logic;
		reset_n:           in std_logic;
		rx_data:           in std_logic_vector (g_m - 1 downto 0);
		rx_datak:          in std_logic_vector (3 downto 0);
		error_counts:      out std_logic_vector (31 downto 0);
		bit_counts:        out std_logic_vector (31 downto 0);
		state_out:         out std_logic_vector(3 downto 0)--;
);
end entity link_observer;

architecture rtl of link_observer is
	
	signal error_counter : std_logic_vector(31 downto 0);
	signal bit_counter   : std_logic_vector(31 downto 0);

	signal r_lfsr : std_logic_vector (g_m downto 1);
	signal w_mask : std_logic_vector (g_m downto 1);
    signal w_poly : std_logic_vector (g_m downto 1);

begin

	error_counts 	<=	error_counter;
	bit_counts 		<=	bit_counter;


    w_poly <= g_poly;
    g_mask : for k in g_m downto 1 generate
        w_mask(k) <= w_poly(k) and r_lfsr(1);
    end generate g_mask;

	process(clk, reset_n)
	begin
		if(reset_n = '0') then
			error_counter 	<= (others => '0');
			bit_counter 	<= (others => '0');
			r_lfsr	<= (others => '0');
		elsif(rising_edge(clk)) then
			if (rx_data = x"000000BC" and rx_datak = "0001") then
	         	-- idle
	        elsif (rx_datak = "0000") then
	        	r_lfsr  <= '0' & r_lfsr(g_m downto 2) xor w_mask;
				if(rx_data = r_lfsr(g_m downto 1)) then
					bit_counter 	<= bit_counter + '1';
				else
					error_counter 	<= error_counter + '1';
				end if;
	      end if;
		end if;
	end process;


end rtl;