-- link observer for BERT
-- Marius Koeppel, August 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.daq_constants.all;

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
		mem_add:           out std_logic_vector (2 downto 0);
		mem_data:          out std_logic_vector (31 downto 0);
		mem_wen:				 out std_logic--;
);
end entity link_observer;

architecture rtl of link_observer is

	signal error_counter : std_logic_vector(63 downto 0);
	signal bit_counter   : std_logic_vector(63 downto 0);

	signal tmp_rx_data   : std_logic_vector(g_m - 1 downto 0);
	signal next_rx_data  : std_logic_vector(g_m - 1 downto 0);
	signal enable		 : std_logic;
	signal sync_reset	 : std_logic;
	
	type state_type is (err_low, err_high, bit_low, bit_high);
	signal state : state_type;

begin


	process(clk, reset_n)
	begin
		if(reset_n = '0') then
         mem_add    <= (others => '0');
         mem_data   <= (others => '0');
			mem_wen	<= '0';
         state      <= err_low;
		elsif(rising_edge(clk)) then
        case state is
            
            when err_low =>
                mem_add     <= "001";
                mem_data    <= error_counter(31 downto 0);
					 mem_wen		<= '1';
                state      <= err_high;
            when err_high =>
                mem_add     <= "010";
                mem_data    <= error_counter(63 downto 32);
					 mem_wen		<= '1';
                state      <= bit_low;
            when bit_low =>
                mem_add     <= "011";
                mem_data    <= bit_counter(31 downto 0);
					 mem_wen		<= '1';
                state      <= bit_high;
            when bit_high =>
                mem_add     <= "100";
                mem_data    <= bit_counter(63 downto 32);
					 mem_wen		<= '1';
                state      <= err_low;
            when others =>
                mem_add    <= (others => '0');
                mem_data   <= (others => '0');
					 mem_wen	<= '0';
                state       <= err_low;
        end case;
		end if;
	end process;

	e_linear_shift_link : entity work.linear_shift_link
	generic map(
		g_m 	=> g_m,
		g_poly 	=> g_poly
	)
    port map (
		i_clk 			=> clk,
		reset_n 			=> reset_n,
		i_sync_reset 	=> sync_reset,
		i_seed			=> rx_data,
		i_en 				=> enable,
		o_lsfr			=> next_rx_data,
		o_datak 			=> open--,
    );

	process(clk, reset_n)
	begin
		if(reset_n = '0') then
			error_counter 	<= (others => '0');
			bit_counter 	<= (others => '0');
			tmp_rx_data		<=  x"000000BC";
			enable			<= '0';
			sync_reset 		<= '1';
		elsif(rising_edge(clk)) then
			tmp_rx_data		<= rx_data;
			if (rx_data = x"000000BC" and rx_datak = "0001") then
	         -- idle
	         enable			<= '0';
	         sync_reset 		<= '1';
	      elsif (rx_datak = "0000") then
				enable			<= '1';
	        	sync_reset     <= '0';
	        	bit_counter 	<= bit_counter + '1';
				if(tmp_rx_data = next_rx_data) then
					-- no error
				else
					error_counter 	<= error_counter + '1';
				end if;
	      end if;
		end if;
	end process;


end rtl;
