library ieee;
use ieee.std_logic_1164.all;

package protocol is

    type data_merger_state is (idle, sending_data, sending_slowcontrol);

    constant HEADER_K:    std_logic_vector(31 downto 0) := x"000000bc";
    constant HEADER_K_DATAK:    std_logic_vector(3 downto 0) := "0001";
    constant WORD_ALIGN:    std_logic_vector(31 downto 0) := x"beefcafe";
    constant DATA_HEADER_ID:    std_logic_vector(5 downto 0) := "111010";
    constant DATA_SUB_HEADER_ID:    std_logic_vector(5 downto 0) := "111111";
    constant ACTIVE_SIGNAL_HEADER_ID:    std_logic_vector(5 downto 0) := "111101";
    constant RUN_TAIL_HEADER_ID:    std_logic_vector(5 downto 0) := "111110";
    constant TIMING_MEAS_HEADER_ID:    std_logic_vector(5 downto 0) := "111100";
    constant SC_HEADER_ID:    std_logic_vector(5 downto 0) := "111011";

end package protocol;

-- simple data generator (for slowcontrol and pixel data)
-- writes into pix_data_fifo and sc_data_fifo
-- only Header(sc or pix) + data
-- other headers/signals are added in data_merger.vhd

-- Martin Mueller, January 2019
-- Marius Koeppel, March 2019
-- Marius Koeppel, July 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use work.protocol.all;


entity data_generator_a10 is
    port(
		clk:                 	in  std_logic;
		reset:               	in  std_logic;
		enable_pix:          	in  std_logic;
        i_dma_half_full:     	in  std_logic;
		random_seed:				in  std_logic_vector (15 downto 0);
		start_global_time:		in  std_logic_vector(47 downto 0);
		data_pix_generated:  	out std_logic_vector(31 downto 0);
		datak_pix_generated:  	out std_logic_vector(3 downto 0);
		data_pix_ready:      	out std_logic;
		slow_down:			in  std_logic_vector(31 downto 0);
		state_out:  	out std_logic_vector(3 downto 0)
);
end entity data_generator_a10;

architecture rtl of data_generator_a10 is

----------------signals---------------------
	signal global_time:			  std_logic_vector(47 downto 0);
	signal reset_n:				  std_logic;
	-- state_types
	type data_header_states is (part1, part2, part3, part4, part5, trailer, overflow);
	signal data_header_state:   data_header_states;

	-- random signals
	signal lsfr_chip_id:     	  std_logic_vector (5 downto 0);
	signal lsfr_tot:     	  	  std_logic_vector (5 downto 0);
	signal lsfr_row:     	  	  std_logic_vector (7 downto 0);
	signal lsfr_col:     	     std_logic_vector (7 downto 0);
	signal lsfr_overflow:        std_logic_vector (15 downto 0);
	
	-- slow down signals
	signal waiting:				  std_logic;
	signal wait_counter:			  std_logic_vector(31 downto 0);

----------------begin data_generator------------------------
begin

	reset_n <= not reset;

	chip_id_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 6,
		g_poly 	   	=> "110000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed(5 downto 0),
		i_en 				=> enable_pix,    
		o_lsfr 			=> lsfr_chip_id
	);
	
	pix_tot_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 6,
		g_poly 	   	=> "110000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed(15 downto 10),
		i_en 				=> enable_pix,    
		o_lsfr 			=> lsfr_tot
	);
	
	pix_row_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 8,
		g_poly 	   	=> "10111000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed(7 downto 0),
		i_en 				=> enable_pix,    
		o_lsfr 			=> lsfr_row
	);
	
	pix_col_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 8,
		g_poly 	   	=> "10111000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed(8 downto 1),
		i_en 				=> enable_pix,    
		o_lsfr 			=> lsfr_col
	);
	
	overflow_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 16,
		g_poly 	   	=> "1101000000001000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed,
		i_en 				=> enable_pix,    
		o_lsfr 			=> lsfr_overflow
	);

-- slow down process
process(clk, reset)
begin
	if(reset = '1') then
		waiting 			<= '0';
		wait_counter	<= (others => '0');
	elsif(rising_edge(clk)) then
		if(wait_counter >= slow_down) then
			wait_counter 	<= (others => '0');
			waiting 			<= '0';
		else
			wait_counter		<= wait_counter + '1';
			waiting			<= '1';
		end if;
	end if;
end process;
	
	
	
process (clk, reset)

variable current_overflow : std_logic_vector(15 downto 0) := "0000000000000000";
variable overflow_idx	  : integer range 0 to 15 := 0;

begin
	if (reset = '1') then
		data_pix_ready          <= '0';
		data_pix_generated      <= (others => '0');
		global_time       		<= (others => '0');--start_global_time;
		data_header_state			<= part1;
		current_overflow 			:= "0000000000000000";
		overflow_idx				:= 0;
		state_out					<= (others => '0');
		datak_pix_generated		<= (others => '1');
	elsif rising_edge(clk) then
		if(enable_pix = '1' and waiting = '0' and i_dma_half_full = '0') then
				data_pix_ready <= '1';
				case data_header_state is
					when part1 =>
						state_out <= x"A";
						global_time <= global_time + '1';
						data_pix_generated(31 downto 26) <= "111010";
						data_pix_generated(25 downto 8) 	<= (others => '0');
						data_pix_generated(7 downto 0) 	<= x"bc";
						datak_pix_generated              <= "0001";
						data_header_state 					<= part2;
				
					when part2 =>
						state_out <= x"B";
						global_time <= global_time + '1';
						data_pix_generated(31 downto 0) 	<= global_time(23 downto 0) & x"00";
						datak_pix_generated              <= "0000";
						data_header_state 					<= part3;
					
					when part3 =>
						state_out <= x"C";
						global_time <= global_time + '1';
						data_pix_generated					<= global_time(23 downto 0) & x"00";
						datak_pix_generated              <= "0000";
						data_header_state 					<= part4;
						
					when part4 =>
						state_out <= x"D";
						global_time <= global_time + '1';
						data_pix_generated 					<= "0000" & DATA_SUB_HEADER_ID & global_time(13 downto 0) & x"00";
						datak_pix_generated              <= "0000";
						overflow_idx 							:= 0;
						current_overflow						:= lsfr_overflow;
						data_header_state 					<= part5;
					
					when part5 =>
						state_out <= x"E";
						global_time <= global_time + '1';
						data_pix_generated					<= global_time(23 downto 0) & x"00";
						datak_pix_generated              <= "0000";
						if (global_time(3 downto 0) = "1111") then
							data_header_state					<= trailer;
						elsif (global_time(2 downto 0) = "111") then
							data_header_state					<= part4;
						elsif (current_overflow(overflow_idx) = '1') then
							overflow_idx 						:= overflow_idx + 1;
							data_header_state					<= overflow;
						else
							overflow_idx 						:= overflow_idx + 1;
						end if;
							
					when overflow =>
						state_out <= x"9";
						data_pix_generated					<= global_time(23 downto 0) & x"00";
						datak_pix_generated              <= "0000";
						data_header_state						<= part5;
					
					when trailer =>
						state_out <= x"8";
						global_time <= global_time + '1';
						data_pix_generated(31 downto 8)	<= (others => '0');
						data_pix_generated(7 downto 0)	<= x"9c";
						datak_pix_generated              <= "0001";
						data_header_state 					<= part1;
						
					when others =>
						state_out <= x"7";
						data_header_state 					<= trailer;
						---
				end case;
		else
			state_out <= x"F";
			data_pix_generated					<= x"000000BC";
			datak_pix_generated              <= "0001";
			data_pix_ready <= '0';
		end if;
	end if;
end process;


end rtl;
