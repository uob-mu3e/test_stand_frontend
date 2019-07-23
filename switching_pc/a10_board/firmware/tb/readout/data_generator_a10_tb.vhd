library ieee;
use ieee.std_logic_1164.all;

package protocol is

    type data_merger_state is (idle, sending_data, sending_slowcontrol);
    --type feb_state is (idle, run_prep, sync, running, terminating, link_test, sync_test, reset_state, out_of_DAQ);

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


entity data_generator_a10_tb is
    port(
		clk:                 	in  std_logic;
		reset:               	in  std_logic;
		enable_pix:          	in  std_logic;
		random_seed:			in  std_logic_vector (15 downto 0);
		start_global_time:		in  std_logic_vector(47 downto 0);
		data_pix_generated:  	out std_logic_vector(31 downto 0);
		data_pix_ready:      	out std_logic;
		slow_down:				in  std_logic_vector(31 downto 0)
);
end entity data_generator_a10_tb;

architecture rtl of data_generator_a10_tb is

----------------signals---------------------
	signal global_time:			  std_logic_vector(47 downto 0);
	signal reset_n:				  std_logic;
	-- state_types
	type data_header_states is (header, ts1, ts2, subheader, data, trailer);
	signal data_header_state:   data_header_states;
	
	-- slow down signals
	signal waiting:				  std_logic;
	signal wait_counter:		  std_logic_vector(31 downto 0);

----------------begin data_generator------------------------
begin

reset_n <= not reset;

-- slow down process
process(clk, reset)
begin
	if(reset = '1' or enable_pix = '0') then
		waiting 			<= '0';
		wait_counter	<= (others => '0');
	elsif(rising_edge(clk)) then
		if(wait_counter = slow_down) then
			wait_counter 	<= (others => '0');
			waiting 			<= '0';
		else
			wait_counter		<= wait_counter + '1';
			waiting			<= '1';
		end if;
	end if;
end process;
	
	
	
process (clk, reset)
begin
	if (reset = '1') then
		data_pix_ready          <= '0';
		data_pix_generated      <= (others => '0');
		global_time       		<= start_global_time;
		data_header_state		<= header;	
	elsif rising_edge(clk) then
		if(enable_pix = '1' and waiting = '0') then
				data_pix_ready <= '1';
				case data_header_state is

					when header =>
						data_pix_generated(31 downto 26) 	<= "111010";
						data_pix_generated(25 downto 8) 	<= (others => '0');
						data_pix_generated(7 downto 0) 		<= x"bc";
						data_header_state 					<= ts1;
						
					when ts1 =>
						data_pix_generated(31 downto 0) 	<= global_time(47 downto 24) & x"00";
						data_header_state 					<= ts2;
						
					when ts2 =>
						data_pix_generated					<= global_time(15 downto 0) & x"0000";
						data_header_state 					<= subheader;
						
					when subheader =>
						data_pix_generated 					<= "0000" & DATA_SUB_HEADER_ID & global_time(9 downto 4) & x"0000";
						global_time							<= global_time + '1';
						data_header_state 					<= data;
					
					when data =>
						data_pix_generated 					<= "0000" & global_time(15 downto 4) & x"0000";
						global_time							<= global_time + '1';
						data_header_state 					<= trailer;	
						
					when trailer =>
						data_pix_generated 					<= x"0000009c";
						global_time 						<= global_time + '1';
						data_header_state					<= header;
						
					when others =>
						data_header_state 					<= header;
						---
				end case;
		else
			data_pix_ready <= '0';
		end if;
	end if;
end process;
end rtl;