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

use work.mudaq.all;

entity data_generator_a10 is
generic (
    fpga_id: std_logic_vector(15 downto 0) := x"FFFF";
    max_row: std_logic_vector (7 downto 0) := (others => '0');
    max_col: std_logic_vector (7 downto 0) := (others => '0');
    wtot: std_logic := '0';
    go_to_sh : positive := 2;
    go_to_trailer : positive := 3;
    wchip: std_logic := '0';
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE : std_logic_vector(7 downto 0) := x"01"--;
);
port (
    clk                 : in    std_logic;
    i_reset_n           : in    std_logic;
    enable_pix          : in    std_logic;
    i_dma_half_full     : in    std_logic;
    delay               : in    std_logic_vector (15 downto 0);
    random_seed         : in    std_logic_vector (15 downto 0);
    start_global_time   : in    std_logic_vector(47 downto 0);
    data_pix_generated  : out   std_logic_vector(31 downto 0);
    datak_pix_generated : out   std_logic_vector(3 downto 0);
    data_pix_ready      : out   std_logic;
    slow_down           : in    std_logic_vector(31 downto 0);
    state_out           : out   std_logic_vector(3 downto 0)
);
end entity;

architecture rtl of data_generator_a10 is

----------------signals---------------------
	signal global_time:			  std_logic_vector(47 downto 0);
	signal time_cnt_t:			  std_logic_vector(31 downto 0);
	signal overflow_time:			  std_logic_vector(14 downto 0);
	signal reset:				  std_logic;
	-- state_types
	type data_header_states is (part1, part2, part3, part4, part5, part6, trailer, overflow);
	signal data_header_state:   data_header_states;

	-- random signals
	signal lsfr_chip_id, lsfr_chip_id_reg:     	  std_logic_vector (5 downto 0);
	signal lsfr_tot, lsfr_tot_reg:     	  	  std_logic_vector (5 downto 0);
	signal row:     	  	  std_logic_vector (7 downto 0);
	signal col:     	     std_logic_vector (7 downto 0);
	signal lsfr_overflow, delay_cnt:        std_logic_vector (15 downto 0);

	-- slow down signals
	signal waiting:				  std_logic;
	signal wait_counter:			  std_logic_vector(31 downto 0);

----------------begin data_generator------------------------
begin

	reset <= not i_reset_n;

	chip_id_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 6,
		g_poly 	   	=> "110000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> i_reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed(5 downto 0),
		i_en 				=> enable_pix,
		o_lfsr 			=> lsfr_chip_id_reg
	);

	pix_tot_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 6,
		g_poly 	   	=> "110000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> i_reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed(15 downto 10),
		i_en 				=> enable_pix,
		o_lfsr 			=> lsfr_tot_reg
	);

	overflow_shift : entity work.linear_shift
	generic map(
		g_m 	   		=> 16,
		g_poly 	   	=> "1101000000001000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> i_reset_n,
		i_sync_reset	=> reset,--sync_reset,
		i_seed   		=> random_seed,
		i_en 				=> enable_pix,
		o_lfsr 			=> lsfr_overflow
	);

    -- slow down process
    process(clk, i_reset_n)
    begin
	if(i_reset_n = '0') then
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

lsfr_tot <= (others => '0') when wtot = '0' else lsfr_tot_reg;
lsfr_chip_id <= (others => '0') when wchip = '0' else lsfr_chip_id_reg;



    process (clk, i_reset_n, start_global_time)

        variable current_overflow : std_logic_vector(15 downto 0) := "0000000000000000";
        variable overflow_idx	  : integer range 0 to 15 := 0;

    begin
	if (i_reset_n = '0') then
		data_pix_ready          <= '0';
		data_pix_generated      <= (others => '0');
		global_time       		<= start_global_time;
		time_cnt_t       		<= (others => '0');
		data_header_state			<= part1;
		current_overflow 			:= "0000000000000000";
		overflow_idx				:= 0;
		state_out					<= (others => '0');
		delay_cnt					<= (others => '0');
		datak_pix_generated		<= (others => '1');
		row <= (others => '0');
		col <= (others => '0');
	elsif rising_edge(clk) then
		if(enable_pix = '1' and waiting = '0' and i_dma_half_full = '0') then
            data_pix_ready <= '1';
            case data_header_state is
            when part1 =>
            	state_out <= x"A";
            	if ( delay_cnt = delay ) then
            		data_header_state 					<= part2;
            		data_pix_generated(31 downto 26) 	<= DATA_HEADER_ID;
            		data_pix_generated(25 downto 24) 	<= (others => '0');
            		data_pix_generated(23 downto 8) 	<= fpga_id;
            		data_pix_generated(7 downto 0) 		<= x"bc";
            		datak_pix_generated              	<= "0001";
            	else
            		-- send not valid data out of the data package
            		delay_cnt 							<= delay_cnt + '1';
            		data_pix_generated					<= x"AFFEAFFE";
            		datak_pix_generated              	<= "0000";
            	end if;


            when part2 =>
            	state_out <= x"B";
            	delay_cnt <= (others => '0');
            	data_pix_generated(31 downto 0) 	<= global_time(47 downto 16);
            	global_time 						<= global_time + '1';
            	datak_pix_generated              	<= "0000";
            	data_header_state 					<= part3;

            when part3 =>
            	state_out <= x"C";
            	if ( DATA_TYPE = x"01" ) then
                    data_pix_generated					<= global_time(15 downto 0) & x"0000";
                elsif ( DATA_TYPE = x"02" ) then
                    data_pix_generated					<= global_time(15 downto 0) & x"AFFE";
                end if;
            	datak_pix_generated              <= "0000";
            	data_header_state 					<= part4;

            when part4 =>
            	state_out <= x"D";
            	global_time <= global_time + '1';
            	if ( DATA_TYPE = x"01" ) then
                    data_pix_generated 					<= DATA_SUB_HEADER_ID & "000" & global_time(10 downto 4) & lsfr_overflow;
                elsif ( DATA_TYPE = x"02" ) then
                    data_pix_generated 					<= DATA_SUB_HEADER_ID & global_time(13 downto 4) & lsfr_overflow;
                end if;
            	datak_pix_generated              <= "0000";
            	overflow_idx 							:= 0;
            	current_overflow						:= lsfr_overflow;
            	data_header_state 					<= part5;

            when part5 =>
                global_time <= global_time + '1';
                if ( DATA_TYPE = x"01" ) then
                    data_pix_generated 					<= DATA_SUB_HEADER_ID & "000" & global_time(10 downto 4) & lsfr_overflow;
                elsif ( DATA_TYPE = x"02" ) then
                    data_pix_generated 					<= DATA_SUB_HEADER_ID & global_time(13 downto 4) & lsfr_overflow;
                end if;
            	datak_pix_generated              <= "0000";
            	overflow_idx 							:= 0;
            	current_overflow						:= lsfr_overflow;
            	data_header_state 					<= part6;

            when part6 =>
            	state_out <= x"E";
            	global_time <= global_time + '1';
            	time_cnt_t <= time_cnt_t + '1';

            	if (row = max_row) then
                    row <= (others => '0');
                else
                    row <= row + '1';
                end if;

                if (row = max_col) then
                    col <= (others => '0');
                else
                    col <= col + '1';
                end if;

                if ( DATA_TYPE = x"01" ) then
                    data_pix_generated					<= global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
                elsif ( DATA_TYPE = x"02" ) then
                    data_pix_generated(31 downto 21)    <= (others => '0');
                    data_pix_generated(20 downto 6)     <= global_time(14 downto 0);
                    data_pix_generated(5 downto 0)    <= (others => '0');
                end if;
            	overflow_time <= global_time(14 downto 0);
            	datak_pix_generated              <= "0000";
            	if ( work.util.and_reduce(global_time(go_to_trailer downto 0)) = '1' ) then
            		data_header_state					<= trailer;
            		time_cnt_t       		<= (others => '0');
            	elsif ( work.util.and_reduce(global_time(go_to_sh downto 0)) = '1' ) then
            		data_header_state					<= part4;
            	elsif (current_overflow(overflow_idx) = '1') then
            		overflow_idx 						:= overflow_idx + 1;
            		data_header_state					<= overflow;
            	else
            		overflow_idx 						:= overflow_idx + 1;
            	end if;

            when overflow =>
            	state_out <= x"9";
            	if ( DATA_TYPE = x"01" ) then
                    data_pix_generated				  <= overflow_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
                elsif ( DATA_TYPE = x"02" ) then
                    data_pix_generated(31 downto 21)  <= (others => '0');
                    data_pix_generated(20 downto 6)   <= overflow_time(14 downto 0);
                    data_pix_generated(5 downto 0)    <= (others => '0');
                end if;
            	datak_pix_generated              <= "0000";
            	data_header_state						<= part6;

            when trailer =>
            	state_out <= x"8";
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

end architecture;
