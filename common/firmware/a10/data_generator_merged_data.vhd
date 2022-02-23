-- simple data generator merged data after
-- SWB alignment

-- Marius Koeppel, February 2021

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;


entity data_generator_merged_data is
generic (
    max_row: std_logic_vector (7 downto 0) := (others => '0');
    max_col: std_logic_vector (7 downto 0) := (others => '0');
    go_to_sh: positive := 3;
    go_to_trailer: positive := 4;
    wtot: std_logic := '0';
    wchip: std_logic := '0';
    NLINKS: positive := 8--;
);
port (
    i_clk       : in    std_logic;
    i_reset_n   : in    std_logic;

    i_en        : in    std_logic;
    i_sd        : in    std_logic_vector(31 downto 0);

    o_data      : out   std_logic_vector(NLINKS * 38 - 1 downto 0);
    o_data_we   : out   std_logic;
    o_state     : out   std_logic_vector(3 downto 0)--;
);
end entity;

architecture rtl of data_generator_merged_data is

----------------signals---------------------
	signal global_time 			   	: std_logic_vector(47 downto 0);
	signal time_cnt_t, wait_counter	: std_logic_vector(31 downto 0);

	constant time_cnt_sh_one 		: std_logic_vector(go_to_sh downto 0) := (others => '1');
	constant time_cnt_t_one 		: std_logic_vector(go_to_trailer downto 0) := (others => '1');

	signal overflow_time 			: std_logic_vector(3 downto 0);
	signal reset, waiting 			: std_logic;

	-- state_types
	type data_header_states is (part1, part2, part3, part4, part5, trailer, overflow);
	signal data_header_state : data_header_states;

	-- random signals
	signal lsfr_chip_id, lsfr_chip_id_reg, lsfr_tot, lsfr_tot_reg : std_logic_vector (5 downto 0);
	signal row, col 					: std_logic_vector (7 downto 0);
	signal lsfr_overflow, random_seed 	: std_logic_vector (15 downto 0);

----------------begin data_generator------------------------
begin

	reset 		<= not i_reset_n;
	random_seed <= (others => '1');

	e_chip_id : entity work.linear_shift
	generic map(
		g_m 	   	=> 6,
		g_poly 	   	=> "110000"
	)
	port map(
		i_clk    		=> i_clk,
		reset_n   		=> i_reset_n,
		i_sync_reset	=> reset,
		i_seed   		=> random_seed(5 downto 0),
		i_en 			=> i_en,
		o_lfsr 			=> lsfr_chip_id_reg--,
	);

	e_tot : entity work.linear_shift
	generic map(
		g_m 	   	=> 6,
		g_poly 	   	=> "110000"
	)
	port map(
		i_clk    		=> i_clk,
		reset_n   		=> i_reset_n,
		i_sync_reset	=> reset,
		i_seed   		=> random_seed(15 downto 10),
		i_en 			=> i_en,
		o_lfsr 			=> lsfr_tot_reg--,
	);

	e_overflow : entity work.linear_shift
	generic map(
		g_m 	   	=> 16,
		g_poly 	   	=> "1101000000001000"
	)
	port map(
		i_clk    		=> i_clk,
		reset_n   		=> i_reset_n,
		i_sync_reset	=> reset,
		i_seed   		=> random_seed,
		i_en 			=> i_en,
		o_lfsr 			=> lsfr_overflow--,
	);

-- slow down process
process(i_clk, i_reset_n)
begin
	if ( i_reset_n = '0' ) then
		waiting 			<= '0';
		wait_counter		<= (others => '0');
		--
	elsif ( rising_edge(i_clk) ) then

		if(wait_counter >= i_sd) then
			wait_counter 	<= (others => '0');
			waiting 		<= '0';
		else
			wait_counter	<= wait_counter + '1';
			waiting			<= '1';
		end if;
		--
	end if;
end process;

lsfr_tot 		<= (others => '0') when wtot = '0' else lsfr_tot_reg;
lsfr_chip_id 	<= (others => '0') when wchip = '0' else lsfr_chip_id_reg;

process(i_clk, i_reset_n)

	variable current_overflow : std_logic_vector(15 downto 0) := "0000000000000000";
	variable overflow_idx	  : integer range 0 to 15 := 0;

begin
	if ( i_reset_n = '0' ) then
		o_data_we         	<= '0';
		o_data      		<= (others => '0');
		global_time       	<= (others => '0');
		time_cnt_t       	<= (others => '0');
		data_header_state	<= part1;
		current_overflow 	:= "0000000000000000";
		overflow_idx		:= 0;
		o_state				<= (others => '0');
		row 				<= (others => '0');
		col 				<= (others => '0');
		--
	elsif ( rising_edge(i_clk) ) then

		if ( i_en = '1' and waiting = '0' ) then

			o_data_we <= '1';
			o_data    <= (others => '0');

				case data_header_state is

					when part1 =>
						o_state 				<= x"A";
						o_data(37 downto 32)	<= pre_marker;
						o_data(31 downto 26) 	<= "111010";
						o_data(7 downto 0) 		<= x"BC";
						data_header_state 		<= part2;

					when part2 =>
						o_state 			 	<= x"B";
                    	o_data(37 downto 32) 	<= ts1_marker;
                    	o_data(31 downto 0)  	<= global_time(47 downto 16);
						data_header_state 	 	<= part3;

					when part3 =>
						o_state 				<= x"C";
                    	o_data(37 downto 32) 	<= ts2_marker;
                    	o_data(31 downto 0) 	<= global_time(15 downto 0) & x"0000";
						data_header_state 					<= part4;

					when part4 =>
						o_state 				<= x"D";
						global_time 			<= global_time + '1';
						o_data(37 downto 32) 	<= sh_marker;
                    	o_data(31 downto 26) 	<= "111111";
                    	o_data(25 downto 23) 	<= "000";
                    	o_data(22 downto 16) 	<= global_time(10 downto 4);
                    	o_data(15 downto 0)		<= lsfr_overflow;
						overflow_idx 			:= 0;
						current_overflow		:= lsfr_overflow;
						data_header_state 		<= part5;

					when part5 =>
						o_state <= x"E";
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

						o_data(37  downto   0)	<= "000000" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(75  downto  38)	<= "000001" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(113 downto  76)	<= "000010" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(151 downto 114)	<= "000011" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(189 downto 152)	<= "000100" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(227 downto 190)	<= "000101" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(265 downto 228)	<= "000110" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(303 downto 266)	<= "000111" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;

						overflow_time 			<= global_time(3 downto 0);

						if ( time_cnt_t(go_to_trailer downto 0) = time_cnt_t_one ) then
							data_header_state <= trailer;
							time_cnt_t        <= (others => '0');
						elsif ( global_time(go_to_sh downto 0) = time_cnt_sh_one ) then
							data_header_state <= part4;
						elsif ( current_overflow(overflow_idx) = '1' ) then
							overflow_idx 	  := overflow_idx + 1;
							data_header_state <= overflow;
						else
							overflow_idx 	  := overflow_idx + 1;
						end if;

					when overflow =>
						o_state <= x"9";

						o_data(37  downto   0)	<= "000000" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(75  downto  38)	<= "000001" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(113 downto  76)	<= "000010" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(151 downto 114)	<= "000011" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(189 downto 152)	<= "000100" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(227 downto 190)	<= "000101" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(265 downto 228)	<= "000110" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;
						o_data(303 downto 266)	<= "000111" & global_time(3 downto 0) & lsfr_chip_id & row & col & lsfr_tot;

						data_header_state						<= part5;

					when trailer =>
						o_state 				<= x"8";
						o_data(37 downto 32) 	<= tr_marker;
                    	o_data(7 downto 0) 		<= x"9C";
						data_header_state 					<= part1;

					when others =>
						o_state				<= x"7";
						data_header_state 	<= trailer;
						--
				end case;

		else
			o_state 	<= x"F";
			o_data_we 	<= '0';
		end if;
	end if;
end process;

end architecture;
