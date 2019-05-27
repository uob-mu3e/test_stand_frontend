-- simple data generator (for slowcontrol and pixel data)
-- writes into pix_data_fifo and sc_data_fifo
-- only Header(sc or pix) + data
-- other headers/signals are added in data_merger.vhd

-- Martin Mueller, January 2019
-- Marius Koeppel, March 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use work.protocol.all;
use work.downstream_communication_components.all;


entity data_generator is
    port(
		clk:                 	in  std_logic;
		reset:               	in  std_logic;
		enable_pix:          	in  std_logic;
		--enable_sc:         	in  std_logic;
		random_seed:		in  std_logic_vector (15 downto 0);
		data_pix_generated:  	out std_logic_vector(35 downto 0);
		--data_sc_generated:   	out std_logic_vector(31 downto 0);
		data_pix_ready:      	out std_logic;
		--data_sc_ready:      	out std_logic;
		start_global_time:	in std_logic_vector(47 downto 0)
			  -- TODO: add some rate control
);
end entity data_generator;

architecture rtl of data_generator is

----------------signals---------------------
   --signal sc_data_counter:     std_logic_vector(3 downto 0);
	signal global_time:			  std_logic_vector(47 downto 0);
	signal reset_n:				  std_logic;
	-- state_types
	type data_header_states is (part1, part2, part3, part4, trailer, overflow);
	signal data_header_state:   data_header_states;

	-- random signals
	signal lsfr_chip_id:     	  std_logic_vector (5 downto 0);
	signal lsfr_tot:     	  	  std_logic_vector (5 downto 0);
	signal lsfr_row:     	  	  std_logic_vector (7 downto 0);
	signal lsfr_col:     	     std_logic_vector (7 downto 0);
	signal lsfr_overflow:       std_logic_vector (15 downto 0);
	signal wait_cnt: std_logic := '0';

----------------begin data_generator------------------------
begin

	reset_n <= not reset;

	chip_id_shift: linear_shift 
	generic map(
		g_m 	   		=> 6,
		g_poly 	   		=> "110000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset		=> reset,--sync_reset,
		i_seed   		=> random_seed(5 downto 0),
		i_en 			=> enable_pix,    
		o_lsfr 			=> lsfr_chip_id
	);
	
	pix_tot_shift: linear_shift 
	generic map(
		g_m 	   		=> 6,
		g_poly 	   		=> "110000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset		=> reset,--sync_reset,
		i_seed   		=> random_seed(15 downto 10),
		i_en 			=> enable_pix,    
		o_lsfr 			=> lsfr_tot
	);
	
	pix_row_shift: linear_shift 
	generic map(
		g_m 	   		=> 8,
		g_poly 	   		=> "10111000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset		=> reset,--sync_reset,
		i_seed   		=> random_seed(7 downto 0),
		i_en 			=> enable_pix,    
		o_lsfr 			=> lsfr_row
	);
	
	pix_col_shift: linear_shift 
	generic map(
		g_m 	   		=> 8,
		g_poly 	   		=> "10111000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset		=> reset,--sync_reset,
		i_seed   		=> random_seed(8 downto 1),
		i_en 			=> enable_pix,    
		o_lsfr 			=> lsfr_col
	);
	
	overflow_shift: linear_shift 
	generic map(
		g_m 	   		=> 16,
		g_poly 	   		=> "1101000000001000"
	)
	port map(
		i_clk    		=> clk,
		reset_n   		=> reset_n,
		i_sync_reset		=> reset,--sync_reset,
		i_seed   		=> random_seed,
		i_en 			=> enable_pix,    
		o_lsfr 			=> lsfr_overflow
	);

process (clk)

variable current_overflow : std_logic_vector(15 downto 0) := "0000000000000000";
variable overflow_idx	  : integer range 0 to 15 := 0;

begin
	if (reset = '1') then
		data_pix_ready          <= '0';
		--data_sc_ready           <= '0';
		data_pix_generated      <= (others => '0');
		--data_sc_generated       <= (others => '0');
		global_time       		<= start_global_time;
		--sc_data_counter         <= (others => '1');
		data_header_state			<= part1;
		wait_cnt						<= '0';
		current_overflow 			:= "0000000000000000";
		overflow_idx				:= 0;
	elsif rising_edge(clk) then
        -- generate pix data
		if(enable_pix='1') then
			if (wait_cnt = '1') then
				data_pix_ready <= '1';
				case data_header_state is
					when trailer =>
						data_pix_generated(35 downto 32)	<= "0011";
						data_pix_generated(31 downto 0)	<= (others => '0');
						data_header_state 					<= part1;
					when part1 =>
						data_pix_generated					<= "0010" & global_time(47 downto 16);
						data_header_state 					<= part2;
					when part2 =>
						data_pix_generated					<= "0000" & global_time(15 downto 0) & x"0000";
						data_header_state 					<= part3;
					when part3 =>
						data_pix_generated 					<= "0000" & "0000" & DATA_SUB_HEADER_ID & global_time(9 downto 4) & lsfr_overflow;
						global_time								<= global_time + '1';
						overflow_idx 							:= 0;
						current_overflow						:= lsfr_overflow;
						data_header_state 					<= part4;
					when part4 =>
						if (lsfr_chip_id = DATA_SUB_HEADER_ID) then
							data_pix_generated				<= "0000" & global_time(3 downto 0) & "000000" & global_time(21 downto 0);-- "101010" & lsfr_row & lsfr_col & lsfr_tot;
						elsif (lsfr_chip_id = DATA_HEADER_ID) then
							data_pix_generated				<= "0000" & global_time(3 downto 0) & "000000" & global_time(21 downto 0); -- "010101" & lsfr_row & lsfr_col & lsfr_tot;
						else
							data_pix_generated				<= "0000" & global_time(3 downto 0) & "000000" & global_time(21 downto 0); --lsfr_chip_id & lsfr_row & lsfr_col & lsfr_tot;
						end if;
						
						if (current_overflow(overflow_idx) = '1') then
							overflow_idx 						:= overflow_idx + 1;
							data_header_state					<= overflow;
						else
							overflow_idx 						:= overflow_idx + 1;
							global_time 						<= global_time + '1';
						end if;
						
						if (global_time(9 downto 0) = "1111111111") then
							data_header_state 				<= trailer;
						elsif (global_time(3 downto 0) = "1111") then
							data_header_state 				<= part3;
						end if;
					when overflow =>
						if (lsfr_chip_id = DATA_SUB_HEADER_ID) then
							data_pix_generated				<= "0000" & global_time(3 downto 0) & "000000" & global_time(21 downto 0);-- "101010" & lsfr_row & lsfr_col & lsfr_tot;
						elsif (lsfr_chip_id = DATA_HEADER_ID) then
							data_pix_generated				<= "0000" & global_time(3 downto 0) & "000000" & global_time(21 downto 0); -- "010101" & lsfr_row & lsfr_col & lsfr_tot;
						else
							data_pix_generated				<= "0000" & global_time(3 downto 0) & "000000" & global_time(21 downto 0); --lsfr_chip_id & lsfr_row & lsfr_col & lsfr_tot;
						end if;
						global_time 						<= global_time + '1';
						data_header_state					<= part4;
					when others =>
						data_header_state 					<= trailer;
						---
				end case;
			else
				data_pix_ready <= '0';
				wait_cnt <= not wait_cnt;
			end if;
		else 
			data_pix_ready <= '0';
      end if;
        
        -- generate sc data
        --if(enable_sc='1') then 
        --    if(sc_data_counter(3) = '1')then
        --        data_sc_generated <= "0000" & SC_HEADER_ID &"000000"& x"0000";
        --        data_sc_ready <= '1';
        --       sc_data_counter <= (others => '0');
        --    else 
        --        data_sc_generated <= (8 =>'1',others => '0');
        --        sc_data_counter <= sc_data_counter + '1';
        --        data_sc_ready   <= '1';
        --    end if;
        --else 
        --    data_sc_ready <= '0';
        --    sc_data_counter <= (others => '1');
        --end if;
        
	end if;
end process;


END rtl;
