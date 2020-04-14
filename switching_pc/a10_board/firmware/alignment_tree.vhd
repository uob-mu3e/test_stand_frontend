library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity alginment_tree is 
generic (
	NLINKS : integer := 32;
	NFIRST : integer := 4;
	NSECOND : integer := 2;
    LINK_FIFO_ADDR_WIDTH : integer := 10 --;
);
port (
	i_clk_250 	: in  std_logic;
	i_reset_n	: in  std_logic;

	i_data 		: in std_logic_vector(NLINKS * 32 - 1 downto 0);
	i_datak		: in std_logic_vector(NLINKS * 4 - 1 downto 0);
	
	o_data	    : out std_logic_vector(255 downto 0);
	o_datak	    : out std_logic_vector(32 downto 0);
	o_error		: out std_logic_vector(3 downto 0)--;
);
end entity;

architecture rtl of alginment_tree is
----------------signals--------------------
signal reset : std_logic;

-- link fifos
signal link_fifo_wren       : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_data       : std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_ren        : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_data_out   : std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_empty      : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_full       : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_almost_full: std_logic_vector(NLINKS - 1 downto 0);

-- first layer fifos
type first_layer_type is (wait_for_preamble, read_out_ts_1, read_out_ts_2, read_out_sub_header, ts_error, algin_first_layer);
signal first_layer_state : first_layer_type;
type array_16_type is array (NLINKS - 1 downto 0) of std_logic_vector(15 downto 0);
signal link_fpga_id : array_16_type;
signal link_sub_overflow : array_16_type;
type ts_array_type is array (NLINKS - 1 downto 0) of std_logic_vector(47 downto 0);
signal link_fpga_ts : ts_array_type;

signal first_layer_wren       : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_data       : std_logic_vector(NFIRST * 38 - 1 downto 0);
signal first_layer_ren        : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_data_out   : std_logic_vector(NFIRST * 38 - 1 downto 0);
signal first_layer_empty      : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_full       : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_almost_full: std_logic_vector(NFIRST - 1 downto 0);
signal saw_preamble			  : std_logic_vector(NLINKS - 1 downto 0);
signal saw_sheader			  : std_logic_vector(NLINKS - 1 downto 0);
signal saw_trailer			  : std_logic_vector(NLINKS - 1 downto 0);
constant all_ones 			  : std_logic_vector(NLINKS - 1 downto 0) := (others => '1');

begin

reset <= not i_reset_n;

-- write to link fifos
process(i_clk_250, i_reset_n)
begin
	if(i_reset_n = '0') then
		link_fifo_wren <= (others => '0');
		link_fifo_data <= (others => '0');
	elsif(rising_edge(i_clk_250)) then
		set_link_data : FOR i in 0 to NLINKS - 1 LOOP
			link_fifo_data(35 + i * 36 downto i * 36) <= i_data(31 + i * 32 downto i * 32) & i_datak(3 + i * 4 downto i * 4);
			if ( ( i_data(31 + i * 32 downto i * 32) = x"000000BC" and i_datak(3 + i * 4 downto i * 4) = "0001" ) or 
                 ( i_data(31 + i * 32 downto i * 32) = x"00000000" and i_datak(3 + i * 4 downto i * 4) = "1111" )                 
            ) then
	         	link_fifo_wren(i) <= '0';
        	else
				link_fifo_wren(i) <= '1';
      		end if;
		END LOOP set_link_data;
	end if;
end process;

-- generate fifos per link
link_fifos:
FOR i in 0 to NLINKS - 1 GENERATE
	
    e_link_fifo : entity work.ip_scfifo
    generic map(
        ADDR_WIDTH 	=> LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH 	=> 36,
        DEVICE 		=> "Arria 10"--,
	)
	port map (
		data     		=> link_fifo_data(35 + i * 36 downto i * 36),
		wrreq    		=> link_fifo_wren(i),
		rdreq    		=> link_fifo_ren(i),
		clock    		=> i_clk_250,
		q    	 		=> link_fifo_data_out(35 + i * 36 downto i * 36),
		full     		=> link_fifo_full(i),
		empty    		=> link_fifo_empty(i),
		almost_empty 	=> open,
		almost_full 	=> link_fifo_almost_full(i),
		usedw 			=> open,
		sclr     		=> reset--,
	);

END GENERATE link_fifos;

-- write link data to first tree layer
process(i_clk_250, i_reset_n)
	variable temp_hit_ts : std_logic_vector (3 downto 0);  -- lower 4 bits of ts
    variable var_hits    : std_logic_vector (37 downto 0); -- new hit sice
begin
	if( i_reset_n = '0' ) then
		-- state machine singals
		first_layer_state	<= wait_for_preamble;
		link_fifo_ren 		<= (others => '0');
		saw_preamble		<= (others => '0');
		saw_sheader			<= (others => '0');
		link_fpga_id		<= (others => (others => '0'));
		link_fpga_ts		<= (others => (others => '0'));
		o_error				<= (others => '0');
		saw_trailer			<= (others => '0');
		-- first layer signals
		first_layer_data 	<= (others => '0');
		first_layer_wren 	<= (others => '0');
	elsif( rising_edge(i_clk_250) ) then

		o_error			 <= (others => '0');
		first_layer_wren <= (others => '0');

		case first_layer_state is

			when wait_for_preamble =>
			-- TODO: timeout?
				if ( saw_preamble = all_ones ) then
					link_fifo_ren <= (others => '1');
					first_layer_state <= read_out_ts_1;
				else
					FOR I IN 0 to NLINKS - 1 LOOP
						if(	(link_fifo_data_out(35 + I * 36 downto I * 36 + 30) = "111010" )
							and 
							(link_fifo_data_out(11 + I * 36 downto I * 36 + 4) = x"bc")
							and
							(link_fifo_data_out(3 + I * 36 downto I * 36) = "0001")
						) then
							saw_preamble(I) 	<= '1';
							link_fpga_id(I) 	<= link_fifo_data_out(I * 36 + 23 downto I * 36 + 4);
							link_fifo_ren(I) 	<= '0';
						else
							link_fifo_ren(I) <= '1';
						end if;
					END LOOP;
				end if;

			when read_out_ts_1 =>
				FOR I IN 0 to NLINKS - 1 LOOP
					link_fpga_ts(I)(47 downto 16) <= link_fifo_data_out(35 + I * 36 downto I * 36 + 4);
				END LOOP;
				first_layer_state <= read_out_ts_2;

			when read_out_ts_2 =>
				FOR I IN 0 to NLINKS - 1 LOOP
					link_fpga_ts(I)(15 downto 0) <= link_fifo_data_out(35 + I * 36 downto I * 36 + 20);
				END LOOP;
				first_layer_state <= read_out_sub_header;

			when read_out_sub_header =>
				saw_sheader	<= (others => '0');
				first_layer_state <= algin_first_layer;
				-- check if TS are all the same
				FOR I IN 1 to NLINKS - 1 LOOP
					if ( link_fpga_ts(0) /= link_fpga_ts(I) ) then
						first_layer_state <= ts_error;
					end if;
					-- here we just send the same sheader 
					first_layer_data(I) <= (others => '1');
					first_layer_wren(I) <= '1';
				END LOOP;


			when algin_first_layer =>
				-- TODO: timeout?
				if ( saw_trailer = all_ones ) then
					first_layer_state <= saw_trailer;
				elsif ( saw_sheader = all_ones ) then
					link_fifo_ren <= (others => '1');
					first_layer_state <= read_out_sub_header;
				else
					FOR I IN 0 to NLINKS - 1 LOOP
						if(	(link_fifo_data_out(31 + I * 36 downto I * 36 + 26) = "111111" )
							and 
							(link_fifo_data_out(11 + I * 36 downto I * 36 + 4) = x"bc")
							and
							(link_fifo_data_out(3 + I * 36 downto I * 36) = "0001")
						) then
							saw_sheader(I) 		<= '1';
							link_fifo_ren(I) 	<= '0';
						elsif( (link_fifo_data_out(11 + I * 36 downto I * 36 + 4) = x"9c")
							   and
							   (link_fifo_data_out(3 + I * 36 downto I * 36) = "0001")
						) then
							saw_trailer(I)   <= '1';
							link_fifo_ren(I) <= '0';
						end if;
					END LOOP;
				end if;

				-- start with alignment here


			when ts_error =>
				o_error <= x"1";

			when others =>
				first_layer_state <= wait_for_preamble;


				w_ram_en					<= '1';
				w_ram_add   			<= w_ram_add + 1;
				w_ram_data  			<= trigger_mask & event_id;
				last_event_add 		<= w_ram_add + 1;
				event_tagging_state 	<= event_num;

				when event_num =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data  			<= serial_number;
					event_tagging_state 	<= event_tmp;

				when event_tmp =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data  			<= time_tmp;
					event_tagging_state 	<= event_size;

				when event_size =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data  			<= (others => '0');
					cur_size_add 			<= w_ram_add + 1;
					event_tagging_state 	<= bank_size;

				when bank_size =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data  			<= (others => '0');
					cur_bank_size_add 	<= w_ram_add + 1;
					event_size_cnt      	<= event_size_cnt + 4;
					event_tagging_state 	<= bank_flags;

				when bank_flags =>
					w_ram_en					<= '1';
					event_size_cnt      	<= event_size_cnt + 4;
					w_ram_add   			<= w_ram_add + 1;
					w_ram_add_reg 			<= w_ram_add + 1;
					w_ram_data				<= flags;
					event_tagging_state 	<= bank_name;

				when bank_name =>
               link_fifo_ren(current_link) <= '0';
					--here we check if the link is masked and if the current fifo is empty
					if ( i_link_mask_n(current_link) = '0' or link_fifo_empty(current_link) = '1' ) then
						--skip this link
						current_link <= current_link + 1;
						--last link, go to trailer bank
						if ( current_link + 1 = NLINKS ) then
                     if ( data_flag = '0' ) then
                        current_link <= 0;
                     else
                        event_tagging_state <= trailer_name;
                     end if;
						end if;
					else
						--check for mupix or mutrig data header
						if(	
							(link_fifo_data_out(35 + current_link * 36 downto current_link * 36 + 30) = "111010" or link_fifo_data_out(35 + current_link * 36 downto current_link * 36 + 30) = "111000")
							and
							(link_fifo_data_out(11 + current_link * 36 downto current_link * 36 + 4) = x"bc")
							and
							(link_fifo_data_out(3 + current_link * 36 downto current_link * 36) = "0001")
						) then
                     data_flag	<= '1';
							w_ram_en		<= '1';
							w_ram_add   <= w_ram_add_reg + 1;
							-- toDo: proper conversion into ASCII for the midas banks here !! 
							if(link_fifo_data_out(27 + current_link * 36 downto current_link * 36 + 12) = x"FEB0") then
								w_ram_data  		<= x"30424546";
							elsif(link_fifo_data_out(27 + current_link * 36 downto current_link * 36 + 12) = x"FEB1") then
								w_ram_data  		<= x"31424546";
							elsif(link_fifo_data_out(27 + current_link * 36 downto current_link * 36 + 12) = x"FEB2") then
								w_ram_data  		<= x"32424546";
							elsif(link_fifo_data_out(27 + current_link * 36 downto current_link * 36 + 12) = x"FEB3") then
								w_ram_data  		<= x"33424546";
							else
								w_ram_data  		<= x"34424546"; -- We should not see this !! (FEB4)
							end if;
							event_size_cnt      	<= event_size_cnt + 4;
							event_tagging_state 	<= bank_type;
						--throw data away until a header
						else
						   link_fifo_ren(current_link) <= '1';
						end if;
					end if;

				when bank_type =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data				<= type_bank;
					event_size_cnt    	<= event_size_cnt + 4;
					event_tagging_state 	<= bank_length;

				when bank_length =>
					w_ram_en								<= '1';
					w_ram_add   						<= w_ram_add + 1;
					w_ram_data  						<= (others => '0');
					event_size_cnt      				<= event_size_cnt + 4;
					cur_bank_length_add(11 + current_link * 12 downto current_link * 12) <= w_ram_add + 1;
					link_fifo_ren(current_link) 	<= '1';
					event_tagging_state 				<= bank_data;

				when bank_data =>
					-- check again if the fifo is empty
					if ( link_fifo_empty(current_link) = '0' ) then
						w_ram_en				<= '1';
						w_ram_add   		<= w_ram_add + 1;
						w_ram_data  		<= link_fifo_data_out(35 + current_link * 36 downto current_link * 36 + 4);
						event_size_cnt 	<= event_size_cnt + 4;
 					   bank_size_cnt 		<= bank_size_cnt + 4;
						if(  
							(link_fifo_data_out(11 + current_link * 36 downto current_link * 36 + 4) = x"9c")
							and 
							(link_fifo_data_out(3 + current_link * 36 downto current_link * 36) = "0001")
						) then
							-- check if the size of the bank data is in 64 bit if not add a word
							-- this word is not counted to the bank size
							if ( bank_size_cnt(2 downto 0) = "000" ) then
								event_tagging_state 	<= set_algin_word;
							else
								event_tagging_state 	<= bank_set_length;
								w_ram_add_reg 			<= w_ram_add + 1;
							end if;
							link_fifo_ren(current_link) <= '0';
						else
							link_fifo_ren(current_link) <= '1';
						end if;
					end if;
					
				when set_algin_word =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data 				<= x"AFFEAFFE";
					w_ram_add_reg 			<= w_ram_add + 1;
					event_size_cnt      	<= event_size_cnt + 4;
					event_tagging_state 	<= bank_set_length;

				when bank_set_length =>
               w_ram_en						<= '1';
					w_ram_add   				<= cur_bank_length_add(11 + current_link * 12 downto current_link * 12);
					w_ram_data 					<= bank_size_cnt;
					bank_size_cnt 				<= (others => '0');
					if ( current_link + 1 = NLINKS ) then
						event_tagging_state 	<= trailer_name;
					else
						current_link <= current_link + 1;
						event_tagging_state 	<= bank_name;
					end if;

				when trailer_name =>
					w_ram_en					<= '1';
	            w_ram_add   			<= w_ram_add_reg + 1;
					w_ram_data  			<= x"454b4146"; -- FAKE in ascii
					data_flag      		<= '0';
					current_link   		<= 0;
	            event_size_cnt 		<= event_size_cnt + 4;
	            event_tagging_state 	<= trailer_type;
	                
				when trailer_type =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data  			<= type_bank;
					event_size_cnt      	<= event_size_cnt + 4;
					event_tagging_state 	<= trailer_length;

				when trailer_length =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add + 1;
					w_ram_data  			<= (others => '0');
					-- reg trailer length add
					w_ram_add_reg 			<= w_ram_add + 1;
					event_size_cnt      	<= event_size_cnt + 4;
					-- write at least one AFFEAFFE
					align_event_size		<= w_ram_add + 1 - last_event_add;
					event_tagging_state 	<= trailer_data;

				when trailer_data =>
					w_ram_en						<= '1';
					w_ram_add   				<= w_ram_add + 1;
					w_ram_data					<= x"AFFEAFFE";
					align_event_size 			<= align_event_size + 1;
					if ( align_event_size(2 downto 0) + '1' = "000" ) then
						event_tagging_state 	<= trailer_set_length;
					else
						bank_size_cnt 			<= bank_size_cnt + 4;
						event_size_cnt 		<= event_size_cnt + 4;
					end if;

				when trailer_set_length =>
					w_ram_en					<= '1';
					w_ram_add   			<= w_ram_add_reg;
					-- bank length: size in bytes of the following data
					w_ram_data 				<= bank_size_cnt;
					w_ram_add_reg 			<= w_ram_add;
					bank_size_cnt 			<= (others => '0');
					event_tagging_state 	<= event_set_size;

				when event_set_size =>
					w_ram_en  				<= '1';
					w_ram_add 				<= cur_size_add;
					-- Event Data Size: The event data size contains the size of the event in bytes excluding the event header
					w_ram_data 				<= event_size_cnt;
					event_tagging_state 	<= bank_set_size;

				when bank_set_size =>
					w_ram_en 					<= '1';
					w_ram_add 					<= cur_bank_size_add;
					-- All Bank Size: Size in bytes of the following data plus the size of the bank header
					w_ram_data 					<= event_size_cnt - 8;
					event_size_cnt 			<= (others => '0');
					event_tagging_state 		<= write_tagging_fifo;

				when write_tagging_fifo =>
					w_fifo_en 					<= '1';
					w_fifo_data 				<= w_ram_add_reg;
					last_event_add				<= w_ram_add_reg;
					w_ram_add 					<= w_ram_add_reg - 1;
					event_tagging_state 		<= event_head;
					cur_bank_length_add 		<= (others => '0');
					serial_number 				<= serial_number + '1';

				when others =>
					event_tagging_state <= event_head;

			end case;
		end if;
	end if;
end process;


end architecture;
