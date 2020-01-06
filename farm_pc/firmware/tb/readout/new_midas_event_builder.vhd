-- event counter for pixel data
-- Marius Koeppel, July 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity midas_event_builder is
	generic (
		NLINKS: integer := 4--;
	);
    port(
         i_clk_data:          in std_logic;
         i_clk_dma:           in std_logic;
         i_reset_data_n:      in std_logic;
         i_reset_dma_n:       in std_logic;
         i_rx_data:     	  in std_logic_vector (NLINKS * 32 - 1 downto 0);
         i_rx_datak:          in std_logic_vector (NLINKS * 4 - 1 downto 0);
         i_wen_reg:       	  in std_logic;
         i_link_mask:         in std_logic_vector (NLINKS - 1 downto 0);
         o_all_done:          out std_logic_vector (NLINKS * 2 + NLINKS downto 0);
         o_event_wren:     	  out std_logic;
         o_endofevent: 		  out std_logic; 
         o_event_data:        out std_logic_vector (255 downto 0);
         o_state_out:         out std_logic_vector(3 downto 0)--;
);
end entity midas_event_builder;

architecture rtl of midas_event_builder is

----------------signals---------------------

-- link fifos
signal link_fifo_wren 		: std_logic_vector(NLINKS downto 0);
signal link_fifo_data 		: std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_ren 		: std_logic_vector(NLINKS downto 0);
signal link_fifo_data_out 	: std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_empty 		: std_logic_vector(NLINKS downto 0);
signal link_fifo_not_empty 	: std_logic;

-- event ram
signal w_ram_data : std_logic_vector(31 downto 0);
signal w_ram_add  : std_logic_vector(11 downto 0);
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(255 downto 0);
signal r_ram_add  : std_logic_vector(8 downto 0);

-- tagging fifo
type event_tagging_state_type is (event_head, event_num, event_tmp, event_size, bank_size, bank_flags, bank_name, bank_type, bank_length, bank_data, trailer_data, trailer_length, trailer_name, trailer_size, trailer_type, bank_size_trailer, event_size_trailer, reset_ram_add);
signal event_tagging_state : event_tagging_state_type;
signal current_link : integer;
signal cur_size_add : std_logic_vector(11 downto 0);
signal cur_bank_size_add : std_logic_vector(11 downto 0);
signal bank_length 		: std_logic_vector(NLINKS * 12 - 1 downto 0);
signal w_fifo_data      : std_logic_vector(11 downto 0);
signal w_fifo_en        : std_logic;
signal r_fifo_data      : std_logic_vector(11 downto 0);
signal r_fifo_en        : std_logic;
signal tag_fifo_empty   : std_logic;

-- midas event stuff
signal event_id 		: std_logic_vector(15 downto 0);
signal trigger_mask 	: std_logic_vector(15 downto 0);
signal serial_number 	: std_logic_vector(31 downto 0);
signal time_tmp 		: std_logic_vector(31 downto 0);
signal type_bank 		: std_logic_vector(31 downto 0);
signal flags 			: std_logic_vector(31 downto 0);










-- old

signal reset_data : std_logic;
signal reset_dma : std_logic;

-- bank
signal bank_length 		: std_logic_vector(NLINKS * 12 - 1 downto 0);
signal bank_wren	 	: std_logic_vector(NLINKS - 1 downto 0);
signal bank_data	 	: std_logic_vector(NLINKS * 36 - 1 downto 0);

signal bank_length_fifo	: std_logic_vector(NLINKS * 12 - 1 downto 0);
signal bank_length_ren	: std_logic_vector(NLINKS - 1 downto 0);
signal bank_length_empty: std_logic_vector(NLINKS - 1 downto 0);

signal bank_data_fifo	: std_logic_vector(NLINKS * 36 - 1 downto 0);
signal bank_ren	 		: std_logic_vector(NLINKS - 1 downto 0);
signal bank_empty	 	: std_logic_vector(NLINKS - 1 downto 0);

signal banks_all_done   : std_logic_vector(NLINKS * 2 - 1 downto 0);

signal mux_link			: integer range 0 to NLINKS - 1;
signal buffer_not_empty	: std_logic;

-- event ram
signal w_ram_data : std_logic_vector(31 downto 0);
signal w_ram_add  : std_logic_vector(11 downto 0);
signal last_w_ram_add : std_logic_vector(11 downto 0);
signal event_length_add : std_logic_vector(11 downto 0);
signal all_bank_add : std_logic_vector(11 downto 0);
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(255 downto 0);
signal r_ram_add  : std_logic_vector(8 downto 0);
signal trailer_size_sig : integer;
signal event_size_int : integer;

-- event tagging fifo
type event_tagging_state_type is (waiting, event_serial_number, event_tmp, event_size, event_bank_size, event_flags, bank_name, bank_type, bank_length_state, bank_data_state, trailer_data, trailer_length, trailer_name, trailer_size, trailer_type, bank_size_trailer, event_size_trailer, reset_ram_add);
signal event_tagging_state : event_tagging_state_type;
signal w_fifo_data      : std_logic_vector(11 downto 0);
signal w_fifo_en        : std_logic;
signal r_fifo_data      : std_logic_vector(11 downto 0);
signal r_fifo_en        : std_logic;
signal tag_fifo_empty   : std_logic;

-- midas event stuff
signal event_id : std_logic_vector(15 downto 0);
signal trigger_mask : std_logic_vector(15 downto 0);
signal serial_number : std_logic_vector(31 downto 0);
signal time_stamp : std_logic_vector(31 downto 0);
signal event_data_size : std_logic_vector(31 downto 0);
signal all_bank_size : std_logic_vector(31 downto 0);
signal flags : std_logic_vector(31 downto 0);

-- event readout stuff
type event_counter_state_type is (waiting, get_data, runing, ending);
signal event_counter_state : event_counter_state_type;
signal wait_cnt : std_logic;
signal event_last_ram_add : std_logic_vector(8 downto 0);

----------------begin event_counter------------------------
begin

reset_data <= not i_reset_data_n;
reset_dma <= not i_reset_dma_n;
o_event_data <= r_ram_data;
o_all_done(NLINKS - 1 + (NLINKS) * 2 + 1) <= tag_fifo_empty;

-- write link fifos
process(i_clk_data, i_reset_data_n)
begin
	if(i_reset_data_n = '0') then
		link_fifo_wren <= (others => '0');
		link_fifo_data <= (others => '0');
	elsif(rising_edge(i_clk_data)) then
		set_link_data : FOR i in 0 to NLINKS - 1 LOOP
			link_fifo_data(35 + i * 36 downto i * 36) <= i_rx_data(31 + i * 32 downto i * 32) & i_rx_datak(3 + i * 4 downto i * 4);
			if ( i_rx_data(31 + i * 32 downto i * 32) = x"000000BC" and i_rx_datak(3 + i * 4 downto i * 4) = "0001" ) then
	            link_fifo_wren(i) <= '0';
        	else
				link_fifo_wren(i) <= '1';
      		end if;
		END LOOP set_link_data;
	end if;
end process;

-- generate fifos per link
buffer_link_fifos:
FOR i in 0 to NLINKS - 1 GENERATE
	
	e_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH => 8,
        DATA_WIDTH => 36,
        DEVICE => "Arria 10"--,
	)
	port map (
		data     => link_fifo_data(35 + i * 36 downto i * 36),
		wrreq    => link_fifo_wren(i),
		rdreq    => link_fifo_ren(i),
		wrclk    => i_clk_data,
		rdclk    => i_clk_dma,
		q    	 => link_fifo_data_out(35 + i * 36 downto i * 36),
		rdempty	 => link_fifo_empty(i),
		rdusedw	 => open,
		wrfull	 => open,
		wrusedw	 => open,
		aclr     => reset_data--,
		
	);
END GENERATE buffer_link_fifos;

-- check if link fifos are empty
link_fifo_not_empty <= '1' when ( link_fifo_empty = (link_fifo_empty'range => '0') ) else '0';

-- write buffer data to ram
-- e_ram_32_256 : entity work.ip_ram_32_256
-- simulation only
e_ram_32_256 : entity work.ip_ram
generic map (
	ADDR_WIDTH_A => 12,
    ADDR_WIDTH_B => 9,
    DATA_WIDTH_A => 32,
    DATA_WIDTH_B => 256,
    DEVICE => "Arria 10"--,
)
   port map (
		address_a 	=> w_ram_add,
		address_b 	=> r_ram_add,
		clock_a 	=> i_clk_dma,
		clock_b 	=> i_clk_dma,
		data_a 		=> w_ram_data,
		data_b 		=> (others => '0'),
		wren_a 		=> w_ram_en,
		wren_b 		=> '0',
		q_a 		=> open,
		q_b 		=> r_ram_data--,
);

-- e_tagging_fifo_event : entity work.ip_tagging_fifo
-- simulation only
e_tagging_fifo_event : entity work.ip_scfifo
    generic map(
        ADDR_WIDTH => 12,
        DATA_WIDTH => 12,
        DEVICE => "Arria 10"--,
	)
	port map (
		data     		=> w_fifo_data,
		wrreq    		=> w_fifo_en,
		rdreq    		=> r_fifo_en,
		clock    		=> i_clk_dma,
		q    	 		=> r_fifo_data,
		full     		=> open,
		empty    		=> tag_fifo_empty,
		almost_empty 	=> open,
		almost_full 	=> open,
		usedw 			=> open,
		sclr     		=> reset_dma--,
);

	event_head, event_num, event_tmp, event_size, bank_size, bank_flags, bank_name, bank_type, bank_length, bank_data

-- write link data to event ram
process(i_clk_dma, i_reset_dma_n)
begin
	if( i_reset_dma_n = '0' ) then
		-- state machine singals
		event_tagging_state	<= event_head;
		current_link = 0;
		cur_size_add <= (others => '0');
		cur_bank_size_add <= (others => '0');
		cur_bank_length_add <= (others => '0');
		link_fifo_ren <= (others => '0');

		-- ram and tagging fifo write signals
		w_ram_en            <= '0';
		w_ram_data			<= (others => '0');
		w_ram_add			<= (others => '1');
		w_fifo_en           <= '0';
		w_fifo_data			<= (others => '0');

		-- midas signals
		event_id 			<= x"0001";
		trigger_mask		<= (others => '0');
		serial_number 		<= x"00000001";
		time_tmp			<= (others => '0');
		flags				<= x"00000001";
		type_bank			<= x"00000006"; -- MIDAS Bank Type TID_DWORD

	elsif( rising_edge(i_clk_dma) ) then
	
        flags				<= x"00000011";
        trigger_mask		<= (others => '0');
		event_id     		<= x"0001";
		type_bank			<= x"00000006";
		
		w_ram_en  <= '0';
		w_fifo_en <= '0';
		link_fifo_ren <= (others => '0');

		-- Note: we only do something if there is data in all fifos
		if( link_fifo_not_empty = '1' ) then

			-- count time for midas event header
			time_tmp <= time_tmp + '1';

			case event_tagging_state is

				when event_head =>
					w_ram_en	<= '1';
					w_ram_add   <= w_ram_add + 1;
					w_ram_data  <= trigger_mask & event_id;
					event_tagging_state <= event_num;

				when event_num =>
					w_ram_en	<= '1';
					w_ram_add   <= w_ram_add + 1;
					w_ram_data  <= serial_number;
					event_tagging_state <= event_tmp;

				when event_tmp =>
					w_ram_en	<= '1';
					w_ram_add   <= w_ram_add + 1;
					w_ram_data  <= time_tmp;
					event_tagging_state <= event_size;

				when event_size =>
					w_ram_en		<= '1';
					w_ram_add   	<= w_ram_add + 1;
					cur_size_add 	<= w_ram_add + 1;
					w_ram_data  	<= (others => '0');
					event_tagging_state <= bank_size;

				when bank_size =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					cur_bank_size_add 	<= w_ram_add + 1;
					w_ram_data  		<= (others => '0');
					event_tagging_state <= bank_flags;

				when bank_flags =>
					w_ram_en	<= '1';
					w_ram_add   <= w_ram_add + 1;
					w_ram_data	<= flags;
					event_tagging_state <= bank_name;

				when bank_name =>

					if ( i_link_mask(current_link) = '0' ) then
						current_link <= current_link + 1;
						if ( current_link + 1 = NLINKS ) then
							event_tagging_state <= trailer_name;
						end if;
					else
						--check for mupix or mutrig data header
						if(	
							(link_fifo_data_out(35 + current_link * 36 downto i * 36 + 30) = "111010" or link_fifo_data_out(35 + current_link * 36 downto i * 36 + 30) = "111000")
							and
							(link_fifo_data_out(11 + current_link * 36 downto i * 36 + 4) = x"bc")
							and
							(link_fifo_data_out(3 + current_link * 36 downto i * 36) = "0001")
						) then
							w_ram_en			<= '1';
							w_ram_add   		<= w_ram_add + 1;
							w_ram_data  		<= std_logic_vector(to_unsigned(current_link, w_ram_data'length));
							
							event_tagging_state <= bank_type;
						end if;

				when bank_type =>
					w_ram_en	<= '1';
					w_ram_add   <= w_ram_add + 1;
					w_ram_data	<= type_bank;
					event_tagging_state <= bank_length;

				when bank_length =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					cur_bank_length_add(11 + current_link * 12 downto current_link * 12) <= w_ram_add + 1;
					w_ram_data  		<= (others => '0');
					event_tagging_state <= bank_data;

				when bank_data
					w_ram_en	<= '1';
					w_ram_add   <= w_ram_add + 1;
					w_ram_data  <= link_fifo_data_out(35 + current_link * 36 downto i * 36 + 4);
					link_fifo_ren(current_link) <= '1';
					if(  
						(link_fifo_data_out(11 + current_link * 36 downto i * 36 + 4) = x"9c")
						and 
						(link_fifo_data_out(3 + current_link * 36 downto i * 36) = "0001")
					) then




				when event_tmp =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					w_ram_data  		<= time_stamp;
					event_tagging_state <= event_size;

				when event_size =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					event_length_add    <= w_ram_add + 1;
					w_ram_data  		<= event_data_size;
					event_tagging_state <= event_bank_size;

				when event_bank_size =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					all_bank_add        <= w_ram_add + 1;
					w_ram_data  		<= all_bank_size;
					event_tagging_state <= event_flags;

				when event_flags =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					w_ram_data  		<= flags;
					event_tagging_state <= bank_name;

				when bank_name =>
	                if ( mux_link = NLINKS ) then -- here we stop to not overflow with the mux_link
					 	event_tagging_state <= trailer_name;
						mux_link <= 0;
						serial_number <= serial_number + '1';
	                elsif ( i_link_mask(mux_link) = '0' ) then
	                    mux_link <= mux_link + 1;
	                else
			           	w_ram_en			<= '1';
	                	w_ram_add   		<= w_ram_add + 1;
			 	        w_ram_data  		<= std_logic_vector(to_unsigned(mux_link, w_ram_data'length));  -- MIDAS Bank Name
	                    event_tagging_state <= bank_type;
	                end if;

				when bank_type =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					w_ram_data  		<= x"00000006"; -- MIDAS Bank Type TID_DWORD
					event_tagging_state <= bank_length_state;

				when bank_length_state => -- check for preamble and start then
	                if ( bank_data_fifo(11 + 36 * mux_link downto mux_link * 36 + 4) = x"bc" and
						 bank_data_fifo(3 + 36 * mux_link downto mux_link * 36 ) = "0001" ) then
	                    w_ram_en					<= '1';
	                    w_ram_add   				<= w_ram_add + 1;
	                    w_ram_data                	<= std_logic_vector(to_unsigned(conv_integer(bank_length_fifo(11 + 12 * mux_link downto mux_link * 12)) * 4 + 4, w_ram_data'length)); -- length in 8 bit, +4 bcz we count from 0
	                    event_tagging_state 		<= bank_data_state;
	                    bank_length_ren(mux_link)   <= '1';
	                    bank_ren(mux_link)  <= '1';
	                end if;

				when bank_data_state =>
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					w_ram_data			<= bank_data_fifo(35 + 36 * mux_link downto mux_link * 36 + 4);
					if ( bank_data_fifo(11 + 36 * mux_link downto mux_link * 36 + 4) = x"9c" and
						 bank_data_fifo(3 + 36 * mux_link downto mux_link * 36 ) = "0001" ) then 
						if ( mux_link = NLINKS - 1 ) then
							 event_tagging_state <= trailer_name;
							 mux_link <= 0;
							 serial_number <= serial_number + '1'; 
						else
							mux_link					<= mux_link + 1;
							event_tagging_state 		<= bank_name;
						end if;
					else
						bank_length_ren(mux_link) 	<= '1';
						bank_ren(mux_link) 	<= '1';
					end if;

				when trailer_name => -- if one is in this state the midas event size will not match the size in the ram
					w_ram_en			<= '1';
	                w_ram_add   		<= w_ram_add + 1;
			 	    w_ram_data  		<= x"FFFFFFFF";  -- MIDAS Bank Name
	                event_tagging_state <= trailer_type;
	                
	            when trailer_type =>
	                w_ram_en			<= '1';
	                w_ram_add   		<= w_ram_add + 1;
			 	    w_ram_data  		<= x"00000006"; -- MIDAS Bank Type TID_DWORD
	                event_tagging_state <= trailer_length;
	                
	            when trailer_length =>
	                w_ram_en			<= '1';
	                w_ram_add   		<= w_ram_add + 1;
	                w_ram_data       <= std_logic_vector(to_unsigned(4 * (8 - conv_integer(w_ram_add + 2) mod 8), w_ram_data'length)); -- length in 8 bit of following data
					trailer_size_sig     <= 8 - conv_integer(w_ram_add + 2) mod 8 + 3;
	                if ( conv_integer(w_ram_add + 2) mod 8 = 0 ) then
						event_tagging_state <= event_size_trailer;
						last_w_ram_add <= w_ram_add + 1;
	                    w_fifo_en <= '1';
						w_fifo_data <= w_ram_add + 1;
					else
	                    event_tagging_state <= trailer_data;
					end if;
					
	            when trailer_data =>
	                w_ram_en			<= '1';
	                w_ram_add   		<= w_ram_add + 1;
	                if ( conv_integer(w_ram_add + 1) mod 8 = 0 ) then
						event_tagging_state <= event_size_trailer;
						last_w_ram_add <= w_ram_add + 1;
						w_fifo_en <= '1';
						w_fifo_data <= w_ram_add + 1;
	                end if;
	                w_ram_data			<= x"AFFEAFFE";
	                
	            when event_size_trailer =>
	                w_ram_en			<= '1';
	                w_ram_add   		<= event_length_add;
	                w_ram_data          <= std_logic_vector(to_unsigned(4 * (event_size_int + trailer_size_sig), w_ram_data'length));
					event_tagging_state <= bank_size_trailer;
	            
	            when bank_size_trailer =>
	                w_ram_en			<= '1';
	                w_ram_add   		<= all_bank_add;
	                w_ram_data          <= std_logic_vector(to_unsigned(4 * (event_size_int + trailer_size_sig - 4), w_ram_data'length));
	                event_tagging_state <= reset_ram_add;
	            
	            when reset_ram_add =>
	                w_ram_add <= last_w_ram_add;
	                event_tagging_state <= waiting;
					
				when others =>
					event_tagging_state <= waiting;

			end case;
		end if;
	end if;
end process;


-- dma end of events, count events and write control
process(i_clk_dma, i_reset_dma_n)
begin
	if(i_reset_dma_n = '0') then
		o_event_wren				<= '0';
		o_endofevent				<= '0';
		o_state_out              	<= x"0";
		r_fifo_en					<= '0';
		wait_cnt 					<= '0';
		r_ram_add					<= (others => '1');
		event_last_ram_add			<= (others => '0');
		event_counter_state 		<= waiting;	
	elsif(rising_edge(i_clk_dma)) then
	
		r_fifo_en		<= '0';
		o_event_wren	<= '0';
		wait_cnt        <= '0';
		o_endofevent    <= '0';
			
      case event_counter_state is
		
			when waiting =>
				o_state_out					<= x"A";
				if (tag_fifo_empty = '0') then
					r_fifo_en    		  	<= '1';
					event_last_ram_add  	<= r_fifo_data(11 downto 3);
					r_ram_add			  	<= r_ram_add + '1';
					event_counter_state		<= get_data;
				end if;
				
			when get_data =>
				o_state_out 		<= x"B";
				r_fifo_en    		<= '0';
				o_event_wren	<= i_wen_reg; -- todo: check this for arria10 ram
				r_ram_add			<= r_ram_add + '1';
				event_counter_state	<= runing;
				
			when runing =>
				o_state_out 	<= x"C";
				o_event_wren	<= i_wen_reg;
				if(r_ram_add = event_last_ram_add - '1') then
					event_counter_state 	<= waiting;
					o_endofevent <= '1';
				else
					r_ram_add <= r_ram_add + '1';
				end if;
				
			when ending =>
				o_state_out <= x"D";
				--if (wait_cnt = '0') then -- todo: check this for arria10 ram
    --           		wait_cnt <= '1';
    --        	else
               		event_counter_state 	<= waiting;
               		o_endofevent			<= '1';
            	--end if;
            	o_event_wren <= i_wen_reg;
				
			when others =>
				o_state_out 		<= x"E";
				event_counter_state	<= waiting;
				
		end case;
			
	end if;
end process;

end rtl;
