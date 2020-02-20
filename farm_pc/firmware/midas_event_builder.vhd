-- event counter for pixel data
-- Marius Koeppel, July 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity midas_event_builder is
    generic (
        NLINKS: integer := 4;
        LINK_FIFO_ADDR_WIDTH : integer := 10 --;
    );
    port(
        i_clk_data:         in  std_logic;
        i_clk_dma:          in  std_logic;
        i_reset_data_n:     in  std_logic;
        i_reset_dma_n:      in  std_logic;
        i_rx_data:          in  std_logic_vector (NLINKS * 32 - 1 downto 0);
        i_rx_datak:         in  std_logic_vector (NLINKS * 4 - 1 downto 0);
        i_wen_reg:          in  std_logic;
        i_link_mask_n:      in  std_logic_vector (NLINKS - 1 downto 0);
        i_get_n_words:      in std_logic_vector (31 downto 0);
        i_dmamemhalffull:   in std_logic;
        o_fifos_full:       out std_logic_vector (NLINKS downto 0); -- fifos and dmamemhalffull
        o_done:             out std_logic;
        o_all_done:         out std_logic_vector (NLINKS downto 0);
        o_event_wren:       out std_logic;
        o_endofevent:       out std_logic; 
        o_event_data:       out std_logic_vector (255 downto 0);
        o_state_out:        out std_logic_vector(3 downto 0);
        o_fifo_almost_full: out std_logic_vector(NLINKS - 1 downto 0)--;
);
end entity midas_event_builder;

architecture rtl of midas_event_builder is

----------------signals---------------------
signal reset_data : std_logic;
signal reset_dma : std_logic;

-- link fifos
signal link_fifo_wren       : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_data       : std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_ren        : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_data_out   : std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_empty      : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_full       : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_usedw      : std_logic_vector(LINK_FIFO_ADDR_WIDTH * NLINKS - 1 downto 0);

-- event ram
signal w_ram_data : std_logic_vector(31 downto 0);
signal w_ram_add  : std_logic_vector(11 downto 0);
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(255 downto 0);
signal r_ram_add  : std_logic_vector(8 downto 0);

-- tagging fifo
type event_tagging_state_type is (event_head, event_num, event_tmp, event_size, bank_size, bank_flags, bank_name, bank_type, bank_length, bank_data, bank_set_length, trailer_name, trailer_type, trailer_length, trailer_data, trailer_set_length, event_set_size, bank_set_size, write_tagging_fifo, set_algin_word);
signal event_tagging_state : event_tagging_state_type;
signal current_link : integer;
signal data_flag : std_logic;
signal cur_size_add : std_logic_vector(11 downto 0);
signal cur_bank_size_add : std_logic_vector(11 downto 0);
signal cur_bank_length_add : std_logic_vector(NLINKS * 12 - 1 downto 0);

signal w_ram_add_reg : std_logic_vector(11 downto 0);
signal last_event_add : std_logic_vector(11 downto 0);
signal align_event_size : std_logic_vector(11 downto 0);
signal w_fifo_data      : std_logic_vector(11 downto 0);
signal w_fifo_en        : std_logic;
signal r_fifo_data      : std_logic_vector(11 downto 0);
signal r_fifo_en        : std_logic;
signal tag_fifo_empty   : std_logic;

-- midas event 
signal event_id 		: std_logic_vector(15 downto 0);
signal trigger_mask 	: std_logic_vector(15 downto 0);
signal serial_number 	: std_logic_vector(31 downto 0);
signal time_tmp 		: std_logic_vector(31 downto 0);
signal type_bank 		: std_logic_vector(31 downto 0);
signal flags 			: std_logic_vector(31 downto 0);
signal bank_length_cnt : std_logic_vector(31 downto 0);

-- event readout state machine
type event_counter_state_type is (waiting, get_data, runing, skip_event);
signal event_counter_state : event_counter_state_type;
signal event_last_ram_add : std_logic_vector(8 downto 0);
signal word_counter : std_logic_vector(31 downto 0);

----------------begin event_counter------------------------
begin

reset_data <= not i_reset_data_n;
reset_dma <= not i_reset_dma_n;
o_event_data <= r_ram_data;
o_all_done(0) <= tag_fifo_empty;
o_all_done(NLINKS downto 1) <= link_fifo_empty;
o_fifos_full(NLINKS) <= i_dmamemhalffull;

-- write to link fifos
process(i_clk_data, i_reset_data_n)
begin
	if(i_reset_data_n = '0') then
		link_fifo_wren <= (others => '0');
		link_fifo_data <= (others => '0');
	elsif(rising_edge(i_clk_data)) then
		set_link_data : FOR i in 0 to NLINKS - 1 LOOP
			link_fifo_data(35 + i * 36 downto i * 36) <= i_rx_data(31 + i * 32 downto i * 32) & i_rx_datak(3 + i * 4 downto i * 4);
			if ( ( i_rx_data(31 + i * 32 downto i * 32) = x"000000BC" and i_rx_datak(3 + i * 4 downto i * 4) = "0001" ) or 
                 ( i_rx_data(31 + i * 32 downto i * 32) = x"00000000" and i_rx_datak(3 + i * 4 downto i * 4) = "1111" )                 
            ) then
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
        ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH => 36,
        DEVICE => "Arria 10"--,
    )
    port map (
        data        => link_fifo_data(35 + i * 36 downto i * 36),
        wrreq       => link_fifo_wren(i),
        rdreq       => link_fifo_ren(i),
        wrclk       => i_clk_data,
        rdclk       => i_clk_dma,
        q           => link_fifo_data_out(35 + i * 36 downto i * 36),
        rdempty     => link_fifo_empty(i),
        rdusedw     => open,
        wrfull      => o_fifos_full(i),
        wrusedw     => link_fifo_usedw(i*LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH-1 downto i*LINK_FIFO_ADDR_WIDTH),
        aclr        => reset_data--,
    );

    process(i_clk_data, i_reset_data_n)
    begin
        if(i_reset_data_n = '0') then
            o_fifo_almost_full(i)       <= '0';
        elsif(rising_edge(i_clk_data)) then
            if(link_fifo_usedw(i * LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH - 1) = '1') then
                o_fifo_almost_full(i)   <= '1';
            else 
                o_fifo_almost_full(i)   <= '0';
            end if;
        end if;
    end process;

END GENERATE buffer_link_fifos;

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

-- write link data to event ram
process(i_clk_dma, i_reset_dma_n)
begin
	if( i_reset_dma_n = '0' ) then
		-- state machine singals
		event_tagging_state	<= event_head;
		current_link <= 0;
		data_flag <= '0';
		cur_size_add <= (others => '0');
		cur_bank_size_add <= (others => '0');
		cur_bank_length_add <= (others => '0');
		link_fifo_ren <= (others => '0');
		w_ram_add_reg <= (others => '0');
		last_event_add <= (others => '0');
		align_event_size <= (others => '0');

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
		bank_length_cnt	<= (others => '0');

	elsif( rising_edge(i_clk_dma) ) then
	
        flags				<= x"00000011";
        trigger_mask		<= (others => '0');
		event_id     		<= x"0001";
		type_bank			<= x"00000006";
		
		w_ram_en  		<= '0';
		w_fifo_en 		<= '0';
		--link_fifo_ren 	<= (others => '0');

		-- Note: we only do something if there is data in all fifos
		-- KB: if( link_fifo_not_empty = '1' ) then
		-- KB: change to do something once there is data from at least one enabled link
		if( unsigned( (not link_fifo_empty) and i_link_mask_n) /= 0 ) then

			-- count time for midas event header
			time_tmp <= time_tmp + '1';

			case event_tagging_state is

				when event_head =>
					last_event_add <= w_ram_add + 1;
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
					w_ram_add_reg <= w_ram_add + 1;
					w_ram_data	<= flags;
					event_tagging_state <= bank_name;

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
                            data_flag           <= '1';
							w_ram_en			<= '1';
							w_ram_add   		<= w_ram_add_reg + 1;
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
                                w_ram_data  		<= x"34424546"; -- We should not see this !! (FEB3)
                            end if;
							--w_ram_data  		<= x"30424546"; -- one link fixed bank name + std_logic_vector(to_unsigned(current_link, 4));
							event_tagging_state <= bank_type;
                        else
                            --throw data away until a header
                            link_fifo_ren(current_link) <= '1';
						end if;
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
					link_fifo_ren(current_link) <= '1';
					event_tagging_state <= bank_data;

				when bank_data =>
					-- check again if the fifo is empty
					if ( link_fifo_empty(current_link) = '0' ) then
						w_ram_en	<= '1';
						w_ram_add   <= w_ram_add + 1;
						bank_length_cnt <= bank_length_cnt + 1;
						w_ram_data  <= link_fifo_data_out(35 + current_link * 36 downto current_link * 36 + 4);
						if(  
							(link_fifo_data_out(11 + current_link * 36 downto current_link * 36 + 4) = x"9c")
							and 
							(link_fifo_data_out(3 + current_link * 36 downto current_link * 36) = "0001")
						) then
						
							if ( bank_length_cnt(0) = '0' ) then
								event_tagging_state <= set_algin_word;
							else
								event_tagging_state <= bank_set_length;
								w_ram_add_reg <= w_ram_add + 1;
							end if;
							link_fifo_ren(current_link) <= '0';
						else
							link_fifo_ren(current_link) <= '1';
						end if;
					end if;
					
				when set_algin_word =>
					w_ram_en	<= '1';
					bank_length_cnt <= bank_length_cnt + 1;
					w_ram_add   <= w_ram_add + 1;
					w_ram_add_reg <= w_ram_add + 1;
					w_ram_data <= x"AFFEAFFE";
					event_tagging_state <= bank_set_length;

				when bank_set_length =>
					bank_length_cnt <= (others => '0');
                    w_ram_en	<= '1';
					w_ram_add   <= cur_bank_length_add(11 + current_link * 12 downto current_link * 12);
					-- bank length: size in bytes of the following data
					w_ram_data	<= std_logic_vector(to_unsigned(conv_integer(w_ram_add_reg - cur_bank_length_add(11 + current_link * 12 downto current_link * 12)) * 4, w_ram_data'length));
					if ( current_link + 1 = NLINKS ) then
						event_tagging_state <= trailer_name;
					else
						current_link <= current_link + 1;
						event_tagging_state <= bank_name;
					end if;

				when trailer_name =>
                    data_flag           <= '0';
					current_link        <= 0;
					w_ram_en			<= '1';
	                w_ram_add   		<= w_ram_add_reg + 1;
			 	    w_ram_data  		<= x"454b4146"; -- FAKE in ascii
	                event_tagging_state <= trailer_type;
	                
	            when trailer_type =>
	                w_ram_en			<= '1';
	                w_ram_add   		<= w_ram_add + 1;
			 	    w_ram_data  		<= type_bank;
	                event_tagging_state <= trailer_length;

	            when trailer_length =>
	            	w_ram_en			<= '1';
	                w_ram_add   		<= w_ram_add + 1;
	                -- here trailer length add
	                w_ram_add_reg 		<= w_ram_add + 1;
			 	    w_ram_data  		<= (others => '0');
			 	    -- write at least one AFFEAFFE
			 	    align_event_size	<= w_ram_add + 1 - last_event_add;
			 	    event_tagging_state <= trailer_data;

	            when trailer_data =>
	            	w_ram_en	<= '1';
	                w_ram_add   <= w_ram_add + 1;
	                bank_length_cnt <= bank_length_cnt + 1;
                    align_event_size <= align_event_size + 1;
	                w_ram_data	<= x"AFFEAFFE";
	            	if ( align_event_size(2 downto 0) + '1' = "000" and bank_length_cnt(0) = '0' ) then
	            		event_tagging_state <= trailer_set_length;
	            	end if;

	            when trailer_set_length =>
	            	w_ram_en		<= '1';
                    bank_length_cnt <= (others => '0');
	                w_ram_add   	<= w_ram_add_reg;
	                w_ram_add_reg 	<= w_ram_add;
	                -- bank length: size in bytes of the following data
	                w_ram_data 		<= std_logic_vector(to_unsigned((conv_integer(w_ram_add - w_ram_add_reg) - 1) * 4, w_ram_data'length));
	                event_tagging_state <= event_set_size;

	            when event_set_size =>
	            	w_ram_en  <= '1';
	            	w_ram_add <= cur_size_add;
	            	-- Event Data Size: The event data size contains the size of the event in bytes excluding the event header
	            	w_ram_data <= std_logic_vector(to_unsigned((conv_integer(w_ram_add_reg - last_event_add) - 4) * 4, w_ram_data'length));
	            	event_tagging_state <= bank_set_size;

	            when bank_set_size =>
	            	w_ram_en <= '1';
	            	w_ram_add <= cur_bank_size_add;
	            	-- All Bank Size: Size in bytes of the following data plus the size of the bank header
	            	w_ram_data <= std_logic_vector(to_unsigned((conv_integer(w_ram_add_reg - last_event_add) - 6) * 4, w_ram_data'length));
	            	event_tagging_state <= write_tagging_fifo;

	            when write_tagging_fifo =>
	            	w_fifo_en <= '1';
	            	w_fifo_data <= w_ram_add_reg;
	            	last_event_add <= w_ram_add_reg;
	            	w_ram_add <= w_ram_add_reg - 1;
	            	event_tagging_state <= event_head;
	            	cur_bank_length_add <= (others => '0');
	            	serial_number <= serial_number + '1';

				when others =>
					event_tagging_state <= event_head;

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
		o_state_out             <= x"0";
		o_done 						<= '0';
		r_fifo_en					<= '0';
		r_ram_add					<= (others => '1');
		event_last_ram_add		<= (others => '0');
		event_counter_state 		<= waiting;	
      word_counter <= (others => '0');
	elsif(rising_edge(i_clk_dma)) then
		
		o_done 			<= '0';
		r_fifo_en		<= '0';
		o_event_wren	<= '0';
		o_endofevent   <= '0';
		
		if ( i_wen_reg = '0' ) then
		    word_counter <= (others => '0');
		end if;
		
		if ( i_wen_reg = '1' and word_counter >= i_get_n_words ) then
			o_done <= '1';
		end if;
			
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
				if ( i_dmamemhalffull = '1' or ( i_get_n_words /= (i_get_n_words'range => '0') and word_counter >= i_get_n_words ) ) then
					event_counter_state <= skip_event;
				else
					o_event_wren <= i_wen_reg;
					o_endofevent <= '1'; -- begin of event
					word_counter <= word_counter + '1';
					event_counter_state	<= runing;
				end if;
				r_ram_add			<= r_ram_add + '1';
				
			when runing =>
				o_state_out 	<= x"C";
				o_event_wren <= i_wen_reg;
				word_counter <= word_counter + '1';
				if(r_ram_add = event_last_ram_add - '1') then
					event_counter_state	<= waiting;
				else
					r_ram_add <= r_ram_add + '1';
				end if;
				
			when skip_event =>
				o_state_out 	<= x"E";
				if(r_ram_add = event_last_ram_add - '1') then
					event_counter_state	<= waiting;
				else
					r_ram_add <= r_ram_add + '1';
				end if;
				
			
			when others =>
				o_state_out 		<= x"D";
				event_counter_state	<= waiting;
				
		end case;
			
	end if;
end process;

end rtl;
