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
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(255 downto 0);
signal r_ram_add  : std_logic_vector(8 downto 0);

-- event tagging fifo
type event_tagging_state_type is (waiting, event_serial_number, event_tmp, event_size, event_bank_size, event_flags, bank_name, bank_type, bank_length_state, bank_data_state, trailer);
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

-- generate buffer
buffer_banks:
FOR i in 0 to NLINKS - 1 GENERATE	

    o_all_done(1 + 2 * i downto i * 2) <= banks_all_done(1 + 2 * i downto i * 2);
    o_all_done(i + (NLINKS) * 2) <= bank_empty(i);

	e_bank : entity work.midas_bank_builder
	    port map(
			i_clk_data          => i_clk_data,
			i_clk_dma			=> i_clk_dma,
			i_reset_data_n		=> i_reset_data_n,
            i_reset_dma_n		=> i_reset_dma_n,
			i_rx_data 			=> i_rx_data(31 + 32 * i downto i * 32),
			i_rx_datak 			=> i_rx_datak(3 + 4 * i downto i * 4),
            o_all_done          => banks_all_done(1 + 2 * i downto i * 2),
			o_bank_length 		=> bank_length(11 + 12 * i downto i * 12),
			o_bank_wren 		=> bank_wren(i),
			o_bank_data 		=> bank_data(35 + 36 * i downto i * 36),
	 		o_state_out 		=> open--,
	);

	-- e_receiver_fifo_32 : entity work.ip_receiver_fifo_36
	-- simulation only
	e_receiver_fifo_32 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH => 8,
        DATA_WIDTH => 36,
        DEVICE => "Arria 10"--,
    )
	-- simulation only
		port map(
			data    => bank_data(35 + 36 * i downto i * 36),
			wrreq   => bank_wren(i),
			rdreq   => bank_ren(i),
			wrclk   => i_clk_dma,
			rdclk   => i_clk_dma,
			aclr    => reset_dma,
			q       => bank_data_fifo(35 + 36 * i downto i * 36),
			rdempty => bank_empty(i),
			-- simulation only
			wrusedw => open,
			rdusedw => open,
			-- simulation only
			wrfull  => open--,
	);

	-- e_bank_length : entity work.ip_tagging_fifo
	-- simulation only
	e_bank_length : entity work.ip_scfifo
	    generic map(
        ADDR_WIDTH => 8,
        DATA_WIDTH => 12,
        DEVICE => "Arria 10"--,
    )
	-- simulation only
   		port map (
			data     => bank_length(11 + 12 * i downto i * 12),
			wrreq    => bank_wren(i),
			rdreq    => bank_length_ren(i),
			clock    => i_clk_dma,
			q    	 => bank_length_fifo(11 + 12 * i downto i * 12),
			full     => open,
			empty    => bank_length_empty(i),
			-- simulation only
			almost_empty => open,
			almost_full => open,
			usedw => open,
			sclr     => reset_dma--,
			-- simulation only
			--aclr     => reset--,
	);
END GENERATE buffer_banks;

-- check if buffer is empty
process(i_clk_dma, i_reset_dma_n)
variable not_empty : std_logic;
begin
	if( i_reset_dma_n = '0' ) then
		buffer_not_empty <= '0';
		not_empty := '1';
	elsif( rising_edge(i_clk_dma) ) then
		buffer_not_empty <= not not_empty;
		l_empty : FOR i in 0 to NLINKS - 1 LOOP
			not_empty := not_empty and bank_empty(i);
		END LOOP l_empty;
	end if;
end process;

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
		--clock          => i_clk_dma,
		--data           => w_ram_data,
		--rdaddress      => r_ram_add,
		--wraddress      => w_ram_add,
		--wren           => w_ram_en,
		--q              => r_ram_data--,
		address_a => w_ram_add,
		address_b => r_ram_add,
		clock_a => i_clk_dma,
		clock_b => i_clk_dma,
		data_a => w_ram_data,
		data_b => (others => '0'),
		wren_a => w_ram_en,
		wren_b => '0',
		q_a => open,
		q_b => r_ram_data--,
);

-- e_tagging_fifo_event : entity work.ip_tagging_fifo
-- simulation only
e_tagging_fifo_event : entity work.ip_scfifo
    generic map(
        ADDR_WIDTH => 12,
        DATA_WIDTH => 12,
        DEVICE => "Arria 10"--,
	)
-- simulation only
	port map (
		data     => w_fifo_data,
		wrreq    => w_fifo_en,
		rdreq    => r_fifo_en,
		clock    => i_clk_dma,
		q    	 => r_fifo_data,
		full     => open,
		empty    => tag_fifo_empty,
		-- simulation only
		almost_empty => open,
		almost_full => open,
		usedw => open,
		sclr     => reset_dma--,
		-- simulation only
		--aclr     => reset--,
);

-- write banks to event ram
process(i_clk_dma, i_reset_dma_n)
	variable count_size : integer range 0 to 2**12*NLINKS;
begin
	if( i_reset_dma_n = '0' ) then
		event_tagging_state	<= waiting;
		w_ram_en            <= '0';
		w_fifo_en           <= '0';
		mux_link			<= 0;
		w_fifo_data			<= (others => '0');
		w_ram_data			<= (others => '0');
		bank_ren			<= (others => '0');
		bank_length_ren		<= (others => '0');
		w_ram_add			<= (others => '1');

		count_size 			:= 0;
		event_id 			<= (others => '0');
		trigger_mask		<= x"FFFF";
		serial_number 		<= x"BABEBABE";
		time_stamp			<= (others => '0');
		event_data_size		<= (others => '0');
		all_bank_size		<= (others => '0');
		flags				<= x"CAFEAFFE";

	elsif( rising_edge(i_clk_dma) ) then
	
		w_ram_en  <= '0';
		w_fifo_en <= '0';
		count_size := 0;
		time_stamp <= time_stamp + '1'; -- TODO: this counts now all the time
		bank_ren <= (others => '0');
		bank_length_ren <= (others => '0');

		case event_tagging_state is

			when waiting =>
				if( buffer_not_empty = '1' ) then
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					w_ram_data  		<= event_id & trigger_mask;
					l_count_size : FOR i in 0 to NLINKS - 1 LOOP
						count_size := count_size + conv_integer(bank_length_fifo(11 + 12 * i downto i * 12));
					END LOOP l_count_size;
					event_data_size		<= std_logic_vector(to_unsigned(3 * NLINKS + 6 + count_size, event_data_size'length)); -- length in 32 bit
					all_bank_size		<= std_logic_vector(to_unsigned(3 * NLINKS + count_size, event_data_size'length)); -- length in 32 bit
					event_tagging_state <= event_serial_number;
				end if;

			when event_serial_number =>
				w_ram_en			<= '1';
				w_ram_add   		<= w_ram_add + 1;
				w_ram_data  		<= serial_number;
				event_tagging_state <= event_tmp;

			when event_tmp =>
				w_ram_en			<= '1';
				w_ram_add   		<= w_ram_add + 1;
				w_ram_data  		<= time_stamp;
				event_tagging_state <= event_size;

			when event_size =>
				w_ram_en			<= '1';
				w_ram_add   		<= w_ram_add + 1;
				w_ram_data  		<= event_data_size;
				event_tagging_state <= event_bank_size;

			when event_bank_size =>
				w_ram_en			<= '1';
				w_ram_add   		<= w_ram_add + 1;
				w_ram_data  		<= all_bank_size;
				event_tagging_state <= event_flags;

			when event_flags =>
				w_ram_en			<= '1';
				w_ram_add   		<= w_ram_add + 1;
				w_ram_data  		<= flags;
				event_tagging_state <= bank_name;

			when bank_name =>
                if ( mux_link = NLINKS ) then -- here we stop to not overflow with the mux_link
                    if ( conv_integer(w_ram_add + 2) mod 8 = 0 ) then
						event_tagging_state <= waiting;
					 	w_fifo_en   <= '1';
						w_fifo_data <= w_ram_add + 1;
					else
					 	event_tagging_state <= trailer;
					end if;
					mux_link <= 0;
					event_id <= event_id + '1';
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
				w_ram_data  		<= std_logic_vector(to_unsigned(mux_link, w_ram_data'length)); -- MIDAS Bank Type
				event_tagging_state <= bank_length_state;

			when bank_length_state => -- check for preamble and start then
                if ( bank_data_fifo(11 + 36 * mux_link downto mux_link * 36 + 4) = x"bc" and
					 bank_data_fifo(3 + 36 * mux_link downto mux_link * 36 ) = "0001" ) then
                    w_ram_en					<= '1';
                    w_ram_add   				<= w_ram_add + 1;
                    w_ram_data(11 downto 0)  	<= bank_length_fifo(11 + 12 * mux_link downto mux_link * 12);
                    w_ram_data(31 downto 12)  	<= (others => '0');
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
						 if ( conv_integer(w_ram_add + 2) mod 8 = 0 ) then
						 	event_tagging_state <= waiting;
						 	w_fifo_en   <= '1';
							w_fifo_data <= w_ram_add + 1;
						 else
						 	event_tagging_state <= trailer;
						 end if;
						 mux_link <= 0;
						 event_id <= event_id + '1';
					else
						mux_link					<= mux_link + 1;
						event_tagging_state 		<= bank_name;
					end if;
				else
					bank_length_ren(mux_link) 	<= '1';
					bank_ren(mux_link) 	<= '1';
				end if;

			when trailer => -- if one is in this state the midas event size will not match the size in the ram
				if ( conv_integer(w_ram_add + 1) mod 8 = 0 ) then
					event_tagging_state <= waiting;
					w_fifo_en <= '1';
					w_fifo_data <= w_ram_add + 1;
				else
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					w_ram_data			<= x"AFFEAFFE";
				end if;

			when others =>
				event_tagging_state <= waiting;

		end case;
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
