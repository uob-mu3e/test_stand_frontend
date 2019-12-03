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
         i_reset_n:           in std_logic;
         i_rx_data:     	  in std_logic_vector (NLINKS * 32 - 1 downto 0);
         i_rx_datak:          in std_logic_vector (NLINKS * 4 - 1 downto 0);
         i_dma_wen_reg:       in std_logic;
         o_dma_data_wren:     out std_logic;
         o_dmamem_endofevent: out std_logic; 
         o_dma_data:          out std_logic_vector (255 downto 0);
         o_state_out:         out std_logic_vector(3 downto 0)--;
);
end entity midas_event_builder;

architecture rtl of midas_event_builder is

----------------signals---------------------
signal reset : std_logic;

-- bank
signal bank_length 		: std_logic_vector(NLINKS * 12 - 1 downto 0);
signal bank_wren	 	: std_logic_vector(NLINKS downto 0);
signal bank_data	 	: std_logic_vector(NLINKS * 32 - 1 downto 0);

signal bank_length_fifo	: std_logic_vector(NLINKS * 12 - 1 downto 0);
signal bank_length_ren	: std_logic_vector(NLINKS downto 0);
signal bank_length_empty: std_logic_vector(NLINKS downto 0);

signal bank_data_fifo	: std_logic_vector(NLINKS * 32 - 1 downto 0);
signal bank_ren	 		: std_logic_vector(NLINKS downto 0);
signal bank_empty	 	: std_logic_vector(NLINKS downto 0);

signal mux_link			: integer range 0 to NLINKS - 1;
signal buffer_not_empty	: std_logic_vector(NLINKS downto 0);

signal bank_header : std_logic_vector(63 downto 0) := x"AAAAAAAA" & -- Bank Name
													  x"AAAAAAAA" ; -- Bank Type

-- event ram
signal w_ram_data : std_logic_vector(31 downto 0);
signal w_ram_add  : std_logic_vector(11 downto 0);
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(255 downto 0);
signal r_ram_add  : std_logic_vector(8 downto 0);

-- event tagging figo
type event_tagging_state_type is (waiting, mupix, scifi_0, scifi_1, tile_0, tile_1);
signal event_tagging_state : event_tagging_state_type;
signal w_fifo_data      : std_logic_vector(11 downto 0);
signal w_fifo_en        : std_logic;
signal r_fifo_data      : std_logic_vector(11 downto 0);
signal r_fifo_en        : std_logic;
signal tag_fifo_empty   : std_logic;


----------------begin event_counter------------------------
begin

reset <= not reset_n;

-- generate buffer
buffer_banks:
FOR i in 0 to NLINKS-1 GENERATE	
	e_bank : entity work.midas_bank_builder
	    port map(
			i_clk_data          => i_clk_data,	
			i_clk_dma			=> i_clk_dma,
			i_reset_n			=> reset_n,
			i_rx_data 			=> i_rx_data(31 + 32 * i downto i * 32),
			i_rx_datak 			=> i_rx_datak(3 + 4 * i downto i * 4),
			o_bank_length 		=> bank_length(11 + 12 * i downto i * 12),
			o_bank_wren 		=> bank_wren(i),
			o_bank_data 		=> bank_data(31 + 32 * i downto i * 32),
	 		o_state_out 		=> open--,
	);

	e_receiver_fifo_32 : entity work._receiver_fifo_32
		port map(
			data    => bank_data(31 + 32 * i downto i * 32),
			wrreq   => bank_wren(i),
			rdreq   => bank_ren(i),
			wrclk   => i_clk_dma,
			rdclk   => i_clk_dma,
			aclr    => reset,
			q       => bank_data_fifo(31 + 32 * i downto i * 32),
			rdempty => bank_empty(i),
			wrfull  => open--,
	);

	e_bank_length : entity work.ip_tagging_fifo
   		port map (
			data     => bank_length(11 + 12 * i downto i * 12),
			wrreq    => bank_wren(i),
			rdreq    => bank_length_ren(i),
			clock    => i_clk_dma,
			q    	 => bank_length_fifo(11 + 12 * i downto i * 12),
			full     => open,
			empty    => bank_length_empty(i),
			aclr     => reset--,
	);
END GENERATE buffer_banks;

-- check if buffer is empty
process(i_clk_dma, reset_n)
variable not_empty : std_logic;
begin
	if( reset_n = '0' ) then
		buffer_not_empty <= '0';
		not_empty := '0';
	elsif( rising_edge(i_clk_dma) ) then
		buffer_not_empty <= not_empty;
		not_empty := '0';
		l_empty : FOR i in 0 to NLINKS LOOP
			not_empty <= not_empty and bank_empty(i);
		END LOOP l_empty;
	end if;
end process;

-- write buffer data to ram
e_ram_32_256 : entity work.ip_ram_32_256
   port map (
		clock          => i_clk_dma,
		data           => w_ram_data,
		rdaddress      => r_ram_add,
		wraddress      => w_ram_add,
		wren           => w_ram_en,
		q              => r_ram_data--,
);

e_tagging_fifo_event : entity work.ip_tagging_fifo
   port map (
		data     => w_fifo_data,
		wrreq    => w_fifo_en,
		rdreq    => r_fifo_en,
		clock    => i_clk_dma,
		q    	 => r_fifo_data,
		full     => open,
		empty    => tag_fifo_empty,
		aclr     => reset--,
);

-- write banks to event ram
process(i_clk_dma, reset_n)
begin
	if( reset_n = '0' ) then
		event_tagging_state	<= waiting;
		w_ram_en            <= '0';
		w_fifo_en           <= '0';
		mux_link			<= 0;
		w_fifo_data			<= (others => '0');
		w_ram_data			<= (others => '0');
		bank_ren			<= (others => '0');
		bank_length_ren		<= (others => '0');
		w_ram_add			<= (others => '1');
	elsif( rising_edge(i_clk_dma) ) then
	
		w_ram_en  <= '0';
		w_fifo_en <= '0';

		case event_tagging_state is

			when waiting =>
				if( buffer_not_empty = '1' ) then
					w_ram_en			<= '1';
					w_ram_add   		<= w_ram_add + 1;
					w_ram_data  		<= bank_header(63 downto 32);
					event_tagging_state <= header;
				end if;
	
			when header =>
				w_ram_en			<= '1';
				w_ram_add   		<= w_ram_add + 1;
				w_ram_data  		<= bank_header(31 downto 0);
				event_tagging_state <= length_state;

			when length_state =>
				w_ram_en					<= '1';
				w_ram_add   				<= w_ram_add + 1;
				bank_length_ren(mux_link) 	<= '1';
				w_ram_data					<= bank_length_fifo(11 + 12 * mux_link downto mux_link * 12);
				event_tagging_state 		<= read_data;

			when read_data =>
				w_ram_en			<= '1';
				w_ram_add   		<= w_ram_add + 1;
				bank_ren(mux_link) 	<= '1';
				w_ram_data			<= bank_data_fifo(31 + 32 * mux_link downto mux_link * 32);
				if ( bank_data_fifo(31 + 32 * mux_link downto mux_link * 32) = x"0000009c" ) then -- TODO: need to have also datak 
					bank_length_ren(mux_link) 	<= '0';
					if ( mux_link = NLINKS - 1 ) then
						 event_tagging_state <= waiting;
						 mux_link := 0;
					else
						mux_link					<= mux_link + 1;
						event_tagging_state 		<= header;
					end if;
					w_fifo_en   <= '1';
					w_fifo_data <= w_ram_add + 1;
				else
					bank_length_ren(mux_link) 	<= '1';
				end if;

			when others =>
				event_tagging_state <= waiting;

		end case;
	end if;
end process;

-- dma end of events, count events and write control
process(i_clk_dma, reset_n)
begin
	if(reset_n = '0') then
		o_state_out              	<= x"0";
		r_fifo_en					<= '0';
		o_event_wren    			<= '0';
		wait_cnt 					<= '0';
		o_event_length				<= (others => '0');
		r_ram_add					<= (others => '1');
		event_last_ram_add			<= (others => '1');
		event_counter_state 		<= waiting;	
	elsif(rising_edge(i_clk_dma)) then
	
		r_fifo_en			<= '0';
		o_bank_wren		<= '0';
		wait_cnt          <= '0';
			
      case event_counter_state is
		
			when waiting =>
				o_state_out					<= x"A";
				if (tag_fifo_empty = '0') then
					r_fifo_en    		  	<= '1';
					event_last_ram_add  	<= r_fifo_data;
					o_bank_length			<= r_fifo_data - event_last_ram_add;
					r_ram_add			  	<= r_ram_add + '1';
					event_counter_state	<= get_data;
				end if;
				
			when get_data =>
				o_state_out 				<= x"B";
				r_fifo_en    		  	<= '0';
				r_ram_add			  	<= r_ram_add + '1';
				event_counter_state	<= runing;
				
			when runing =>
				o_state_out 		<= x"C";
				r_ram_add 		<= r_ram_add + '1';
				o_bank_wren	<= '1';
				if(r_ram_add = event_last_ram_add - '1') then
					event_counter_state 	<= ending;
				end if;
				
			when ending =>
				o_state_out <= x"D";
				if (wait_cnt = '0') then
               wait_cnt <= '1';
            else
               event_counter_state 	<= waiting;
            end if;
            o_bank_wren	<= '1';
				
			when others =>
				o_state_out 				<= x"E";
				event_counter_state	<= waiting;
				
		end case;
			
	end if;
end process;

end rtl;
