-- midas_bank_builder.vhd
-- entity for counting event length for dma readout
-- Marius Koeppel, July 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use work.mudaq_components.all;


entity midas_bank_builder is
    port(
		i_clk_data:               in std_logic;
		i_clk_dma:           in std_logic;
		i_reset_n:           in std_logic;
		i_rx_data:           in std_logic_vector (31 downto 0);
		i_rx_datak:          in std_logic_vector (3 downto 0);
		o_bank_length:      out std_logic_vector (11 downto 0);
		o_bank_wren:		   out std_logic;
 		o_bank_data:          out std_logic_vector (31 downto 0);
		o_state_out:         out std_logic_vector(3 downto 0)--;
);
end entity midas_bank_builder;

architecture rtl of midas_bank_builder is

----------------signals---------------------
signal reset : std_logic;

signal w_ram_data : std_logic_vector(31 downto 0);
signal w_ram_add  : std_logic_vector(11 downto 0);
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(31 downto 0);
signal r_ram_add  : std_logic_vector(11 downto 0);

signal w_fifo_data      : std_logic_vector(11 downto 0);
signal w_fifo_en        : std_logic;
signal r_fifo_data      : std_logic_vector(11 downto 0);
signal r_fifo_en        : std_logic;
signal tag_fifo_empty   : std_logic;

type event_tagging_state_type is (waiting, ending);
type event_counter_state_type is (waiting, get_data, runing, ending);
signal event_counter_state : event_counter_state_type;
signal event_tagging_state : event_tagging_state_type;
signal wait_cnt : std_logic;
signal event_last_ram_add : std_logic_vector(11 downto 0);

signal rx_data_in : std_logic_vector(35 downto 0);
signal fifo_data_out : std_logic_vector(35 downto 0);
signal fifo_wrreq : std_logic;
signal fifo_empty : std_logic;
signal not_fifo_empty : std_logic;
signal fifo_data_ready : std_logic;

----------------begin midas_bank_builder------------------------
begin

reset <= not i_reset_n;
o_bank_data <= r_ram_data;

not_fifo_empty <= not fifo_empty;

e_ram : entity work.ip_ram
   port map (
		clock          => i_clk_dma,
		data           => w_ram_data,
		rdaddress      => r_ram_add,
		wraddress      => w_ram_add,
		wren           => w_ram_en,
		q              => r_ram_data--,
);

e_tagging_fifo : entity work.ip_tagging_fifo
   port map (
		data     => w_fifo_data,
		wrreq    => w_fifo_en,
		rdreq    => r_fifo_en,
		clock    => i_clk_dma,
		q    		=> r_fifo_data,
		full     => open,
		empty    => tag_fifo_empty,
		aclr     => reset--,
);

fifo : entity work.transceiver_fifo
	port map (
		data    => rx_data_in,
		wrreq   => fifo_wrreq,
		rdreq   => not_fifo_empty,
		wrclk   => i_clk_data,
		rdclk   => i_clk_dma,
		aclr    => reset,
		q       => fifo_data_out,
		rdempty => fifo_empty,
		wrfull  => open--,
);

-- write fifo
process(i_clk_data, reset_n)
begin
	if(reset_n = '0') then
		fifo_wrreq <= '0';
		rx_data_in <= (others => '0');
	elsif(rising_edge(i_clk_data)) then
		rx_data_in <= i_rx_data & i_rx_datak;
		if (i_rx_data = x"000000BC" and i_rx_datak = "0001") then
         fifo_wrreq <= '0';
        else
			fifo_wrreq <= '1';
      end if;
	end if;
end process;

-- link data to dma ram
process(i_clk_dma, reset_n)
begin
	if(reset_n = '0') then
		event_tagging_state	<= waiting;
		w_ram_en             <= '0';
		w_fifo_en            <= '0';
		w_fifo_data				<= (others => '0');
		w_ram_data				<= (others => '0');
		w_ram_add				<= (others => '1');
	elsif(rising_edge(i_clk_dma)) then
	
		w_ram_en  <= '0';
		w_fifo_en <= '0';

		if (not_fifo_empty = '1') then
			
			case event_tagging_state is
			
				when waiting =>
					if(((fifo_data_out(35 downto 30) = "111010") or (fifo_data_out(35 downto 30) = "111000")) and --mupix or mutrig data header
						fifo_data_out(11 downto 4) = x"bc" and --data header identifier
						fifo_data_out(3 downto 0) = "0001") then --data header k
						w_ram_en				  <= '1';
						w_ram_add   		  <= w_ram_add + 1;
						w_ram_data  		  <= fifo_data_out(35 downto 4);
						event_tagging_state <= ending;
					end if;
					
				when ending =>
					w_ram_en		<= '1';
					w_ram_data  <= fifo_data_out(35 downto 4);
					w_ram_add   <= w_ram_add + 1;
					if(fifo_data_out(11 downto 4) = x"9c" and --x"0000009c" and
						fifo_data_out(3 downto 0) = "0001") then
						w_fifo_en   			<= '1';
						w_fifo_data 			<= w_ram_add + 1;
						event_tagging_state 	<= waiting;
					elsif(fifo_data_out(11 downto 4) = x"EE" and -- run end / termi command
						fifo_data_out(3 downto 0) = "0001") then
						w_fifo_en   			<= '1';
						w_fifo_data 			<= w_ram_add + 1;
						event_tagging_state 	<= waiting;
					end if;
					
				when others =>
					event_tagging_state <= waiting;

			end case;
		end if;
	end if;
end process;

-- dma end of events, count events and write control
process(i_clk_dma, reset_n)
begin
	if(reset_n = '0') then
		o_state_out               <= x"0";
		r_fifo_en					<= '0';
		o_bank_wren	    		<= '0';
		wait_cnt 					<= '0';
		o_bank_length				<= (others => '0');
		r_ram_add					<= (others => '1');
		event_last_ram_add		<= (others => '1');
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
