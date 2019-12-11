-- midas_bank_builder.vhd
-- entity for counting event length for dma readout
-- Marius Koeppel, July 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.daq_constants.all;


entity midas_bank_builder is
    port(
		i_clk_data:               in std_logic;
		i_clk_dma:           in std_logic;
		i_reset_data_n:           in std_logic;
        i_reset_dma_n:           in std_logic;
		i_rx_data:           in std_logic_vector (31 downto 0);
		i_rx_datak:          in std_logic_vector (3 downto 0);
        o_all_done:      out std_logic_vector (1 downto 0);
		o_bank_length:      out std_logic_vector (11 downto 0);
		o_bank_wren:		   out std_logic;
 		o_bank_data:          out std_logic_vector (35 downto 0);
		o_state_out:         out std_logic_vector(3 downto 0)--;
);
end entity midas_bank_builder;

architecture rtl of midas_bank_builder is

----------------signals---------------------
signal reset_data : std_logic;
signal reset_dma : std_logic;

signal w_ram_data : std_logic_vector(35 downto 0);
signal w_ram_add  : std_logic_vector(11 downto 0);
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(35 downto 0);
signal r_ram_add  : std_logic_vector(11 downto 0);

signal w_fifo_data      : std_logic_vector(11 downto 0);
signal w_fifo_en        : std_logic;
signal r_fifo_data      : std_logic_vector(11 downto 0);
signal r_fifo_en        : std_logic;
signal tag_fifo_empty   : std_logic;

type bank_tagging_state_type is (waiting, ending);
type bank_counter_state_type is (waiting, get_data, runing, ending);
signal bank_counter_state : bank_counter_state_type;
signal bank_tagging_state : bank_tagging_state_type;
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

reset_data <= not i_reset_data_n;
reset_dma <= not i_reset_dma_n;
o_bank_data <= r_ram_data;

not_fifo_empty <= not fifo_empty;

o_all_done(0) <= tag_fifo_empty; -- tag fifo
o_all_done(1) <= fifo_empty; -- data fifo


-- write buffer data to ram
-- e_ram : entity work.ip_ram_36
-- simulation only
e_ram : entity work.ip_ram
generic map (
    DEVICE => "Arria X",
	ADDR_WIDTH_A => 12,
    ADDR_WIDTH_B => 12,
    DATA_WIDTH_A => 36,
    DATA_WIDTH_B => 36--,
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

-- e_tagging_fifo : entity work.ip_tagging_fifo
-- simulation only
e_bank_length : entity work.ip_scfifo
    generic map(
        DEVICE => "Arria X",
        ADDR_WIDTH => 12,
        DATA_WIDTH => 12--,
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

--e_fifo : entity work.ip_receiver_fifo_36
-- simulation only
e_fifo : entity work.ip_dcfifo
    generic map(
        DEVICE => "Arria X",
        ADDR_WIDTH => 8,
        DATA_WIDTH => 36--,
)
-- simulation only
		port map (
		data     => rx_data_in,
		wrreq    => fifo_wrreq,
		rdreq    => not_fifo_empty,
		wrclk   => i_clk_data,
		rdclk   => i_clk_dma,
		q    	 => fifo_data_out,

		-- simulation only
		rdempty	 => fifo_empty,
		rdusedw	 => open,
		wrfull	 => open,
		wrusedw	 => open,
		aclr     => reset_data--,
		-- simulation only
		--full     => open,
		--empty    => fifo_empty,
);

-- write fifo
process(i_clk_data, i_reset_data_n)
begin
	if(i_reset_data_n = '0') then
		fifo_wrreq <= '0';
		rx_data_in <= (others => '0');
	elsif(rising_edge(i_clk_data)) then
		rx_data_in <= i_rx_data & i_rx_datak;
		if ( (i_rx_data = x"000000BC" and i_rx_datak = "0001") or 
            (i_rx_data = RUN_END and i_rx_datak = "0001") or 
            (i_rx_data = run_prep_acknowledge and i_rx_datak = "0001") ) then
            fifo_wrreq <= '0';
        else
			fifo_wrreq <= '1';
      end if;
	end if;
end process;

-- link data to dma ram
process(i_clk_dma, i_reset_dma_n)
begin
	if(i_reset_dma_n = '0') then
		bank_tagging_state	<= waiting;
		w_ram_en             <= '0';
		w_fifo_en            <= '0';
		w_fifo_data				<= (others => '0');
		w_ram_data				<= (others => '0');
		w_ram_add				<= (others => '1');
	elsif(rising_edge(i_clk_dma)) then
	
		w_ram_en  <= '0';
		w_fifo_en <= '0';

		if (not_fifo_empty = '1') then
			
			case bank_tagging_state is
			
				when waiting =>       
					if(((fifo_data_out(35 downto 30) = "111010") or (fifo_data_out(35 downto 30) = "111000")) and --mupix or mutrig data header
						fifo_data_out(11 downto 4) = x"bc" and --data header identifier
						fifo_data_out(3 downto 0) = "0001") then --data header k
						w_ram_en				  <= '1';
						w_ram_add   		  <= w_ram_add + 1;
						w_ram_data  		  <= fifo_data_out;
						bank_tagging_state <= ending;
					end if;
					
				when ending =>
					w_ram_en		<= '1';
					w_ram_data  <= fifo_data_out;
					w_ram_add   <= w_ram_add + 1;
					if( (fifo_data_out(11 downto 4) = x"9c" and --x"0000009c" and
						fifo_data_out(3 downto 0) = "0001") ) then
						w_fifo_en   			<= '1';
						w_fifo_data 			<= w_ram_add + 1;
						bank_tagging_state 	<= waiting;
					end if;
					
				when others =>
					bank_tagging_state <= waiting;

			end case;
		end if;
	end if;
end process;

-- dma end of events, count events and write control
process(i_clk_dma, i_reset_dma_n)
begin
	if(i_reset_dma_n = '0') then
		o_state_out               <= x"0";
		r_fifo_en					<= '0';
		o_bank_wren	    		<= '0';
		wait_cnt 					<= '0';
		o_bank_length				<= (others => '0');
		r_ram_add					<= (others => '1');
		event_last_ram_add		<= (others => '0');
		bank_counter_state 		<= waiting;	
	elsif(rising_edge(i_clk_dma)) then
	
		r_fifo_en			<= '0';
		o_bank_wren		<= '0';
		wait_cnt          <= '0';
			
      case bank_counter_state is
		
			when waiting =>
				o_state_out					<= x"A";
				if (tag_fifo_empty = '0') then
					r_fifo_en    		  	<= '1';
					event_last_ram_add  	<= r_fifo_data;
					o_bank_length			<= r_fifo_data - event_last_ram_add;
					r_ram_add			  	<= r_ram_add + '1';
					bank_counter_state	<= get_data;
				end if;
				
			when get_data =>
				o_state_out 				<= x"B";
				r_fifo_en    		  	<= '0';
				o_bank_wren	<= '1'; -- todo: check this for arria10 ram
				r_ram_add			  	<= r_ram_add + '1';
				bank_counter_state	<= runing;
				
			when runing =>
				o_state_out 		<= x"C";
				r_ram_add 		<= r_ram_add + '1';
				o_bank_wren	<= '1';
				if(r_ram_add = event_last_ram_add - '1') then
					bank_counter_state 	<= ending;
				end if;
				
			when ending =>
				o_state_out <= x"D";
				--if (wait_cnt = '0') then -- todo: check this for arria10 ram
    --           		wait_cnt <= '1';
    --        	else
               		bank_counter_state 	<= waiting;
            	--end if;
            	o_bank_wren	<= '1';
				
			when others =>
				o_state_out 				<= x"E";
				bank_counter_state	<= waiting;
				
		end case;
			
	end if;
end process;

end rtl;
