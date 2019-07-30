-- event counter for pixel data
-- Marius Koeppel, July 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity event_counter is
    port(
		clk:               in std_logic;
		reset_n:           in std_logic;
		rx_data:           in std_logic_vector (31 downto 0);
		rx_datak:          in std_logic_vector (3 downto 0);
		data_ready:        in std_logic;
		dma_wen_reg:       in std_logic;
		event_length:      out std_logic_vector (7 downto 0);
		dma_data_wren:     out std_logic;
		dmamem_endofevent: out std_logic; 
 		dma_data:          out std_logic_vector (31 downto 0);
		state_out:         out std_logic_vector(3 downto 0)
);
end entity event_counter;

architecture rtl of event_counter is

component ip_ram is
   port(
   clock		: IN STD_LOGIC  := '1';
   data		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
   rdaddress	: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
   wraddress	: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
   wren		: IN STD_LOGIC  := '0';
   q		    : OUT STD_LOGIC_VECTOR (31 DOWNTO 0)
   );
end component ip_ram;

component fifo is
   port (
      CLK		: in  STD_LOGIC;
      RST		: in  STD_LOGIC;
      WriteEn	: in  STD_LOGIC;
      DataIn	: in  STD_LOGIC_VECTOR (8 - 1 downto 0);
      ReadEn	: in  STD_LOGIC;
      DataOut	: out STD_LOGIC_VECTOR (8 - 1 downto 0);
      Empty	: out STD_LOGIC;
      Full	: out STD_LOGIC
      );
end component fifo;

----------------signals---------------------
signal reset : std_logic;

signal w_ram_data : std_logic_vector(31 downto 0);
signal w_ram_add  : std_logic_vector(7 downto 0);
signal w_ram_en   : std_logic;
signal r_ram_data : std_logic_vector(31 downto 0);
signal r_ram_add  : std_logic_vector(7 downto 0);

signal w_fifo_data      : std_logic_vector(7 downto 0);
signal w_fifo_en        : std_logic;
signal r_fifo_data      : std_logic_vector(7 downto 0);
signal r_fifo_en        : std_logic;
signal tag_fifo_empty   : std_logic;

type event_tagging_state_type is (waiting, ending);
type event_counter_state_type is (waiting, get_data, runing, ending);
signal event_counter_state : event_counter_state_type;
signal event_tagging_state : event_tagging_state_type;
signal wait_cnt : std_logic;
signal event_last_ram_add : std_logic_vector(7 downto 0);

----------------begin event_counter------------------------
begin

reset <= not reset_n;
dma_data <= r_ram_data;

e_ram : component ip_ram
   port map (
		clock          => clk,
		data           => w_ram_data,
		rdaddress      => r_ram_add,
		wraddress      => w_ram_add,
		wren           => w_ram_en,
		q              => r_ram_data--,
);

e_tagging_fifo : component fifo
   port map (
		DataIn     => w_fifo_data,
		WriteEn    => w_fifo_en,
		ReadEn     => r_fifo_en,
		CLK        => clk,
		DataOut    => r_fifo_data,
		Full       => open,
		Empty      => tag_fifo_empty,
		RST        => reset--,
);

-- link data to dma ram
process(clk, reset_n)
begin
	if(reset_n = '0') then
		event_tagging_state   <= waiting;
		w_ram_en              <= '0';
		w_fifo_en             <= '0';
		w_fifo_data				 <= (others => '0');
		w_ram_data				 <= (others => '0');
		w_ram_add				 <= (others => '1');
	elsif(rising_edge(clk)) then
	
		w_ram_en  <= '0';
		w_fifo_en <= '0';

		if (data_ready = '1') then
			
			w_ram_add   <= w_ram_add + 1;
			
			case event_tagging_state is

				when waiting =>
					if(rx_data(31 downto 26) = "111010" and rx_data(7 downto 0) = x"bc" and rx_datak = "0001") then 
						w_ram_en				  <= '1';
						w_ram_data  		  <= rx_data;
						event_tagging_state <= ending;
					end if;
					
				when ending =>
					w_ram_en		<= '1';
					w_ram_data  		  <= rx_data;
					if(rx_data = x"0000009c") then
						w_fifo_data <= w_ram_add + 1;
						w_fifo_en   <= '1';
						event_tagging_state <= waiting;
					end if;
					
				when others =>
					event_tagging_state <= waiting;

			end case;
		end if;
	end if;
end process;

-- dma end of events, count events and write control
process(clk, reset_n)
begin
	if(reset_n = '0') then
		dmamem_endofevent 		<= '0';
		state_out               <= x"0";
		r_fifo_en					<= '0';
		dma_data_wren	    		<= '0';
		wait_cnt 					<= '0';
		event_length				<= (others => '0');
		r_ram_add					<= (others => '1');
		event_last_ram_add		<= (others => '1');
		event_counter_state 		<= waiting;	
	elsif(rising_edge(clk)) then
	
		dmamem_endofevent <= '0';
		r_fifo_en			<= '0';
		dma_data_wren		<= '0';
		wait_cnt          <= '0';
			
      case event_counter_state is

			when waiting =>
				state_out <= x"A";
				if (tag_fifo_empty = '0') then
					r_fifo_en    		  			<= '1';
					event_last_ram_add  			<= r_fifo_data;
					event_length					<= r_fifo_data - event_last_ram_add;
					r_ram_add			  			<= r_ram_add + '1';
					event_counter_state 			<= get_data;
				end if;
				
			when get_data =>
				state_out <= x"B";
				r_fifo_en    		  			<= '0';
				r_ram_add			  			<= r_ram_add + '1';
				event_counter_state 			<= runing;
				
			when runing =>
				state_out <= x"C";
				r_ram_add 		<= r_ram_add + '1';
				dma_data_wren	<= dma_wen_reg;
				if(r_ram_add = event_last_ram_add - '1') then
					event_counter_state 	<= ending;
				end if;
				
			when ending =>
				state_out <= x"D";
				if (wait_cnt = '0') then
               wait_cnt <= '1';
            else
               event_counter_state 	<= waiting;
               dmamem_endofevent   	<= '1';
            end if;
            dma_data_wren			<= dma_wen_reg;
				
			when others =>
				state_out <= x"E";
				event_counter_state 		<= waiting;
				
		end case;
			
	end if;
end process;

end rtl;
