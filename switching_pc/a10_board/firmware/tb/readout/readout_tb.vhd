library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 

--  A testbench has no ports.
entity readout_tb is
end readout_tb;

architecture behav of readout_tb is
  --  Declaration of the component that will be instantiated.
	component data_generator_a10_tb is
		port(
                clk:                 	in std_logic;
                reset:               	in std_logic;
                enable_pix:          	in std_logic;
                random_seed:				in std_logic_vector (15 downto 0);
                start_global_time:		in std_logic_vector(47 downto 0);
                data_pix_generated:  	out std_logic_vector(31 downto 0);
                datak_pix_generated:  	out std_logic_vector(3 downto 0);
                data_pix_ready:      	out std_logic;
                slow_down:					in std_logic_vector (31 downto 0);
                state_out:  	out std_logic_vector(3 downto 0)
			);		
	end component data_generator_a10_tb;
	
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


  --  Specifies which entity is bound with the component.
  		
  		signal clk : std_logic;
  		signal reset_n : std_logic := '1';
  		signal reset : std_logic;
  		signal enable_pix : std_logic;
  		signal slow_down : std_logic_vector(31 downto 0);
  		signal data_pix_generated : std_logic_vector(31 downto 0);
  		signal data_pix_ready : std_logic;
  		
        -- dma control
		signal dma_control_wren 		: std_logic;
		signal dma_control_counter		: std_logic_vector(31 downto 0);
		signal dma_control_prev_rdreq : std_logic_vector(31 downto 0);
		type event_tagging_state_type is (waiting, ending);
		type event_counter_state_type is (waiting, ending, get_fifo_data, start_dma, get_data, runing);
		signal event_counter_state : event_counter_state_type;
		signal event_tagging_state : event_tagging_state_type;
		signal w_ram_en	 : std_logic;
		signal w_fifo_en	 : std_logic;
		signal w_fifo_data : std_logic_vector(7 downto 0);
		signal w_ram_data	 : std_logic_vector(31 downto 0);
		signal w_ram_add	 : std_logic_vector(7 downto 0);
		signal tag_fifo_empty : std_logic;
		signal r_fifo_data : std_logic_vector(7 downto 0);
		signal r_fifo_en : std_logic;
		signal r_ram_data : std_logic_vector(31 downto 0);
		signal r_ram_add  : std_logic_vector(7 downto 0);
		signal event_last_ram_add : std_logic_vector(7 downto 0);
		signal event_length : std_logic_vector(7 downto 0);
		signal dma_data_wren : std_logic;
		signal dmamemhalffull_counter : std_logic_vector(31 downto 0);
		signal dmamemnothalffull_counter : std_logic_vector(31 downto 0);
		
		signal dmamem_endofevent : std_logic;
		signal state_out : std_logic_vector(3 downto 0);
		signal test_state : std_logic_vector(3 downto 0);
		signal running : std_logic;
		signal wait_cnt : std_logic;
		signal datak_pix_generated : std_logic_vector(3 downto 0);
  		
  		constant ckTime: 		time	:= 10 ns;
		
begin
  --  Component instantiation.
  
  reset <= not reset_n;
  enable_pix <= '1';
  slow_down <= x"00000002";--(others => '0');
 
 e_data_gen : component data_generator_a10_tb
	port map (
		clk 				=> clk,
		reset				=> reset,
		enable_pix	        => enable_pix,
		random_seed 		=> (others => '1'),
		start_global_time	=> (others => '0'),
		data_pix_generated  => data_pix_generated,
		datak_pix_generated => datak_pix_generated,
		data_pix_ready		=> data_pix_ready,
		slow_down			=> slow_down,
		state_out			=> state_out--,
    );

    
 e_ram : component ip_ram
  port map (
		clock          => clk,
		data           => w_ram_data,
		rdaddress      => r_ram_add,
		wraddress      => w_ram_add,
		wren           => w_ram_en,
		q              => r_ram_data
);


 e_tagging_fifo : component fifo
  port map (
		DataIn     => w_fifo_data,
		WriteEn      => w_fifo_en,
		ReadEn      => r_fifo_en,
		CLK     => clk,
		DataOut    => r_fifo_data,
		Full    => open,
		Empty   => tag_fifo_empty,
		RST => reset--,
);


  	-- generate the clock
	ckProc: process
	begin
		clk <= '0';
		wait for ckTime/2;
		clk <= '1';
		wait for ckTime/2;
	end process;

	inita : process
	begin
		reset_n	 <= '0';
		wait for 8 ns;
		reset_n	 <= '1';
		
		wait;
	end process inita;
	
	-- link data to dma ram
process(clk, reset_n)
begin
	if(reset_n = '0') then
		event_tagging_state 	<= waiting;
		w_ram_en				<= '0';
		w_fifo_en				<= '0';
		w_fifo_data				<= (others => '0');
		w_ram_data				<= (others => '0');
		w_ram_add				<= (others => '1'); -- '1'
	elsif(rising_edge(clk)) then
	
		w_ram_en		<= '0';
		w_fifo_en	<= '0';

		if (data_pix_ready = '1') then
			
			w_ram_add 	<= w_ram_add + 1;
			
			case event_tagging_state is

				when waiting =>
					if((data_pix_generated(31 downto 26) = "111010") and (data_pix_generated(7 downto 0) = x"bc")) then 
						w_ram_en				  <= '1';
						w_ram_data  		  <= data_pix_generated;
						event_tagging_state <= ending;
					end if;
					
				when ending =>
					w_ram_en		<= '1';
					w_ram_data  		  <= data_pix_generated;
					if(data_pix_generated = x"0000009c") then
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
variable data : line;
variable length : line;
file OutFile : TEXT open write_mode is "output_file.txt"; 
begin
	if(reset_n = '0') then
		dmamem_endofevent 		<= '0';
		test_state <= x"0";
		r_fifo_en					<= '0';
		dma_control_wren	    	<= '0';
		dma_data_wren	    		<= '0';
		running 					   <= '0';
		wait_cnt 					<= '0';
		dma_control_prev_rdreq	<= (others => '0');
		dma_control_counter 		<= (others => '0');
		event_length				<= (others => '0');
		r_ram_add					<= (others => '1'); -- '1'
		event_last_ram_add		<= (others => '1');
		event_counter_state 		<= waiting;	
	elsif(rising_edge(clk)) then
	
		dmamem_endofevent <= '0';
		r_fifo_en			<= '0';
		dma_data_wren		<= '0';
		wait_cnt          <= '0';
			
      case event_counter_state is

			when waiting =>
				test_state <= x"1";
				if (tag_fifo_empty = '0') then
					r_fifo_en    		  			<= '1';
					event_last_ram_add  			<= r_fifo_data;
					event_length					<= r_fifo_data - event_last_ram_add;
					r_ram_add			  			<= r_ram_add + '1';
					event_counter_state 			<= get_data;
				end if;
				
--			when get_fifo_data =>
--				test_state 				<= x"2";
--
--				event_counter_state 	<= get_data;

			when get_data =>
				test_state <= x"4";
				running 							<= '1';
				r_fifo_en    		  			<= '0';
				r_ram_add			  			<= r_ram_add + '1';
				event_counter_state 			<= runing;

--			when start_dma =>
--				test_state <= x"5";
--				dma_data_wren					<= '1' and writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE);
--				r_ram_add			  			<= r_ram_add + '1';
--				event_counter_state 			<= runing;
				
			when runing =>
				test_state <= x"6";
				r_ram_add 		<= r_ram_add + '1';
				dma_data_wren	<= '1';
				if(r_ram_add = event_last_ram_add - '1') then
					event_counter_state 	<= ending;
				end if;

			when ending =>
				test_state <= x"7";
				if (wait_cnt = '0') then
               wait_cnt <= '1';
            else
               event_counter_state 	<= waiting;
               dmamem_endofevent   	<= '1';
            end if;
            dma_data_wren			<= '1';


				
			when others =>
				test_state <= x"8";
				event_counter_state 		<= waiting;
				
		end case;
			
	end if;
end process;

end behav;
