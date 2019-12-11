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


  --  Specifies which entity is bound with the component.
  		
      signal clk : std_logic;
      signal clk_half : std_logic;
  	  signal reset_n : std_logic := '1';
  	  signal reset : std_logic;
      signal reset_0 : std_logic;
      signal reset_1 : std_logic;
  	  signal enable_pix : std_logic;
  	  signal slow_down : std_logic_vector(31 downto 0);
  	  signal data_pix_generated : std_logic_vector(31 downto 0);
      signal datak_pix_generated : std_logic_vector(3 downto 0);
      signal data_scifi_generated : std_logic_vector(31 downto 0);
      signal datak_scifi_generated : std_logic_vector(3 downto 0);
      signal data_tile_generated : std_logic_vector(31 downto 0);
      signal datak_tile_generated : std_logic_vector(3 downto 0);
      signal data_pix_ready : std_logic;
      signal dmamem_endofevent : std_logic;
      signal state_out_datagen : std_logic_vector(3 downto 0);
      signal state_out_eventbuilder : std_logic_vector(3 downto 0);
      signal dma_data_wren : std_logic;
      signal dma_data : std_logic_vector(255 downto 0);
      signal all_done : std_logic_vector(3 * 2 + 3 downto 0);

      signal dma_data_32_0 : std_logic_vector(31 downto 0);
      signal dma_data_32_1 : std_logic_vector(31 downto 0);
      signal dma_data_32_2 : std_logic_vector(31 downto 0);
      signal dma_data_32_3 : std_logic_vector(31 downto 0);
      signal dma_data_32_4 : std_logic_vector(31 downto 0);
      signal dma_data_32_5 : std_logic_vector(31 downto 0);
      signal dma_data_32_6 : std_logic_vector(31 downto 0);
      signal dma_data_32_7 : std_logic_vector(31 downto 0);

      signal rx_data : std_logic_vector(95 downto 0);
      signal rx_datak : std_logic_vector(11 downto 0);
  		  		
  		constant ckTime: 		time	:= 10 ns;
		
begin
  --  Component instantiation.
  
  reset <= not reset_n;
  enable_pix <= '1';
  slow_down <= (others => '0');--x"000003E8";--(others => '0');
  
  -- generate the clock
ckProc: process
begin
   clk <= '0';
   wait for ckTime/2;
   clk <= '1';
   wait for ckTime/2;
end process;

ckProc2: process
begin
   clk_half <= '0';
   wait for ckTime/4;
   clk_half <= '1';
   wait for ckTime/4;
end process;

inita : process
begin
	   reset_n	 <= '0';
	   wait for 8 ns;
	   reset_n	 <= '1';
	   wait for 20 ns;
	   enable_pix    <= '1';
	
	   wait;
end process inita;

inita0 : process
begin
     reset_n   <= '0';
     wait for 7 ns;
     reset_n   <= '1';
     wait for 20 ns;
  
     wait;
end process inita0;
 
e_data_gen_mupix : component data_generator_a10_tb
	port map (
		clk 				   => clk,
		reset				   => reset,
		enable_pix	           => enable_pix,
		random_seed 		   => (others => '1'),
		start_global_time	   => (others => '0'),
		data_pix_generated     => data_pix_generated,
		datak_pix_generated    => datak_pix_generated,
		data_pix_ready		   => data_pix_ready,
		slow_down			   => slow_down,
		state_out			   => state_out_datagen--,
);

e_data_gen_scifi : component data_generator_a10_tb
	port map (
		clk 				     => clk,
		reset				     => reset,
		enable_pix	        => enable_pix,
		random_seed 		  => (others => '1'),
		start_global_time	  => (others => '0'),
		data_pix_generated  => data_scifi_generated,
		datak_pix_generated => datak_scifi_generated,
		data_pix_ready		  => data_pix_ready,
		slow_down			  => slow_down,
		state_out			  => state_out_datagen--,
);

e_data_gen_tiles : component data_generator_a10_tb
	port map (
		clk 				     => clk,
		reset				     => reset,
		enable_pix	        => enable_pix,
		random_seed 		  => (others => '1'),
		start_global_time	  => (others => '0'),
		data_pix_generated  => data_tile_generated,
		datak_pix_generated => datak_tile_generated,
		data_pix_ready		  => data_pix_ready,
		slow_down			  => slow_down,
		state_out			  => state_out_datagen--,
);


rx_data <= data_pix_generated & data_scifi_generated & data_tile_generated;
rx_datak <= datak_pix_generated & datak_scifi_generated & datak_tile_generated;

e_midas_event_builder : entity work.midas_event_builder
  generic map (
    NLINKS => 3--;
  )
  port map(
    i_clk_data => clk,
    i_clk_dma  => clk_half,
    i_reset_data_n  => reset_n,
    i_reset_dma_n => reset_n,
    i_rx_data  => rx_data,
    i_rx_datak => rx_datak,
    i_wen_reg  => '1',
    i_link_mask => "111",
    o_all_done => all_done,
    o_event_wren => dma_data_wren,
    o_endofevent => dmamem_endofevent,
    o_event_data => dma_data,
    o_state_out => state_out_eventbuilder--,
);

  dma_data_32_0 <= dma_data(31 downto 0);
  dma_data_32_1 <= dma_data(63 downto 32);
  dma_data_32_2 <= dma_data(95 downto 64);
  dma_data_32_3 <= dma_data(127 downto 96);
  dma_data_32_4 <= dma_data(159 downto 128);
  dma_data_32_5 <= dma_data(191 downto 160);
  dma_data_32_6 <= dma_data(223 downto 192);
  dma_data_32_7 <= dma_data(255 downto 224);

end behav;
