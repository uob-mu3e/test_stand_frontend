library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 


--  A testbench has no ports.
entity readout_tb is
end entity;

architecture behav of tree_tb is
  --  Specifies which entity is bound with the component.
  		
      signal clk : std_logic;
      signal reset_n : std_logic := '1';
      signal enable_pix : std_logic;
  	  signal slow_down : std_logic_vector(31 downto 0);
  	  signal data_pix_generated_0 : std_logic_vector(31 downto 0);
      signal datak_pix_generated_0 : std_logic_vector(3 downto 0);
      signal data_pix_generated_1 : std_logic_vector(31 downto 0);
      signal datak_pix_generated_1 : std_logic_vector(3 downto 0);
      signal data_pix_generated_2 : std_logic_vector(31 downto 0);
      signal datak_pix_generated_2 : std_logic_vector(3 downto 0);
      signal data_pix_generated_3 : std_logic_vector(31 downto 0);
      signal datak_pix_generated_3 : std_logic_vector(3 downto 0);
      signal alginment_tree_data : std_logic_vector(32 * 32 - 1 downto 0);
      signal alginment_tree_datak : std_logic_vector(32 * 3 - 1 downto 0);
  		constant ckTime: 		time	:= 10 ns;
		
begin
  --  Component instantiation.
  
  reset <= not reset_n;
  enable_pix <= '1';
  slow_down <= x"00000002";--(others => '0');
  
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
	   wait for 20 ns;
	   enable_pix    <= '1';
	
	   wait;
end process inita;
 
e_data_gen_0 : entity work.data_generator_mupix
	port map (
		clk 				       => clk,
		reset				       => reset,
		enable_pix	       => enable_pix,
    i_dma_half_full    => '0',
		random_seed 		   => (others => '1'),
		start_global_time	 => (others => '0'),
		data_pix_generated => data_pix_generated_0,
		datak_pix_generated=> datak_pix_generated_0,
		data_pix_ready		 => open,
		slow_down			     => slow_down,
		state_out			     => open--,
);

e_data_gen_1 : entity work.data_generator_mupix
  port map (
    clk                => clk,
    reset              => reset,
    enable_pix         => enable_pix,
    i_dma_half_full    => '0',
    random_seed        => (others => '1'),
    start_global_time  => (others => '0'),
    data_pix_generated => data_pix_generated_1,
    datak_pix_generated=> datak_pix_generated_1,
    data_pix_ready     => open,
    slow_down          => slow_down,
    state_out          => open--,
);

e_data_gen_2 : entity work.data_generator_mupix
  port map (
    clk                => clk,
    reset              => reset,
    enable_pix         => enable_pix,
    i_dma_half_full    => '0',
    random_seed        => (others => '1'),
    start_global_time  => (others => '0'),
    data_pix_generated => data_pix_generated_2,
    datak_pix_generated=> datak_pix_generated_2,
    data_pix_ready     => open,
    slow_down          => slow_down,
    state_out          => open--,
);

e_data_gen_3 : entity work.data_generator_mupix
  port map (
    clk                => clk,
    reset              => reset,
    enable_pix         => enable_pix,
    i_dma_half_full    => '0',
    random_seed        => (others => '1'),
    start_global_time  => (others => '0'),
    data_pix_generated => data_pix_generated_3,
    datak_pix_generated=> datak_pix_generated_3,
    data_pix_ready     => open,
    slow_down          => slow_down,
    state_out          => open--,
);

alginment_tree_data  <= data_pix_generated_0 & data_scifi_generated & data_tile_generated & data_tile_generated2 & x"000000BC";--data_tile_generated3;
alginment_tree_datak <= datak_pix_generated & datak_scifi_generated & datak_tile_generated & datak_tile_generated2 & "0001";--datak_tile_generated3;

e_midas_event_builder : entity work.midas_event_builder
  generic map (
    NLINKS => 5--;
  )
  port map(
    i_clk_data => clk,
    i_clk_dma  => clk_half,
    i_reset_data_n  => reset_n,
    i_reset_dma_n => reset_n,
    i_rx_data  => rx_data,
    i_rx_datak => rx_datak,
    i_wen_reg  => '1',
    i_link_mask_n => "01011",
    i_get_n_words  => x"00000100",
    i_dmamemhalffull => '0',
    o_fifos_full => open,
    o_done  => open,
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

end architecture;
