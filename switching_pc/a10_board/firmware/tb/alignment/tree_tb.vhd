library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 


--  A testbench has no ports.
entity tree_tb is
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
      signal alginment_tree_datak : std_logic_vector(32 * 4 - 1 downto 0);
  		constant ckTime: 		time	:= 10 ns;
		
begin
  --  Component instantiation.
  
  enable_pix <= '1';
  slow_down <= (others => '0');--x"00000002";--(others => '0');
  
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
		i_clk 				     => clk,
		i_reset_n				   => reset_n,
		i_enable_pix	     => enable_pix,
    i_dma_half_full    => '0',
		i_skip_hits 		   => "0000",
		i_fpga_id	         => x"ABCD",
		i_slow_down        => slow_down,
		o_data             => data_pix_generated_0,
		o_datak		         => datak_pix_generated_0,
		o_data_ready			 => open--,
);

e_data_gen_1 : entity work.data_generator_mupix
  port map (
    i_clk              => clk,
    i_reset_n          => reset_n,
    i_enable_pix       => enable_pix,
    i_dma_half_full    => '0',
    i_skip_hits        => "0001",
    i_fpga_id          => x"BACD",
    i_slow_down        => slow_down,
    o_data             => data_pix_generated_1,
    o_datak            => datak_pix_generated_1,
    o_data_ready       => open--,
);

e_data_gen_2 : entity work.data_generator_mupix
  port map (
    i_clk              => clk,
    i_reset_n          => reset_n,
    i_enable_pix       => enable_pix,
    i_dma_half_full    => '0',
    i_skip_hits        => "0011",
    i_fpga_id          => x"BADC",
    i_slow_down        => slow_down,
    o_data             => data_pix_generated_2,
    o_datak            => datak_pix_generated_2,
    o_data_ready       => open--,
);

e_data_gen_3 : entity work.data_generator_mupix
  port map (
    i_clk              => clk,
    i_reset_n          => reset_n,
    i_enable_pix       => enable_pix,
    i_dma_half_full    => '0',
    i_skip_hits        => "0101",
    i_fpga_id          => x"CBAD",
    i_slow_down        => slow_down,
    o_data             => data_pix_generated_3,
    o_datak            => datak_pix_generated_3,
    o_data_ready       => open--,
);

alginment_tree_data(1*32 - 1 downto 0*32) <= data_pix_generated_0;
alginment_tree_data(2*32 - 1 downto 1*32) <= data_pix_generated_0;
alginment_tree_data(3*32 - 1 downto 2*32) <= data_pix_generated_1;
alginment_tree_data(4*32 - 1 downto 3*32) <= data_pix_generated_1;
alginment_tree_data(5*32 - 1 downto 4*32) <= data_pix_generated_2;
alginment_tree_data(6*32 - 1 downto 5*32) <= data_pix_generated_2;
alginment_tree_data(7*32 - 1 downto 6*32) <= data_pix_generated_3;
alginment_tree_data(8*32 - 1 downto 7*32) <= data_pix_generated_3;

alginment_tree_data(9*32 - 1 downto 8*32) <= data_pix_generated_0;
alginment_tree_data(10*32 - 1 downto 9*32) <= data_pix_generated_0;
alginment_tree_data(11*32 - 1 downto 10*32) <= data_pix_generated_1;
alginment_tree_data(12*32 - 1 downto 11*32) <= data_pix_generated_1;
alginment_tree_data(13*32 - 1 downto 12*32) <= data_pix_generated_2;
alginment_tree_data(14*32 - 1 downto 13*32) <= data_pix_generated_2;
alginment_tree_data(15*32 - 1 downto 14*32) <= data_pix_generated_3;
alginment_tree_data(16*32 - 1 downto 15*32) <= data_pix_generated_3;

alginment_tree_data(17*32 - 1 downto 16*32) <= data_pix_generated_0;
alginment_tree_data(18*32 - 1 downto 17*32) <= data_pix_generated_0;
alginment_tree_data(19*32 - 1 downto 18*32) <= data_pix_generated_1;
alginment_tree_data(20*32 - 1 downto 19*32) <= data_pix_generated_1;
alginment_tree_data(21*32 - 1 downto 20*32) <= data_pix_generated_2;
alginment_tree_data(22*32 - 1 downto 21*32) <= data_pix_generated_2;
alginment_tree_data(23*32 - 1 downto 22*32) <= data_pix_generated_3;
alginment_tree_data(24*32 - 1 downto 23*32) <= data_pix_generated_3;

alginment_tree_data(25*32 - 1 downto 24*32) <= data_pix_generated_0;
alginment_tree_data(26*32 - 1 downto 25*32) <= data_pix_generated_0;
alginment_tree_data(27*32 - 1 downto 26*32) <= data_pix_generated_1;
alginment_tree_data(28*32 - 1 downto 27*32) <= data_pix_generated_1;
alginment_tree_data(29*32 - 1 downto 28*32) <= data_pix_generated_2;
alginment_tree_data(30*32 - 1 downto 29*32) <= data_pix_generated_2;
alginment_tree_data(31*32 - 1 downto 30*32) <= data_pix_generated_3;
alginment_tree_data(32*32 - 1 downto 31*32) <= data_pix_generated_3; 


alginment_tree_datak(1*4 - 1 downto 0*4) <= datak_pix_generated_0;
alginment_tree_datak(2*4 - 1 downto 1*4) <= datak_pix_generated_0;
alginment_tree_datak(3*4 - 1 downto 2*4) <= datak_pix_generated_1;
alginment_tree_datak(4*4 - 1 downto 3*4) <= datak_pix_generated_1;
alginment_tree_datak(5*4 - 1 downto 4*4) <= datak_pix_generated_2;
alginment_tree_datak(6*4 - 1 downto 5*4) <= datak_pix_generated_2;
alginment_tree_datak(7*4 - 1 downto 6*4) <= datak_pix_generated_3;
alginment_tree_datak(8*4 - 1 downto 7*4) <= datak_pix_generated_3;

alginment_tree_datak(9*4 - 1 downto 8*4) <= datak_pix_generated_0;
alginment_tree_datak(10*4 - 1 downto 9*4) <= datak_pix_generated_0;
alginment_tree_datak(11*4 - 1 downto 10*4) <= datak_pix_generated_1;
alginment_tree_datak(12*4 - 1 downto 11*4) <= datak_pix_generated_1;
alginment_tree_datak(13*4 - 1 downto 12*4) <= datak_pix_generated_2;
alginment_tree_datak(14*4 - 1 downto 13*4) <= datak_pix_generated_2;
alginment_tree_datak(15*4 - 1 downto 14*4) <= datak_pix_generated_3;
alginment_tree_datak(16*4 - 1 downto 15*4) <= datak_pix_generated_3;

alginment_tree_datak(17*4 - 1 downto 16*4) <= datak_pix_generated_0;
alginment_tree_datak(18*4 - 1 downto 17*4) <= datak_pix_generated_0;
alginment_tree_datak(19*4 - 1 downto 18*4) <= datak_pix_generated_1;
alginment_tree_datak(20*4 - 1 downto 19*4) <= datak_pix_generated_1;
alginment_tree_datak(21*4 - 1 downto 20*4) <= datak_pix_generated_2;
alginment_tree_datak(22*4 - 1 downto 21*4) <= datak_pix_generated_2;
alginment_tree_datak(23*4 - 1 downto 22*4) <= datak_pix_generated_3;
alginment_tree_datak(24*4 - 1 downto 23*4) <= datak_pix_generated_3;

alginment_tree_datak(25*4 - 1 downto 24*4) <= datak_pix_generated_0;
alginment_tree_datak(26*4 - 1 downto 25*4) <= datak_pix_generated_0;
alginment_tree_datak(27*4 - 1 downto 26*4) <= datak_pix_generated_1;
alginment_tree_datak(28*4 - 1 downto 27*4) <= datak_pix_generated_1;
alginment_tree_datak(29*4 - 1 downto 28*4) <= datak_pix_generated_2;
alginment_tree_datak(30*4 - 1 downto 29*4) <= datak_pix_generated_2;
alginment_tree_datak(31*4 - 1 downto 30*4) <= datak_pix_generated_3;
alginment_tree_datak(32*4 - 1 downto 31*4) <= datak_pix_generated_3; 


e_alginment_tree : entity work.alginment_tree
  generic map (
    NLINKS => 32,
    NFIRST => 4,
    NSECOND => 2,
    LINK_FIFO_ADDR_WIDTH => 10 --,
  )
  port map(
    i_clk_250 => clk,
    i_reset_n => reset_n,

    i_data    => alginment_tree_data,
    i_datak   => alginment_tree_datak,

    o_data    => open,
    o_datak   => open,
    o_error   => open--,
);
end architecture;
