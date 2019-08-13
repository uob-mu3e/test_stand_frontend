library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 


--  A testbench has no ports.
entity dma_test_tb is
end dma_test_tb;

architecture behav of dma_test_tb is
  --  Declaration of the component that will be instantiated.

	component dma_counter is
		generic (
			bits            : integer := 256--;
		);
		port(
			i_clk 			: 	in STD_LOGIC;
			i_reset_n		:	in std_logic;
			i_enable		:	in std_logic;
			i_dma_wen_reg	:	in std_logic;
			i_fraccount     :	in std_logic_vector(7 downto 0);
			o_dma_wen		:	out std_logic;
			o_cnt 			: 	out STD_LOGIC_VECTOR (bits - 1 downto 0)--;
			);
	end component dma_counter;

	signal clk 			: std_logic;
  	signal reset_n 		: std_logic := '1';
  	signal enable 		: std_logic := '0';
  	signal fraccount 	: std_logic_vector(7 downto 0);
  	signal dma_wen 		: std_logic;
  	signal dma_wen_reg  : std_logic;
  	signal cnt 			: std_logic_vector(255 downto 0);

  	constant ckTime		: time := 10 ns;

begin

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
	   enable    <= '1';
	   wait for 100 ns;
	   enable    <= '0';
	   wait for 50 ns;
	   enable    <= '1';
	   
	   wait;
	end process inita;

	fraccount <= x"FF";
	dma_wen_reg <= '1';

	e_counter : component dma_counter
	generic map(
		bits 	=> 256--,
	)
    port map (
		i_clk		=> clk,
		i_reset_n   => reset_n,
		i_enable    => enable,
		i_dma_wen_reg => dma_wen_reg,
		i_fraccount => fraccount,
		o_dma_wen   => dma_wen,
		o_cnt     	=> cnt--,
	);

end behav;
