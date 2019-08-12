library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 


--  A testbench has no ports.
entity link_test_tb is
end link_test_tb;

architecture behav of link_test_tb is
  --  Declaration of the component that will be instantiated.

  	component linear_shift is
  	generic (
        g_m     : integer;
        g_poly  : std_logic_vector--;
    );
    port (
        i_clk           : in  std_logic;
        reset_n         : in  std_logic;
        i_sync_reset    : in  std_logic;
        i_seed          : in  std_logic_vector (g_m-1 downto 0);
        i_en            : in  std_logic;
        o_lsfr          : out std_logic_vector (g_m-1 downto 0);
        o_datak         : out std_logic_vector (3 downto 0)--;
    );
	end component linear_shift;

	component link_observer is
	generic (
        g_m             : integer           := 7;
        g_poly          : std_logic_vector  := "1100000" -- x^7+x^6+1 
    );
    port(
		clk:               in std_logic;
		reset_n:           in std_logic;
		rx_data:           in std_logic_vector (31 downto 0);
		rx_datak:          in std_logic_vector (3 downto 0);
		error_counts:      out std_logic_vector (31 downto 0);
		bit_counts:        out std_logic_vector (31 downto 0);
		state_out:         out std_logic_vector(3 downto 0)--;
	);
	end component link_observer;

	signal clk 			: std_logic;
  	signal reset_n 		: std_logic := '1';
  	signal enable 		: std_logic := '0';
  	signal rx_data 		: std_logic_vector(31 downto 0);
  	signal rx_datak 	: std_logic_vector(3 downto 0);
  	signal error_counts : std_logic_vector(31 downto 0);
  	signal bit_counts 	: std_logic_vector(31 downto 0);

  	constant ckTime: 		time	:= 10 ns;

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


	e_linear_shift : component linear_shift
	generic map(
		g_m 	=> 32,
		g_poly 	=> "10000000001000000000000000000110"
	)
    port map (
		i_clk 			=> clk,
		reset_n 		=> reset_n,
		i_sync_reset 	=> '0',
		i_seed			=> (others => '1'),
		i_en 			=> enable,
		o_lsfr			=> rx_data,
		o_datak 		=> rx_datak--,
    );

	e_link_observer : component link_observer
	generic map(
		g_m 	=> 32,
		g_poly 	=> "10000000001000000000000000000110"
	)
    port map (
		clk     		=> clk,
		reset_n     	=> reset_n,
		rx_data     	=> rx_data,
		rx_datak     	=> rx_datak,
		error_counts    => error_counts,
		bit_counts     	=> bit_counts,
		state_out     	=> open--,
	);

end behav;
