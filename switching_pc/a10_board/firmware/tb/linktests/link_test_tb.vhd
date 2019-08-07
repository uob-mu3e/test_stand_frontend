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

  	component link_tester is
    port (
        cnt     :   out std_logic_vector(31 downto 0);
        datak   :   out std_logic_vector(3 downto 0);
        reset_n :   in  std_logic;
        enable  :   in  std_logic;
        clk     :   in  std_logic--;
    );
	end component link_tester;

	component link_observer is
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
	   
	   wait;
	end process inita;


	e_counter : component link_tester
    port map (
        cnt     => rx_data,
        datak   => rx_datak,
        reset_n => reset_n,
        enable  => enable,
        clk     => clk--,
    );

	e_link_observer : component link_observer
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