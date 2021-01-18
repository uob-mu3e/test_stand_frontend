library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use std.textio.all;
use IEEE.std_logic_textio.all; 


--  A testbench has no ports.
entity cc_mutrig_check_tb is
end entity;

architecture behav of cc_mutrig_check_tb is
  --  Specifies which entity is bound with the component.
  		
      signal clk : std_logic;
  	  signal reset_n : std_logic := '0';
  	  signal enable : std_logic  := '0';
  		  		
  		constant ckTime: 		time	 := 10 ns;
		
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
 
cc_check : entity work.cc_mutrig_check
	port map (
    i_125_clk           => clk,
    i_reset_n           => reset_n,
    i_enable            => enable,
    o_125_counter       => open,
    o_125_counter_right => open,
    o_25_counter        => open,
    o_n_lapses          => open--,
);

end architecture;
