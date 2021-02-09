
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use ieee.math_real.all;


entity lapse_counter is
generic (
	N_TOT : positive := 32767;
	N_CC : positive := 32;
	N_TIME : positive := 48--;
);
port (
	i_reset_n   : in  std_logic;
	i_clk       : in  std_logic; -- 625 MHz

	i_CC		: in  std_logic_vector(N_CC - 1 downto 0);
	o_CC		: out std_logic_vector(N_CC downto 0)--;
);
end lapse_counter;

architecture arch of lapse_counter is

	signal CC_fpga	: std_logic_vector(N_CC - 1 downto 0);
	signal s_o_CC	: std_logic_vector(N_CC downto 0);
	signal s_o_CC_reg	: std_logic_vector(N_CC downto 0);
	signal nLapses	: integer := 0;
	
begin

--counting lapsing of coarse counter
--CC lapses every 2^15-1 cycles @ 625MHz. 
p_gen_lapsing: process(i_clk, i_reset_n)
begin
    if ( i_reset_n = '0' ) then
        nLapses <= 0;
        CC_fpga <= (others => '0');
    elsif ( rising_edge(i_clk) ) then
        if ( CC_fpga = N_TOT - 1 ) then
            nLapses <= nLapses + 1;
            CC_fpga <= (others => '0');
        else
        	CC_fpga <= CC_fpga + 1;
        end if;
    end if;
end process;

s_o_CC 		<= "0" & i_CC;
s_o_CC_reg 	<= 	s_o_CC 				when nLapses = 0 else
				s_o_CC - nLapses 	when i_CC <= CC_fpga else s_o_CC - (nLapses - 1);

o_CC <= s_o_CC_reg when s_o_CC_reg <= N_TOT - 1 else s_o_CC_reg - N_TOT - 1;

--FPGA 2^15-1 counter: 0 1 2 0 1 2 0 1 2
--nLapse:              0 0 0 1 1 1 2 2 2
--FPGA CC_ASIC:        - - 1 2 0 1 2 0 1
--Corrected:           - - 1 2 3 0 1 2 3
--Corrected <= CC - nLapse when CC < FPGA counter else CC - (nLapse - 1);


end architecture;
