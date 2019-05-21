--=============================================================
--
--bclock_gen.vhdl 
--
-- Create divided clock synchronous to a reset signal
--
--
--
--Version Date: 10.06.2011
--
--Author: Tobias Harion
--Kirchhoff-Institut Heidelberg
--
--
--
--=============================================================


LIBRARY ieee;
USE ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;

entity bclock_gen is	--{{{
	generic(Nd : Integer := 10);
PORT(
		i_clk 			: IN  STD_LOGIC;
		i_start_val		: in std_logic_vector(3 downto 0);
		i_rst			: in std_logic;
      o_div_clk     : OUT STD_LOGIC
	);
end bclock_gen;	--}}}

architecture counter of bclock_gen is --{{{1

signal q  : STD_LOGIC;
signal cnt : std_logic_vector(3 downto 0);

begin



divN: process(i_clk, i_rst) -- frequency divider	--{{{2
    begin
		if rising_edge(i_clk) then	 -- assign the new values of the counter
			if i_rst = '1' then		 -- RESET IF THERE WAS A SYNC SIGNAL
				cnt <= X"8";			 -- USE THE PROGRAMMABLE RESET VALUE, EASIER FOR DEBUGGING
	--			cnt <= conv_std_logic_vector(Nd-5,4);
				q <= '1';				 --THE RESET COMES FOR THE RISING EDGE OF THE SIGNAL -> SET q to '1'
			else
				if cnt = "0000" then
					cnt <= conv_std_logic_vector(Nd-1,4);
				else
					cnt <= conv_std_logic_vector(conv_integer(unsigned(cnt))-1,4);
				end if;


				if conv_integer(unsigned(cnt))=0 then
					q <= '1';
				end if;
				if conv_integer(unsigned(cnt)) = Nd/2 then
					q <= '0';
				end if;
			end if;--from reset
		end if;
    end process;	--}}}

o_div_clk <= q;

end architecture counter;	--}}}
