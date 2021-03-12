-- Error counter for Transceiver
-- N. Berger with code from S. Dittmeier
-- 29.6.2015 niberger@uni-mainz.de
-- 1/2020: generic cleanup removing all unused parts, K.Briggl

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;


entity rx_errcounter is
port (
	reset_n:			in std_logic;
	clk:				in std_logic;
	
	rx_sync:			in std_logic; 
	rx_disperr:			in std_logic;
	
--counters
	o_runcounter		: out work.util.reg32;
	o_errcounter		: out work.util.reg32;
	o_synclosscounter   : out work.util.reg32
);
end rx_errcounter;

architecture rtl of rx_errcounter is
signal timer			: std_logic_vector(27 downto 0);
signal runcounter		: work.util.reg32;
signal errcounter		: work.util.reg32;
signal synclosscounter	: work.util.reg32;
signal synced_d			: std_logic;

begin
	o_runcounter<=runcounter;
	o_errcounter<=errcounter;
	o_synclosscounter<=synclosscounter;

process(reset_n, clk)
begin
	if(reset_n = '0')then
		errcounter	<= (others => '0');
		runcounter	<= (others => '0');
		timer	<= (others => '0');
		synclosscounter <= (others => '0');
		synced_d <= '0';

	elsif(rising_edge(clk))then
		synced_d <= rx_sync;
		if(rx_sync = '1') then
			timer 		<= timer + '1';
			if(timer = work.util.TIME_125MHz_1ms)then	-- use 1 ms instead of 1 second (but beware: runs actually with 156.25 MHz, not 125 MHz!)
				timer      <= (others => '0');
				runcounter <= work.util.gray_inc(runcounter);
			end if;
			if(rx_disperr = '1')then
				errcounter <= work.util.gray_inc(errcounter);
			end if;
		end if;
		if(rx_sync='0' and synced_d = '1') then
			synclosscounter <= work.util.gray_inc(synclosscounter);
		end if;
	end if;
end process;



end rtl;
