-- Error counter for Transceiver
-- N. Berger with code from S. Dittmeier
-- 29.6.2015 niberger@uni-mainz.de


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;


entity rx_errcounter is
port (
	reset_n:					in std_logic;
	clk:						in std_logic;
	clk_out:					in std_logic;	-- to sync these values
	
	rx_freqlocked:			in std_logic;
	rx_sync:					in std_logic; 
	rx_err:					in std_logic;
	rx_disperr:				in std_logic;
	rx_pll_locked:			in std_logic;
	rx_patterndetect:		in std_logic;
	
	runcounter:				out work.util.reg32;
	errcounter:				out work.util.reg32;
	
	rx_freqlocked_out:			out std_logic;
	rx_sync_out:					out std_logic; 
	rx_err_out:						out std_logic;
	rx_disperr_out:				out std_logic;
	rx_pll_locked_out:			out std_logic;
	rx_patterndetect_out:		out std_logic

);
end rx_errcounter;

architecture rtl of rx_errcounter is

signal timer:				std_logic_vector(27 downto 0);
signal err:					work.util.reg32;
signal run:					work.util.reg32;

signal err_out:			work.util.reg32;
signal run_out:			work.util.reg32;

signal rx_status : std_logic_vector(5 downto 0);
signal rx_status_out : std_logic_vector(5 downto 0);

signal clk_reg : std_logic_vector(1 downto 0);

begin

-- speed between clocks: clk speeds are supposed to be different
-- for synchronization
process(reset_n,clk_out)
begin
	if(reset_n = '0')then
		clk_reg 		<= (others => '0');
		runcounter	<= (others => '0');
		errcounter	<= (others => '0');
		rx_status_out	<= (others => '0');
	elsif(rising_edge(clk_out))then
		clk_reg <= clk_reg(0) & clk;

		if(clk_reg = "01")then
			runcounter 	<= run;
			errcounter	<= err;	
			rx_status_out	<= rx_status;
		end if;
	end if;
end process;


rx_freqlocked_out		<= rx_status_out(5);
rx_sync_out				<= rx_status_out(4);
rx_err_out				<= rx_status_out(3);
rx_disperr_out			<= rx_status_out(2);
rx_pll_locked_out		<= rx_status_out(1);
rx_patterndetect_out	<= rx_status_out(0);



process(reset_n,clk)
begin
	if(reset_n = '0')then
		rx_status <= (others => '0');
	elsif(rising_edge(clk))then
		rx_status <= rx_freqlocked & rx_sync & rx_err & rx_disperr & rx_pll_locked & rx_patterndetect;
	end if;
end process;


process(reset_n, clk)
begin
	if(reset_n = '0')then
		err	<= (others => '0');
		run	<= (others => '0');
		timer	<= (others => '0');
	elsif(rising_edge(clk))then
		if(rx_status(4) = '1') then
			timer 		<= timer + '1';
			if(timer = work.util.TIME_125MHz_1ms)then	-- use 1 ms instead of 1 second (but beware: runs actually with 156.25 MHz, not 125 MHz!)
				timer 	<= (others => '0');
				run		<= run + '1';
			end if;
			if((rx_status(3) = '1') or (rx_status(2) = '1'))then
				err 	<= err + '1';
			end if;
		end if;

	end if;
end process;



	

end rtl;
