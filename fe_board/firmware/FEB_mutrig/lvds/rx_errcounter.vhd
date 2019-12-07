-- Error counter for Transceiver
-- N. Berger with code from S. Dittmeier
-- 29.6.2015 niberger@uni-mainz.de


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.daq_constants.all;
use work.util.all;

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
	
	runcounter:				out reg32;
	errcounter:				out reg32;
--	sync_lost:				out reg32;
--	freq_lost:				out reg32;
	
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
signal err:					reg32;
signal run:					reg32;
--signal sync_lost_cnt: 	reg32;
--signal freq_lost_cnt:	reg32;
signal sync_init:			std_logic;

signal err_out:					reg32;
signal run_out:					reg32;

signal rx_status : std_logic_vector(5 downto 0);
signal rx_status_del : std_logic_vector(5 downto 0);
signal rx_status_del2 : std_logic_vector(5 downto 0);
signal rx_status_out : std_logic_vector(5 downto 0);

signal wrreq	: std_logic;
signal rdreq	: std_logic;

signal run_wrusedw : std_logic_vector(3 downto 0);
signal run_rdusedw : std_logic_vector(3 downto 0); 


begin

-- speed between clocks: clk is 156.25 MHz, clkout is 125 MHz
-- for synchronization
-- does not work for telescope, also don't care
process(reset_n,clk_out)
begin
	if(reset_n = '0')then
		runcounter	<= (others => '0');
		errcounter	<= (others => '0');
		rx_status_out	<= (others => '0');
	elsif(rising_edge(clk_out))then
		runcounter 	<= run;
		errcounter	<= err;	
		rx_status_out	<= rx_status;
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
		sync_init <= '0';
	elsif(rising_edge(clk))then
		if(rx_status(4) = '1') then
			timer 		<= timer + '1';
			if(timer = TIME_125MHz_1ms)then	-- use 1 ms instead of 1 second (but beware: runs actually with 156.25 MHz, not 125 MHz!)
				timer 	<= (others => '0');
				run		<= gray_inc(run);
			end if;
			if((rx_status(3) = '1') or (rx_status(2) = '1'))then
				err 	<= gray_inc(err);
			end if;
			if(sync_init = '0') then
				sync_init <= '1';
			end if;
		end if;
	end if;
end process;



end rtl;
