-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Receiver block for all the LVDS links
-- Niklaus Berger, May 2013
-- 
-- nberger@physi.uni-heidelberg.de
--
-- Adaptions for MuPix8 Telescope
-- Sebastian Dittmeier, April 2016
-- dittmeier@physi.uni-heidelberg.de
----------------------------------



library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.lvds_components.all;
use work.transceiver_components.all;
use work.mupix_types.all;


entity receiver_block_mupix is 
	generic(
		NINPUT: integer := 16;
		NCHIPS: integer := 4
	);
	port (
		reset_n				: in std_logic;
		reset_n_errcnt		: in std_logic;
		rx_in					: in std_logic_vector (NINPUT-1 DOWNTO 0);
		rx_inclock			: in std_logic;
		rx_data_bitorder	: in std_logic;	-- set to '0' for data as received, set to '1' to invert order LSB to MSB
		rx_state				: out std_logic_vector(NINPUT*2-1 downto 0);
		rx_ready				: out std_logic_vector(NINPUT-1 downto 0);
		rx_data				: out std_logic_vector(NINPUT*8-1 downto 0);
		rx_k					: out std_logic_vector(NINPUT-1 downto 0);
		rx_clkout			: out std_logic_vector(1 downto 0);
		pll_locked			: out std_logic_vector(1 downto 0);

		rx_dpa_locked_out:	out std_logic_vector(NINPUT-1 downto 0);
	
		rx_runcounter:			out links_reg32;
		rx_errorcounter:		out links_reg32		
		);
end receiver_block_mupix;		

		
architecture rtl of receiver_block_mupix is

	signal rx_out : 			std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_out_order :	std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_clk :			std_logic_vector(1 downto 0);

	signal rx_dpa_locked		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_align			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_fifo_reset		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_reset			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);

	signal rx_ready_reg		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);	
	signal rx_locked			: STD_LOGIC_VECTOR(1 downto 0);
	signal rx_disperr			: std_logic_vector(NINPUT-1 downto 0);

begin

	rx_dpa_locked_out	<= rx_dpa_locked;
	pll_locked 			<= rx_locked;
	rx_clkout 			<= rx_clk;

	
-- separate for HSMC BANKs A and B	
gen_lvds: for I in 0 to 1 generate

lvds_rx : work.lvds_receiver_mupix
	PORT MAP
	(
		rx_channel_data_align	=> rx_align(8*(I+1)-1 downto 8*I),
		rx_fifo_reset				=> rx_fifo_reset(8*(I+1)-1 downto 8*I),
		rx_in							=> rx_in(8*(I+1)-1 downto 8*I),
		rx_inclock					=> rx_inclock,
		rx_reset						=>	rx_reset(8*(I+1)-1 downto 8*I),
		rx_dpa_locked				=> rx_dpa_locked(8*(I+1)-1 downto 8*I),
		rx_locked					=> rx_locked(I),
		rx_out						=> rx_out(80*(I+1)-1 downto 80*I),
		rx_outclock					=> rx_clk(I)
	);

end generate gen_lvds;	
	
	
rx_ready <= rx_ready_reg;	
	
-- decoder for each link		
gendec:
FOR i in NINPUT-1 downto 0 generate	

process(rx_clk(i/8), reset_n)
begin
	if(reset_n = '0')then
		rx_out_order(i*10+9 downto i*10) <= (others => '0');
	elsif(rising_edge(rx_clk(i/8)))then
		if (rx_data_bitorder = '0')then
			rx_out_order(i*10+9 downto i*10) <= rx_out(i*10+9 downto i*10);
		else
			rx_out_order(i*10+9) <= rx_out(i*10+0);
			rx_out_order(i*10+8) <= rx_out(i*10+1);
			rx_out_order(i*10+7) <= rx_out(i*10+2);
			rx_out_order(i*10+6) <= rx_out(i*10+3);
			rx_out_order(i*10+5) <= rx_out(i*10+4);
			rx_out_order(i*10+4) <= rx_out(i*10+5);
			rx_out_order(i*10+3) <= rx_out(i*10+6);
			rx_out_order(i*10+2) <= rx_out(i*10+7);
			rx_out_order(i*10+1) <= rx_out(i*10+8);			
			rx_out_order(i*10+0) <= rx_out(i*10+9);			
		end if;
	end if;
end process;

datadec : work.data_decoder_mupix
	port map(
		reset_n				=> reset_n,
--		checker_rst_n		=> checker_rst_n(i),
		clk					=> rx_clk(i/8),
		rx_in					=> rx_out_order(i*10+9 downto i*10),
		
		rx_reset				=> rx_reset(i),
		rx_fifo_reset     => rx_fifo_reset(i),
		rx_dpa_locked		=> rx_dpa_locked(i),
		rx_locked			=> rx_locked(i/8),
		rx_align				=> rx_align(i),
	
		ready					=> rx_ready_reg(i),
		data					=> rx_data(i*8+7 downto i*8),
		k						=> rx_k(i),
		state_out			=> rx_state(i*2+1 downto i*2),
		disp_err				=> rx_disperr(i)
		);
		
		
	errcounter :  work.rx_errcounter_mupix
	port map(
		reset_n					=> reset_n_errcnt,
		clk						=> rx_clk(i/8),
		clk_out					=> rx_inclock,
		
		rx_freqlocked			=> '0',--rx_freqlocked(i),
		rx_sync					=> rx_ready_reg(i),--rx_syncstatus(i),
		rx_err					=> '0',--rx_errdetect(i),
		rx_disperr				=> rx_disperr(i),
		rx_pll_locked			=> '0',--rx_pll_locked(i),
		rx_patterndetect		=> '0',--rx_patterndetect(i),
		
		runcounter				=> rx_runcounter(i),--readregs_slow(RECEIVER_RUNTIME_REGISTER_R+i),
		errcounter				=> rx_errorcounter(i),--readregs_slow(RECEIVER_ERRCOUNT_REGISTER_R+i),
	--	sync_lost				=> sync_lost(i),--readregs_slow(RECEIVER_SYNCLOST_SYNCFIFO_REGISTER_R+i)
	--	freq_lost				=> freq_lost(i),
		
		rx_freqlocked_out		=> open,--rx_freqlocked_out(i),
		rx_sync_out				=> open,
		rx_err_out				=> open,--rx_errdetect_out(i),
		rx_disperr_out			=> open,--rx_disperr_out(i),
		rx_pll_locked_out		=> open,
		rx_patterndetect_out	=> open--rx_patterndetect_out(i)		
	);

end generate;



end rtl;
