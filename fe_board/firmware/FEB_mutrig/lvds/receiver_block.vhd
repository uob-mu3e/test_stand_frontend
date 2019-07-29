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
--use work.lvds_components.all;
--use work.mupix_types.all;
use work.daq_constants.all;



entity receiver_block is
generic (
	NINPUT : positive := 1;
	LVDS_PLL_FREQ : real := 125.0;
	LVDS_DATA_RATE : positive := 1250--;
);
port (
	reset_n             : in std_logic;
	reset_n_errcnt      : in std_logic;
	rx_in               : in std_logic_vector(NINPUT-1 downto 0);
	rx_inclock          : in std_logic;
	rx_state            : out std_logic_vector(2*NINPUT-1 downto 0);
	rx_ready            : out std_logic_vector(NINPUT-1 downto 0);
	rx_data             : out std_logic_vector(NINPUT*8-1 downto 0);
	rx_k                : out std_logic_vector(NINPUT-1 downto 0);
	rx_clkout           : out std_logic;
	pll_locked          : out std_logic;

	rx_dpa_locked_out   : out std_logic_vector(NINPUT-1 downto 0);

	rx_runcounter       : out links_reg32;
	rx_errorcounter     : out links_reg32
);
end receiver_block;



architecture rtl of receiver_block is

component data_decoder is 
	generic (
		EVAL_WINDOW_WORDCNT_BITS : natural := 8; -- number of bits of the counter used to check for the sync pattern
		EVAL_WINDOW_PATTERN_BITS : natural := 1; -- number of bits of the counter of the sync patterns found in the window (realign if not overflow)
		ALIGN_WORD	 : std_logic_vector(7 downto 0):=k28_5 -- pattern byte to search for
	);
	port (
		reset_n				: in std_logic;
--		checker_rst_n		: in std_logic;
		clk					: in std_logic;
		rx_in					: IN STD_LOGIC_VECTOR (9 DOWNTO 0);
		
		rx_reset				: OUT STD_LOGIC;
		rx_fifo_reset		: OUT STD_LOGIC;
		rx_dpa_locked		: IN STD_LOGIC;
		rx_locked			: IN STD_LOGIC;
		rx_bitslip				: OUT STD_LOGIC;
	
		ready					: OUT STD_LOGIC;
		data					: OUT STD_LOGIC_VECTOR(7 downto 0);
		k						: OUT STD_LOGIC;
		state_out			: out std_logic_vector(1 downto 0);		-- 4 possible states
		disp_err				: out std_logic
		);
end component; --data_decoder;

component rx_errcounter is
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
end component; --rx_errcounter;



	signal rx_out : 		std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_out_order : 		std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_clk :			std_logic;

	signal rx_dpa_locked		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_bitslip			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_fifo_reset		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_reset			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);

	signal rx_ready_reg		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);	
	signal rx_locked		: STD_LOGIC;
	signal rx_disperr		: std_logic_vector(NINPUT-1 downto 0);

begin

	rx_dpa_locked_out	<= rx_dpa_locked;
	pll_locked 			<= rx_locked;
	rx_clkout 			<= rx_clk;

	lvds_rx : entity work.ip_altlvds_rx
	GENERIC MAP (
		N => NINPUT,
		PLL_FREQ => LVDS_PLL_FREQ,
		DATA_RATE => LVDS_DATA_RATE--,
	)
	PORT MAP (
		rx_channel_data_align	=> rx_bitslip,
		rx_fifo_reset		=> rx_fifo_reset,
		rx_in			=> rx_in,
		rx_inclock		=> rx_inclock,
		rx_reset		=> rx_reset,
		rx_dpa_locked		=> rx_dpa_locked,
		rx_locked		=> rx_locked,
		rx_out			=> rx_out,
		rx_outclock		=> rx_clk
	);

	rx_ready <= rx_ready_reg;


-- flip bit order of received data (msb-lsb)
flip_bits: process(rx_out)
begin
for i in NINPUT-1 downto 0 loop
	for n in 9 downto 0 loop
		rx_out_order(10*i+n) <= rx_out(10*i+9-n);
	end loop;
end loop;
end process flip_bits;

gen_channels: for i in NINPUT-1 downto 0 generate
	datadec: data_decoder 
		generic map(
			EVAL_WINDOW_WORDCNT_BITS => 10,
			EVAL_WINDOW_PATTERN_BITS => 1,
			ALIGN_WORD	 	 => k28_5
		)
		port map(
			reset_n			=> reset_n,
	--		checker_rst_n		=> checker_rst_n(i),
			clk			=> rx_clk,
			rx_in			=> rx_out_order((i+1)*10-1 downto i*10),
			
			rx_reset		=> rx_reset(i),
			rx_fifo_reset		=> rx_fifo_reset(i),
			rx_dpa_locked		=> rx_dpa_locked(i),
			rx_locked		=> rx_locked,
			rx_bitslip		=> rx_bitslip(i),
		
			ready			=> rx_ready_reg(i),
			data			=> rx_data((i+1)*8-1 downto i*8),
			k			=> rx_k(i),
			state_out		=> rx_state((i+1)*2-1 downto i*2),
			disp_err		=> rx_disperr(i)
		);


	errcounter:rx_errcounter 
		port map(
			reset_n			=> reset_n_errcnt,
			clk			=> rx_clk,
			clk_out			=> rx_inclock,

			rx_freqlocked		=> '0',--rx_freqlocked,
			rx_sync			=> rx_ready_reg(i),
			rx_err			=> '0',--rx_errdetect,
			rx_disperr		=> rx_disperr(i),
			rx_pll_locked		=> '0',--rx_pll_locked,
			rx_patterndetect	=> '0',--rx_patterndetect,

			runcounter		=> rx_runcounter,--readregs_slow(RECEIVER_RUNTIME_REGISTER_R+i),
			errcounter		=> rx_errorcounter,--readregs_slow(RECEIVER_ERRCOUNT_REGISTER_R+i),
		--	sync_lost		=> sync_lost,--readregs_slow(RECEIVER_SYNCLOST_SYNCFIFO_REGISTER_R+i)
		--	freq_lost		=> freq_lost,

			rx_freqlocked_out	=> open,--rx_freqlocked_out,
			rx_sync_out		=> open,
			rx_err_out		=> open,--rx_errdetect_out,
			rx_disperr_out		=> open,--rx_disperr_out,
			rx_pll_locked_out	=> open,
			rx_patterndetect_out	=> open--rx_patterndetect_out
		);
end generate;



end rtl;
