-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Receiver block for all the LVDS links
-- Niklaus Berger, May 2013
-- 
-- nberger@physi.uni-heidelberg.de
--
----------------------------------



library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.lvds_components.all;
use work.transceiver_components.all;
use work.mupix_types.all;
use work.mupix_constants.all;


entity receiver_block_mupix is 
	generic(
		NINPUT: integer := 45;
		NCHIPS: integer := 15
	);
	port (
		reset_n				: in std_logic;
		checker_rst_n		: in std_logic;--_vector(NINPUT-1 downto 0);
		rx_in					: IN STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
		rx_inclock_A		: IN STD_LOGIC ;
		rx_inclock_B		: IN STD_LOGIC ;
		rx_state				: out std_logic_vector(NINPUT*2-1 downto 0);
		rx_ready				: out std_logic_vector(NINPUT-1 downto 0);		-- sync with rx_inclock_A
		rx_data				: out inbyte_array;										-- sync with rx_inclock_A
		rx_k					: out std_logic_vector(NINPUT-1 downto 0);		-- sync with rx_inclock_A

		pll_locked			: out std_logic_vector(1 downto 0);
		
		nios_clock			: in std_logic;
		rx_runcounter:			out links_reg32;
		rx_errorcounter:		out links_reg32	
		);
end receiver_block_mupix;		
		
architecture rtl of receiver_block_mupix is

	signal rx_out : 			std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_out_order :	std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_out_order_to_sync :	std_logic_vector(NINPUT*8-1 downto 0);
	signal rx_k_to_sync :	std_logic_vector(NINPUT*1 downto 0);
	signal rx_to_sync :	std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_sync :	std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_clk :			std_logic_vector(1 downto 0);

	signal rx_dpa_locked		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_align			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_fifo_reset		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_reset			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	
	signal rx_locked			: STD_LOGIC_VECTOR(1 downto 0);
	
	signal 	rx_inclock_A_pll:		std_logic;
	signal 	rx_enable_A:			std_logic;
	signal   rx_synclock_A:			std_logic;	
	
	signal 	rx_inclock_B_pll:		std_logic;
	signal 	rx_enable_B:			std_logic;
	signal   rx_synclock_B:			std_logic;				
	
	signal rx_ready_reg		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);	
	signal rx_disperr			: std_logic_vector(NINPUT-1 downto 0);
	
	signal rx_valid : std_logic_vector(NINPUT-1 downto 0);
	

begin

	pll_locked <= rx_locked;
	rx_clk(0) <= rx_synclock_A;
	rx_clk(1) <= rx_synclock_B;

lpll_A : work.lvdspll
	PORT MAP
	(
		inclk0		=> rx_inclock_A,
		c0				=> rx_inclock_A_pll,
		c1				=> rx_enable_A,
		c2				=> rx_synclock_A,
		locked		=> rx_locked(0)
);	
		
lvds_rec_small : work.lvds_receiver_small
	PORT MAP
	(
		pll_areset						=> not rx_locked(0),
		rx_channel_data_align		=> rx_align(NINPUTS_BANK_A-1 downto 0),
		rx_enable						=> rx_enable_A,
		rx_fifo_reset					=> rx_fifo_reset(NINPUTS_BANK_A-1 downto 0),
		rx_in								=> rx_in(NINPUTS_BANK_A-1 downto 0),
		rx_inclock						=> rx_inclock_A_pll,
		rx_reset							=> rx_reset(NINPUTS_BANK_A-1 downto 0),
		rx_syncclock					=> rx_synclock_A,
		rx_dpa_locked					=> rx_dpa_locked(NINPUTS_BANK_A-1 downto 0),
		rx_out							=> rx_out(NINPUTS_BANK_A*10-1 downto 0)
	);

lpll_B : work.lvdspll
	PORT MAP
	(
		inclk0		=> rx_inclock_B,
		c0				=> rx_inclock_B_pll,
		c1				=> rx_enable_B,
		c2				=> rx_synclock_B,
		locked		=> rx_locked(1)
);
	
lvds_rec : work.lvds_receiver_small
	PORT MAP
	(
		pll_areset					=> not rx_locked(1),
		rx_channel_data_align 	=> rx_align(NINPUTS_BANK_A+NINPUTS_BANK_B-1 downto NINPUTS_BANK_A),
		rx_enable					=> rx_enable_B,
		rx_fifo_reset				=> rx_fifo_reset(NINPUTS_BANK_A+NINPUTS_BANK_B-1 downto NINPUTS_BANK_A),
		rx_in							=> rx_in(NINPUTS_BANK_A+NINPUTS_BANK_B-1 downto NINPUTS_BANK_A),
		rx_inclock		   		=> rx_inclock_B_pll,
		rx_reset						=> rx_reset(NINPUTS_BANK_A+NINPUTS_BANK_B-1 downto NINPUTS_BANK_A),
		rx_syncclock				=> rx_synclock_B,
		rx_dpa_locked				=> rx_dpa_locked(NINPUTS_BANK_A+NINPUTS_BANK_B-1 downto NINPUTS_BANK_A),
		rx_out						=> rx_out((NINPUTS_BANK_A+NINPUTS_BANK_B)*10-1 downto NINPUTS_BANK_A*10)
	);
			
		

gendec:
FOR i in NINPUT-1 downto 0 generate	

	process(rx_clk(i/NINPUTS_BANK_A), reset_n)
	begin
		if(reset_n = '0')then
			rx_out_order(i*10+9 downto i*10) <= (others => '0');
		elsif(rising_edge(rx_clk(i/NINPUTS_BANK_A)))then
			-- default Mupix: inverted bit odering!
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
	end process;

		i_datadec : work.data_decoder 
		port map(
			reset_n				=> reset_n,
			clk					=> rx_clk(i/NINPUTS_BANK_A),
			rx_in					=> rx_out_order(i*10+9 downto i*10),
			
			rx_reset				=> rx_reset(i),
			rx_fifo_reset     => rx_fifo_reset(i),
			rx_dpa_locked		=> rx_dpa_locked(i),
			rx_locked			=> rx_locked(i/NINPUTS_BANK_A),
			rx_align				=> rx_align(i),
		
			ready					=> rx_ready_reg(i),
			data					=> rx_out_order_to_sync(i*8+7 downto i*8),
			k						=> rx_k_to_sync(i),
			state_out			=> rx_state(i*2+1 downto i*2),
			disp_err				=> rx_disperr(i)
			);

	rx_to_sync(i*10+9 downto i*10) <= rx_ready_reg(i) & rx_k_to_sync(i) & rx_out_order_to_sync(i*8+7 downto i*8);

	i_fifo_10b : work.fifo_Nb_sync
	generic map(
      NBITS => 10
	)
	port map(
		clk_wr 	=> rx_clk(i/NINPUTS_BANK_A),
		clk_rd	=> rx_inclock_A,
		rst_n		=> reset_n,
		wrreq_in	=> rx_locked(i/NINPUTS_BANK_A),	-- sent all through!
		data_in	=> rx_to_sync(i*10+9 downto i*10),
		data_out => rx_sync(i*10+9 downto i*10),
		data_valid	=> rx_valid(i)
	);

	output_clocking : process(rx_inclock_A, reset_n)
	begin
		if(reset_n = '0')then
			rx_ready(i)		<= '0';
			rx_k(i)			<= '0';
			rx_data(i)		<= (others => '0');
		elsif(rising_edge(rx_inclock_A))then
			if(rx_valid(i) = '1')then
				rx_ready(i)		<= rx_sync(i*10+9);
				rx_k(i)			<= rx_sync(i*10+8);
				rx_data(i)		<= rx_sync(i*10+7 downto i*10);
			else
				rx_ready(i)		<= '0';
				rx_k(i)			<= '0';
				rx_data(i)		<= (others => '0');			
			end if;
		end if;
	end process output_clocking;


	errcounter : work.rx_errcounter 
	port map(
		reset_n					=> checker_rst_n,
		clk						=> rx_clk(i/NINPUTS_BANK_A),
		clk_out					=> nios_clock,	-- use this as default clock
		
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