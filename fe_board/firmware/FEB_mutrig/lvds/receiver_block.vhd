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
use work.daq_constants.all;



entity receiver_block is
generic (
    NINPUT : positive := 1;
    LVDS_PLL_FREQ : real := 125.0;
    LVDS_DATA_RATE : real := 1250.0;
    INPUT_SIGNFLIP : std_logic_vector := x"0000"--;
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

    rx_runcounter       : out reg32array_t(NINPUT-1 downto 0);
    rx_errorcounter     : out reg32array_t(NINPUT-1 downto 0);
    rx_synclosscounter  : out reg32array_t(NINPUT-1 downto 0)
);
end entity;



architecture rtl of receiver_block is

	signal rx_out : 		std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_out_order : 		std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_clk :			std_logic;

	signal rx_dpa_locked		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_bitslip			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_fifo_reset		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_reset			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);

	signal rx_ready_reg		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);	
	signal rx_pll_locked		: STD_LOGIC;
	signal rx_disperr		: std_logic_vector(NINPUT-1 downto 0);

    signal rx_inclock_A_ctrl    : std_logic;
    signal rx_inclock_A_pll     : std_logic;
    signal rx_locked            : std_logic;
    signal rx_dpaclock_A        : std_logic;
    signal rx_syncclock_A       : std_logic;
    signal rx_enable_A          : std_logic;

begin
	rx_dpa_locked_out	<= rx_dpa_locked;
	pll_locked 			<= rx_pll_locked;
	rx_clkout 			<= rx_clk;

    lvds_clk_ctrl : component work.cmp.clk_ctrl_single
    port map (
        inclk  => rx_inclock,
        outclk => rx_inclock_A_ctrl
    );

    lpll_A: entity work.lvdspll
    PORT MAP
    (
        refclk   => rx_inclock_A_ctrl,
        rst      => '0',
        outclk_0 => rx_inclock_A_pll,
        outclk_1 => rx_enable_A,
        outclk_2 => rx_syncclock_A,
        outclk_3 => rx_dpaclock_A,
        outclk_4 => open,
        locked   => rx_locked
    );

	lvds_rx : entity work.lvds_receiver
--	GENERIC MAP (
--		N => NINPUT,
--		PLL_FREQ => LVDS_PLL_FREQ,
--		DATA_RATE => LVDS_DATA_RATE--,
--	)
	PORT MAP (
        pll_areset                              => not rx_locked,
        rx_channel_data_align(NINPUT-1 downto 0)=> rx_bitslip,
        rx_channel_data_align(26 downto NINPUT) => (others => '0'),
        rx_dpaclock                             => rx_dpaclock_A,
        rx_enable                               => rx_enable_A,
        rx_fifo_reset(NINPUT-1 downto 0)        => rx_fifo_reset,
        rx_in(NINPUT-1 downto 0)                => rx_in,
        rx_in(26 downto NINPUT)                 => (others => '0'),
        rx_inclock                              => rx_inclock_A_pll,
        rx_reset(NINPUT-1 downto 0)             => rx_reset,
        rx_syncclock                            => rx_syncclock_A,
        rx_dpa_locked(NINPUT-1 downto 0)        => rx_dpa_locked,
        rx_out(NINPUT*10-1 downto 0)            => rx_out--,
        
--		rx_channel_data_align	=> rx_bitslip,
--		rx_fifo_reset		=> rx_fifo_reset,
--		rx_in			=> rx_in,
--		rx_inclock		=> rx_inclock_A_pll,--
--		rx_reset		=> rx_reset,
--		rx_dpa_locked		=> rx_dpa_locked,
--		rx_locked		=> rx_locked,
--		rx_out			=> rx_out,
--		rx_outclock		=> rx_clk
	);

	rx_ready <= rx_ready_reg;


-- flip bit order of received data (msb-lsb)
flip_bits: process(rx_out)
begin
for i in NINPUT-1 downto 0 loop
	for n in 9 downto 0 loop
		rx_out_order(10*i+n) <= INPUT_SIGNFLIP(i) xor rx_out(10*i+9-n);
	end loop;
end loop;
end process flip_bits;

gen_channels : for i in NINPUT-1 downto 0 generate
	datadec : entity work.data_decoder 
		generic map(
			EVAL_WINDOW_WORDCNT_BITS => 13,
			EVAL_WINDOW_PATTERN_BITS => 2,
			ALIGN_WORD	 	 => k28_0
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


		errcounter: entity work.rx_errcounter 
		port map(
			reset_n			=> reset_n_errcnt,
			clk			=> rx_clk,

			rx_sync			=> rx_ready_reg(i),
			rx_disperr		=> rx_disperr(i),

			o_runcounter		=> rx_runcounter(i),
			o_errcounter		=> rx_errorcounter(i),
			o_synclosscounter	=> rx_synclosscounter(i)

		);
end generate;



end rtl;
