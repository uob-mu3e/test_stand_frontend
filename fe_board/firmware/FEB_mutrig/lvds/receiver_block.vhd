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

use work.mudaq.all;

entity receiver_block is
generic (
	IS_SCITILE : std_logic := '1';
	NINPUT : positive := 1;
	LVDS_PLL_FREQ : real := 125.0;
	LVDS_DATA_RATE : real := 1250.0;
	INPUT_SIGNFLIP : std_logic_vector(31 downto 0) := x"00000000"--;
);
port (
	reset_n             : in std_logic;
	reset_n_errcnt      : in std_logic;
	rx_in               : in std_logic_vector(NINPUT-1 downto 0);
	rx_inclock_A        : in std_logic;
	rx_inclock_B        : in std_logic;
	rx_state            : out std_logic_vector(2*NINPUT-1 downto 0);
	rx_ready            : out std_logic_vector(NINPUT-1 downto 0);
	rx_data             : out std_logic_vector(NINPUT*8-1 downto 0);
	rx_k                : out std_logic_vector(NINPUT-1 downto 0);
	rx_clkout           : out std_logic;
	pll_locked          : out std_logic;

	rx_dpa_locked_out   : out std_logic_vector(NINPUT-1 downto 0);

	rx_runcounter       : out work.util.slv32_array_t(NINPUT-1 downto 0);
	rx_errorcounter     : out work.util.slv32_array_t(NINPUT-1 downto 0);
	rx_synclosscounter  : out work.util.slv32_array_t(NINPUT-1 downto 0)
);
end entity;

architecture rtl of receiver_block is

component data_decoder is
generic (
		EVAL_WINDOW_WORDCNT_BITS : natural := 8; -- number of bits of the counter used to check for the sync pattern
		EVAL_WINDOW_PATTERN_BITS : natural := 1; -- number of bits of the counter of the sync patterns found in the window (realign if not overflow)
		ALIGN_WORD	 : std_logic_vector(7 downto 0):=K28_5 -- pattern byte to search for
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
    signal rx_locked_A          : std_logic := '1';
    signal rx_dpaclock_A        : std_logic;
    signal rx_syncclock_A       : std_logic;
    signal rx_enable_A          : std_logic;

    signal rx_inclock_B_ctrl    : std_logic;
    signal rx_inclock_B_pll     : std_logic;
    signal rx_locked_B          : std_logic := '1';
    signal rx_dpaclock_B        : std_logic;
    signal rx_syncclock_B       : std_logic;
    signal rx_enable_B          : std_logic;

begin

    rx_dpa_locked_out   <= rx_dpa_locked;
    pll_locked          <= rx_pll_locked;
    rx_clkout           <= rx_clk;

-----------------------------------------------------------
---------------SciTile lvds rx-----------------------------
-----------------------------------------------------------
    gen_scitile: if (IS_SCITILE='1') generate
        rx_clk <= rx_syncclock_A;

        clk_ctrl_A : component work.cmp.clk_ctrl_single
            port map (
                inclk  => rx_inclock_A,
                outclk => rx_inclock_A_ctrl--,
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
            locked   => rx_locked_A
        );

        -- D4, D1, D2, D3, D7, D6, D9
        lvds_rx_A: entity work.lvds_receiver_small
        PORT MAP (
            pll_areset                              => not rx_locked_A,
            rx_channel_data_align                   => '0' & rx_bitslip(12 downto 7) & rx_bitslip(1 downto 0),
            rx_dpaclock                             => rx_dpaclock_A,
            rx_enable                               => rx_enable_A,
            rx_fifo_reset(7 downto 0)               => rx_fifo_reset(12 downto 7) & rx_fifo_reset(1 downto 0),
            rx_in(7 downto 0)                       => rx_in(12 downto 7) & rx_in(1 downto 0),
            rx_in(8 downto 8)                       => (others => '0'),
            rx_inclock                              => rx_inclock_A_pll,
            rx_reset(7 downto 0)                    => rx_reset(12 downto 7) & rx_reset(1 downto 0),
            rx_syncclock                            => rx_syncclock_A,
            rx_dpa_locked(7 downto 2)               => rx_dpa_locked(12 downto 7),
            rx_dpa_locked(1 downto 0)               => rx_dpa_locked(1 downto 0),
            rx_out(79 downto 20)                    => rx_out(129 downto 70),
            rx_out(19 downto  0)                    => rx_out(19 downto 0)--,
        );

        clk_ctrl_B : component work.cmp.clk_ctrl_single
            port map (
                inclk  => rx_inclock_B,
                outclk => rx_inclock_B_ctrl--,
        );

        lpll_B: entity work.lvdspll
        PORT MAP
        (
            refclk   => rx_inclock_B_ctrl,
            rst      => '0',
            outclk_0 => rx_inclock_B_pll,
            outclk_1 => rx_enable_B,
            outclk_2 => rx_syncclock_B,
            outclk_3 => rx_dpaclock_B,
            outclk_4 => open,
            locked   => rx_locked_B
        );

        -- C7, C8, C3, C6, C4, inclock_B
        lvds_rx_B: entity work.lvds_receiver_small
        PORT MAP
        (
            pll_areset                  => not rx_locked_B,
            rx_channel_data_align       => "0000" & rx_bitslip(6 downto 2),
            rx_dpaclock                 => rx_dpaclock_B,
            rx_enable                   => rx_enable_B,
            rx_fifo_reset(4 downto 0)   => rx_fifo_reset(6 downto 2),
            rx_in(4 downto 0)           => rx_in(6 downto 2),
            rx_in(8 downto 5)           => (others => '0'),
            rx_inclock                  => rx_inclock_B_pll,
            rx_reset(4 downto 0)        => rx_reset(6 downto 2),
            rx_syncclock                => rx_syncclock_B,
            rx_dpa_locked(4 downto 0)   => rx_dpa_locked(6 downto 2),
            rx_out(49 downto 0)         => rx_out(69 downto 20)--,
        );
    end generate;

-----------------------------------------------------------
---------------SciFi lvds rx-------------------------------
-----------------------------------------------------------
    gen_scifi: if (IS_SCITILE='0') generate
        rx_clk <= rx_syncclock_B;

        clk_ctrl_B : component work.cmp.clk_ctrl_single
            port map (
                inclk  => rx_inclock_B,
                outclk => rx_inclock_B_ctrl--,
        );

        lpll_B: entity work.lvdspll
        PORT MAP
        (
            refclk   => rx_inclock_B_ctrl,
            rst      => '0',
            outclk_0 => rx_inclock_B_pll,
            outclk_1 => rx_enable_B,
            outclk_2 => rx_syncclock_B,
            outclk_3 => rx_dpaclock_B,
            outclk_4 => open,
            locked   => rx_locked_B
        );

        -- C7, C8, C3, C6, C4, inclock_B
        lvds_rx_B: entity work.lvds_receiver_small
        PORT MAP
        (
            pll_areset                  => not rx_locked_B,
            rx_channel_data_align       => '0' & rx_bitslip(7 downto 0),
            rx_dpaclock                 => rx_dpaclock_B,
            rx_enable                   => rx_enable_B,
            rx_fifo_reset(7 downto 0)   => rx_fifo_reset(7 downto 0),
            rx_in(7 downto 0)           => rx_in(7 downto 0),
            rx_in(8 downto 8)           => (others => '0'),
            rx_inclock                  => rx_inclock_B_pll,
            rx_reset(7 downto 0)        => rx_reset(7 downto 0),
            rx_syncclock                => rx_syncclock_B,
            rx_dpa_locked(7 downto 0)   => rx_dpa_locked(7 downto 0),
            rx_out(79 downto 0)         => rx_out(79 downto 0)--,
        );
    end generate;

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

    gen_channels: for i in NINPUT-1 downto 0 generate
        datadec: data_decoder
		generic map(
			EVAL_WINDOW_WORDCNT_BITS => 13,
			EVAL_WINDOW_PATTERN_BITS => 2,
			ALIGN_WORD	 	 => K28_0
		)
		port map(
			reset_n			=> reset_n,
	--		checker_rst_n		=> checker_rst_n(i),
			clk			=> rx_clk,
			rx_in			=> rx_out_order((i+1)*10-1 downto i*10),

			rx_reset		=> rx_reset(i),
			rx_fifo_reset		=> rx_fifo_reset(i),
			rx_dpa_locked		=> rx_dpa_locked(i),
			rx_locked		=> rx_locked_A and rx_locked_B,
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
