----------------------------------------------------------------------------
-- Entity to talk to Firefly transceivers on V2 Frontent board 
-- opt. data 8TX, 4RX + alignment
-- LVDS data 2RX + alignment + 8b10b decoder
-- I2C reading of firefly regs
-- Avalon interface
--
-- Martin Mueller muellem@uni-mainz.de
--
-- Transceiver plan:
-- one 4-channel TX-only (all ffly2_tx)
-- one 4-channel RX/TX (ffly1_rx0/tx0, ffly2rx0/ffly1tx1, ffly2rx1/ffly1tx2, ffly2rx2/ffly1tx3)
-- one 2-channel LVDS-RX
--
-- Avalon channel map:
-- ch0 : ffly_1_tx_data_0 -- ffly_1_rx_data_0
-- ch1 : ffly_1_tx_data_1 -- ffly_2_rx_data_0
-- ch2 : ffly_1_tx_data_2 -- ffly_2_rx_data_1
-- ch3 : ffly_1_tx_data_3 -- ffly_2_rx_data_2
-- ch4 : ffly_2_tx_data_0 -- RX_CLK_1
-- ch5 : ffly_2_tx_data_1 -- RX_CLK_2
-- ch6 : ffly_2_tx_data_2 -- ffly_1_lvds_in
-- ch7 : ffly_2_tx_data_3 -- ffly_2_lvds_in
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


entity firefly is
    generic(
        I2C_DELAY_g             : positive := 50000000--;
    );
    port(
        i_clk                   : in    std_logic;
        i_sysclk                : in    std_logic;
        i_clk_i2c               : in    std_logic;
        o_clk_reco              : out   std_logic;
        i_clk_lvds              : in    std_logic;
        i_reset_n               : in    std_logic;
        i_reset_156_n           : in    std_logic;
        i_reset_125_rx_n        : in    std_logic;
        i_lvds_align_reset_n    : in    std_logic;
        
        --rx
        i_data_fast_serial      : in    std_logic_vector(      3 downto 0);
        o_data_fast_parallel    : out   std_logic_vector(32*3+31 downto 0);
        o_datak                 : out   std_logic_vector( 4*3+ 3 downto 0);
        
        --tx
        o_data_fast_serial      : out   std_logic_vector(      7 downto 0);
        i_data_fast_parallel    : in    std_logic_vector(32*7+31 downto 0);
        i_datak                 : in    std_logic_vector( 4*7+ 3 downto 0);
        
        --lvds rx
        i_data_lvds_serial      : in    std_logic_vector(1 downto 0);
        o_data_lvds_parallel    : out   std_logic_vector(15 downto 0);
        o_lvds_ready            : out   std_logic;
        
        --I2C
        i_i2c_enable            : in    std_logic;
        o_Mod_Sel_n             : out   std_logic_vector(1 downto 0);
        o_Rst_n                 : out   std_logic_vector(1 downto 0);
        io_scl                  : inout std_logic;
        io_sda                  : inout std_logic;
        i_int_n                 : in    std_logic_vector(1 downto 0);
        i_modPrs_n              : in    std_logic_vector(1 downto 0);
        
        -- avalon slave interface
        -- * 16 bit address space
        i_avs_address           : in    std_logic_vector(13 downto 0);
        i_avs_read              : in    std_logic;
        o_avs_readdata          : out   std_logic_vector(31 downto 0);
        i_avs_write             : in    std_logic;
        i_avs_writedata         : in    std_logic_vector(31 downto 0);
        o_avs_waitrequest       : out   std_logic;
        
        o_testclkout            : out   std_logic;
        o_testout               : out   std_logic;
		  
		  -- outputs to slow control
		  o_pwr                   : out   std_logic_vector(127 downto 0); -- RX optical power in mW
        o_temp                  : out   std_logic_vector(15 downto 0);  -- temperature in °C
		  o_alarm					  : out   std_logic_vector(63 downto 0);  -- latched alarm bits
		  o_vcc						  : out   std_logic_vector(31 downto 0)--;  -- operating voltagein units of 100 uV
		  
    );
end entity firefly;


architecture rtl of firefly is

-- fast rx transceiver status signals ----------------------
signal tx_clk           : std_logic_vector(3 downto 0):= (others => '0');
signal rx_clk           : std_logic_vector(3 downto 0):= (others => '0');
signal tx_clk2          : std_logic_vector(3 downto 0);

signal rx_data_parallel : std_logic_vector(32*3+31 downto 0);
signal rx_datak         : std_logic_vector(15 downto 0);

signal datak_not_aligned: std_logic_vector(15 downto 0);
signal data_not_aligned : std_logic_vector(127 downto 0);
signal enapatternalign  : std_logic_vector(3 downto 0);
signal syncstatus       : std_logic_vector(15 downto 0);
signal patterndetect    : std_logic_vector(15 downto 0);
signal errdetect        : std_logic_vector(15 downto 0);
signal disperr          : std_logic_vector(15 downto 0);

signal tx_analogreset1  : std_logic_vector(3 downto 0):= (others => '0');
signal tx_digitalreset1 : std_logic_vector(3 downto 0):= (others => '0');
signal rx_analogreset   : std_logic_vector(3 downto 0):= (others => '0');
signal rx_digitalreset  : std_logic_vector(3 downto 0):= (others => '0');

signal tx_analogreset2  : std_logic_vector(3 downto 0);
signal tx_digitalreset2 : std_logic_vector(3 downto 0);

signal pll_powerdown    : std_logic_vector(3 downto 0):= (others => '0');
signal pll_locked       : std_logic_vector(3 downto 0):= (others => '0');
signal pll_locked2      : std_logic_vector(3 downto 0);
signal pll_powerdown2   : std_logic_vector(3 downto 0);

signal tx_cal_busy      : std_logic_vector(3 downto 0):= (others => '0');
signal tx_cal_busy2     : std_logic_vector(3 downto 0);
signal rx_cal_busy      : std_logic_vector(3 downto 0):= (others => '0');

signal reconfig_to_xcvr_r   : std_logic_vector(559 downto 0):= (others => '0');
signal reconfig_from_xcvr_r : std_logic_vector(367 downto 0):= (others => '0');

signal reconfig_to_xcvr_r2  : std_logic_vector(559 downto 0);
signal reconfig_from_xcvr_r2: std_logic_vector(367 downto 0);

signal rx_is_lockedtodata   : std_logic_vector(  7 downto 0):= (others => '0');
signal rx_is_lockedtoref    : std_logic_vector(  7 downto 0):= (others => '0');

signal rx_align_reset_n     : std_logic_vector(3 downto 0);

-- lvds receiver control signals
signal lvds_pll_areset      : std_logic;
signal lvds_data_align      : std_logic;
signal lvds_dpa_lock_reset  : std_logic;
signal lvds_fifo_reset      : std_logic;
signal lvds_rx_reset        : std_logic;
signal lvds_cda_max         : std_logic;
signal lvds_dpa_locked      : std_logic;
signal lvds_rx_locked       : std_logic;
signal lvds_align_clicks    : std_logic_vector(7 downto 0);
signal lvds_o_ready         : std_logic;

-- lvds data signals
signal lvds_in_10b                      : std_logic_vector(9 downto 0);
signal lvds_8b10b_in                    : std_logic_vector(9 downto 0);
signal lvds_8b10b_out                   : std_logic_vector(7 downto 0);
signal lvds_rx_clk                      : std_logic;
signal lvds_8b10b_out_in_clk125_global  : std_logic_vector(7 downto 0);

-- avalon interface
signal av_ctrl              : work.util.avalon_t;
signal ch                   : integer range 0 to 7 := 0;
signal rx_seriallpbken      : std_logic_vector(7 downto 0);
signal tx_rst_n             : std_logic_vector(7 downto 0);
signal rx_rst_n             : std_logic_vector(7 downto 0);
signal tx_analogreset       : std_logic_vector(7 downto 0);
signal tx_digitalreset      : std_logic_vector(7 downto 0);
signal tx_ready             : std_logic_vector(7 downto 0);
signal rx_ready             : std_logic_vector(7 downto 0);
signal locked               : std_logic_vector(7 downto 0);

signal av_rx_data_parallel  : std_logic_vector(127 downto 0);
signal av_rx_datak          : std_logic_vector(15 downto 0);
signal av_rx_analogreset    : std_logic_vector(3 downto 0);
signal av_rx_digitalreset   : std_logic_vector(3 downto 0);
signal av_tx_ready          : std_logic_vector(7 downto 0);
signal av_opt_rx_power      : std_logic_vector(127 downto 0);
signal av_temperature       : std_logic_vector(15 downto 0);
signal av_alarms				 : std_logic_vector(63 downto 0);
signal av_vcc					 : std_logic_vector(31 downto 0);
signal av_rx_ready          : std_logic_vector(7 downto 0);
signal av_lvds_data         : std_logic_vector(7 downto 0);
signal av_locked            : std_logic_vector(7 downto 0);
signal av_rx_is_lockedtoref     : std_logic_vector(7 downto 0);
signal av_rx_is_lockedtodata    : std_logic_vector(7 downto 0);
signal av_syncstatus        : std_logic_vector(15 downto 0);
signal av_errdetect         : std_logic_vector(15 downto 0);
signal av_disperr           : std_logic_vector(15 downto 0);

-- Firefly status
signal temperature          : std_logic_vector(15 downto 0);
signal opt_rx_power         : std_logic_vector(127 downto 0);
signal alarms					 : std_logic_vector(63 downto 0);
signal vcc						 : std_logic_vector(31 downto 0);

begin

    o_Rst_n         <= (others => '1');--DO NOT DO THIS: (others => i_reset_n); !!! Phase will be not fixed
    o_clk_reco      <= lvds_rx_clk;
    o_lvds_ready    <= lvds_o_ready;

    process (i_clk)
    begin
        if rising_edge(i_clk) then
            -- spending a round of registers for timing improvement
            o_data_fast_parallel    <= av_rx_data_parallel;
            o_datak                 <= av_rx_datak;
        end if;    
    end process;

--------------------------------------------------
-- transceiver (2)
--------------------------------------------------

-- 4-channel RX/TX
    xcvr: component work.cmp.ip_altera_xcvr_native_av
    port map(
        --clocks
        tx_pll_refclk(0)        => i_clk,
        rx_cdr_refclk(0)        => i_clk,
        tx_std_coreclkin        => tx_clk,
        rx_std_coreclkin        => rx_clk,
        tx_std_clkout           => tx_clk,
        rx_std_clkout           => rx_clk,
        
        --resets
        tx_analogreset          => tx_analogreset1,
        tx_digitalreset         => tx_digitalreset1,
        rx_analogreset          => rx_analogreset,
        rx_digitalreset         => rx_digitalreset,
        
        -- tx data
        tx_serial_data          => o_data_fast_serial(3 downto 0),
        tx_parallel_data        => i_data_fast_parallel(32*3+31 downto 0),
        tx_datak                => i_datak(15 downto 0),
        
        -- rx data
        rx_serial_data          => i_data_fast_serial,
        rx_parallel_data        => data_not_aligned,
        rx_datak                => datak_not_aligned,
        
        -- control outputs
        rx_is_lockedtoref       => rx_is_lockedtoref(3 downto 0),
        rx_is_lockedtodata      => rx_is_lockedtodata(3 downto 0),
        pll_locked              => pll_locked,
        tx_cal_busy             => tx_cal_busy,
        rx_cal_busy             => rx_cal_busy,
        rx_errdetect            => errdetect,
        rx_disperr              => disperr,
        rx_runningdisp          => open,
        rx_patterndetect        => patterndetect,
        rx_syncstatus           => syncstatus,
        
        -- control inputs
        pll_powerdown           => pll_powerdown,
        rx_seriallpbken         => rx_seriallpbken(3 downto 0),
        rx_std_wa_patternalign  => enapatternalign,
        
        -- reconfig
        reconfig_to_xcvr        => reconfig_to_xcvr_r,
        reconfig_from_xcvr      => reconfig_from_xcvr_r,
        
        unused_tx_parallel_data => (others => '0'),
        unused_rx_parallel_data => open--,
    );
    
-- 4-channel TX
    xcvr2: component work.cmp.fastlink_small
    port map(
        --clocks
        tx_pll_refclk(0)        => i_clk,
        tx_std_coreclkin        => tx_clk2,
        tx_std_clkout           => tx_clk2,
        
        --resets
        tx_analogreset          => tx_analogreset2,
        tx_digitalreset         => tx_digitalreset2,
        
        -- tx data
        tx_serial_data          => o_data_fast_serial(7 downto 4),
        tx_parallel_data        => i_data_fast_parallel(32*3+31 downto 0),
        tx_datak                => i_datak(15 downto 0),
        
        -- control outputs
        pll_locked              => pll_locked2,
        tx_cal_busy             => tx_cal_busy2,
        
        -- control inputs
        pll_powerdown           => pll_powerdown2,
        
        -- reconfig
        reconfig_to_xcvr        => reconfig_to_xcvr_r2,
        reconfig_from_xcvr      => reconfig_from_xcvr_r2,
        
        unused_tx_parallel_data => (others => '0')--,
    );

--------------------------------------------------
-- rx byte alignment (4)
--------------------------------------------------
    g_rx_align: for I in 0 to 3 generate
        e_rx_align: entity work.rx_align
        port map(
            o_data                  => rx_data_parallel(31+I*32 downto I*32),
            o_datak                 => rx_datak(3+I*4 downto I*4),

            o_locked                => locked(I),

            i_data                  => data_not_aligned(31+I*32 downto I*32),
            i_datak                 => datak_not_aligned(3+I*4 downto I*4),
            
            i_syncstatus            => syncstatus(3+I*4 downto I*4),
            i_patterndetect         => patterndetect(3+I*4 downto I*4),
            o_enapatternalign       => enapatternalign(I),

            i_errdetect             => errdetect(3+I*4 downto I*4),
            i_disperr               => disperr(3+I*4 downto I*4),

            i_reset_n               => rx_align_reset_n(I),
            i_clk                   => rx_clk(I)--,
        );

        e_sync_align_xcvr : entity work.reset_sync
        port map ( o_reset_n => rx_align_reset_n(I), i_reset_n => i_reset_156_n, i_clk => rx_clk(I));

    end generate g_rx_align;

--------------------------------------------------
-- reset controller (2)
--------------------------------------------------

    reset_controller: component work.cmp.ip_altera_xcvr_reset_control
    port map(
        clock                   => i_sysclk,
        reset                   => not i_reset_n,
        pll_powerdown           => pll_powerdown,
        tx_analogreset          => tx_analogreset1,
        tx_digitalreset         => tx_digitalreset1,
        tx_ready                => tx_ready(3 downto 0),
        pll_locked              => pll_locked,
        pll_select              => (others => '0'),
        tx_cal_busy             => tx_cal_busy,
        rx_analogreset          => rx_analogreset,
        rx_digitalreset         => rx_digitalreset,
        rx_ready                => rx_ready(3 downto 0),
        rx_is_lockedtodata      => rx_is_lockedtodata(3 downto 0),
        rx_cal_busy             => rx_cal_busy--,
    );
    
    reset_controller2: component work.cmp.native_reset_tx
    port map(
        clock                   => i_sysclk,
        reset                   => not i_reset_n,
        pll_powerdown           => pll_powerdown2,
        tx_analogreset          => tx_analogreset2,
        tx_digitalreset         => tx_digitalreset2,
        tx_ready                => tx_ready(7 downto 4),
        pll_locked              => pll_locked2,
        pll_select              => (others => '0'),
        tx_cal_busy             => tx_cal_busy2--,
    );

--------------------------------------------------
-- reconfig controller (2)
--------------------------------------------------

    reconfig_controller: component work.cmp.ip_alt_xcvr_reconfig
    port map(
        reconfig_busy             => open,
        mgmt_clk_clk              => i_sysclk,
        mgmt_rst_reset            => not i_reset_n,
        reconfig_mgmt_address     => (others => '0'),
        reconfig_mgmt_read        => '0',
        reconfig_mgmt_readdata    => open,
        reconfig_mgmt_waitrequest => open,
        reconfig_mgmt_write       => '0',
        reconfig_mgmt_writedata   => (others => '0'),
        ch0_7_to_xcvr             => reconfig_to_xcvr_r,
        ch0_7_from_xcvr           => reconfig_from_xcvr_r,
        ch8_15_to_xcvr            => reconfig_to_xcvr_r2,
        ch8_15_from_xcvr          => reconfig_from_xcvr_r2--,
    );

--------------------------------------------------
-- lvds receiver + alignment and 8b10b decode
--------------------------------------------------

    lvds_rx_inst0 : entity work.lvds_rx
    port map(
        pll_areset                  => lvds_pll_areset,
        rx_channel_data_align(0)    => lvds_data_align,
        rx_dpa_lock_reset(0)        => lvds_dpa_lock_reset,
        rx_fifo_reset(0)            => lvds_fifo_reset,
        rx_in(0)                    => i_data_lvds_serial(0),
        rx_inclock                  => i_clk_lvds,
        rx_reset(0)                 => lvds_rx_reset,
        rx_cda_max(0)               => lvds_cda_max,
        rx_dpa_locked(0)            => lvds_dpa_locked,
        rx_locked                   => lvds_rx_locked, 
        rx_out                      => lvds_in_10b,
        rx_outclock                 => lvds_rx_clk--,
    ); 

    e_lvds_controller : entity work.lvds_controller 
    port map(
        i_clk               => i_clk_lvds,                      -- controller MUST run on 125 Global. DO NOT CHANGE TO lvds_rx_clk !!!
        i_areset_n          => i_lvds_align_reset_n,
        i_data              => lvds_8b10b_out_in_clk125_global, -- feed alignment with 8b10b decoded data in global clk domain
        i_cda_max           => lvds_cda_max,
        i_dpa_locked        => lvds_dpa_locked,
        i_rx_locked         => lvds_rx_locked,
        o_ready             => lvds_o_ready,
        o_data_align        => lvds_data_align,
        o_pll_areset        => lvds_pll_areset,
        o_dpa_lock_reset    => lvds_dpa_lock_reset,
        o_fifo_reset        => lvds_fifo_reset,
        o_rx_reset          => lvds_rx_reset,
        o_cda_reset         => open, --not available on ArriaV
        o_align_clicks      => lvds_align_clicks
    );

    process (lvds_rx_clk)
    begin
        if rising_edge(lvds_rx_clk) then
            lvds_8b10b_in                       <= lvds_in_10b;
        end if;    
    end process;

    udec_8b10b : entity work.dec_8b10b_old
    port map(
        RESET => not i_reset_125_rx_n,
        RBYTECLK => lvds_rx_clk,
        AI => lvds_8b10b_in(9),
        BI => lvds_8b10b_in(8),
        CI => lvds_8b10b_in(7),
        DI => lvds_8b10b_in(6),
        EI => lvds_8b10b_in(5),
        II => lvds_8b10b_in(4),
        FI => lvds_8b10b_in(3),
        GI => lvds_8b10b_in(2),
        HI => lvds_8b10b_in(1), 
        JI => lvds_8b10b_in(0),
        KO => open,--TODO: datak,
        HO => lvds_8b10b_out(7),
        GO => lvds_8b10b_out(6),
        FO => lvds_8b10b_out(5),
        EO => lvds_8b10b_out(4), 
        DO => lvds_8b10b_out(3),
        CO => lvds_8b10b_out(2),
        BO => lvds_8b10b_out(1),
        AO => lvds_8b10b_out(0)
    );

    -- sync 8b10b_out properly into i_clk_lvds for alignment in lvds_controller (e_fifo8b)
    -- forward the "not-synced" signal to state controller (running on reconstructed clock lvds_rx_clk)
    o_data_lvds_parallel(7 downto 0)    <= lvds_8b10b_out;

--------------------------------------------------
-- I2C reading
--------------------------------------------------

    firefly_i2c: entity work.firefly_i2c
    generic map(
        I2C_DELAY_g     => I2C_DELAY_g--,
    )
    port map(
        i_clk           => i_clk_i2c,
        i_reset_n       => i_reset_n,
        i_i2c_enable    => i_i2c_enable,
        o_Mod_Sel_n     => o_Mod_Sel_n,
        io_scl          => io_scl,
        io_sda          => io_sda,
        i_int_n         => i_int_n,
        i_modPrs_n      => i_modPrs_n,

        o_pwr           => opt_rx_power,
        o_temp          => temperature,
		  o_alarm			=> alarms,
		  o_vcc				=> vcc--,
    );

	 --o_pwr 	<= opt_rx_power;
	 --o_temp 	<= temperature;
	 
--    dnca: entity work.doNotCompileAwayMux
--    generic map(
--        WIDTH_g   => 31--,
--    )
--    port map(
--        i_clk               => i_clk_i2c,
--        i_reset_n           => i_reset_n,
--        i_doNotCompileAway  => i2c_data & lvds_8b10b_out_in_clk125_global,
--        o_led               => o_testout--,
--    );

--------------------------------------------------
-- Avalon
--------------------------------------------------

    -- av_ctrl process, avalon iface
    process(i_clk, i_reset_156_n)
    begin
    if ( i_reset_156_n = '0' ) then
        av_ctrl.waitrequest <= '1';
        ch <= 0;
        rx_seriallpbken <= (others => '0');
        tx_rst_n <= (others => '1');
        rx_rst_n <= (others => '1');

    elsif rising_edge(i_clk) then
        av_ctrl.waitrequest <= '1';

        tx_rst_n <= (others => '1');
        rx_rst_n <= (others => '1');

        if ( av_ctrl.read /= av_ctrl.write and av_ctrl.waitrequest = '1' ) then
            av_ctrl.waitrequest <= '0';

            av_ctrl.readdata <= (others => '0');
            case av_ctrl.address(7 downto 0) is
            when X"00" =>
                -- channel select
                av_ctrl.readdata(7 downto 0) <= std_logic_vector(to_unsigned(ch, 8));
                if ( av_ctrl.write = '1' and av_ctrl.writedata(7 downto 0) < 8 ) then
                    ch <= to_integer(unsigned(av_ctrl.writedata(7 downto 0)));
                end if;
                --
            when X"01" =>
                av_ctrl.readdata(7 downto 0) <= std_logic_vector(to_unsigned(8, 8));
            when X"02" =>
                av_ctrl.readdata(7 downto 0) <= std_logic_vector(to_unsigned(32, 8));
            when X"10" =>
                -- tx reset
                av_ctrl.readdata(0) <= tx_analogreset(ch);
                av_ctrl.readdata(4) <= tx_digitalreset(ch);
                if ( av_ctrl.write = '1' ) then tx_rst_n(ch) <= not av_ctrl.writedata(0); end if;
                --
            when X"11" =>
                -- tx status
                av_ctrl.readdata(0) <= av_tx_ready(ch);
                --
            when X"12" =>
                -- tx errors
                av_ctrl.readdata(8) <= '0';--tx_fifo_error(ch);
                --
            when X"20" =>
                -- rx reset
                if(ch < 4) then
                    av_ctrl.readdata(0) <= av_rx_analogreset(ch);
                    av_ctrl.readdata(4) <= av_rx_digitalreset(ch);
                else
                    av_ctrl.readdata(0) <= '0';
                    av_ctrl.readdata(4) <= '0';
                end if;
                if ( av_ctrl.write = '1' ) then rx_rst_n(ch) <= not av_ctrl.writedata(0); end if;
                --
            when X"21" =>
                -- rx status
                av_ctrl.readdata(0) <= av_rx_ready(ch);
                av_ctrl.readdata(1) <= av_rx_is_lockedtoref(ch);
                av_ctrl.readdata(2) <= av_rx_is_lockedtodata(ch);
                -- av_ctrl.readdata(11 downto 8) <= (others => '1');
                av_ctrl.readdata(32/8-1 + 8 downto 8) <= av_syncstatus(ch*4+3 downto ch*4);
                av_ctrl.readdata(12) <= av_locked(ch);
                --
            when X"22" =>
                -- rx errors
                if(ch < 4) then 
                    av_ctrl.readdata(3 downto 0) <= av_errdetect(4*ch+3 downto 4*ch);
                    av_ctrl.readdata(7 downto 4) <= av_disperr(4*ch+3 downto 4*ch);
                else
                    av_ctrl.readdata(3 downto 0) <= (others => '0');
                    av_ctrl.readdata(7 downto 4) <= (others => '0');
                end if;
                av_ctrl.readdata(8) <= '0';--rx_fifo_error(ch);
                --
            when X"23" =>
                av_ctrl.readdata(31 downto 0) <= (others => '0');--rx(ch).LoL_cnt;
            when X"24" =>
                av_ctrl.readdata(31 downto 0) <= (others => '0');--rx(ch).err_cnt;
                --
            when X"25" =>
                av_ctrl.readdata(31 downto 0) <= x"0000" & av_opt_rx_power(16*ch+15 downto 16*ch);-- RX optical power
            when X"26" =>
                if(ch = 0 or ch = 4 or ch = 5 or ch = 6 ) then
                    av_ctrl.readdata(31 downto 0) <= x"000000" & av_temperature(7 downto 0);-- Firefly temperature
                else
                    av_ctrl.readdata(31 downto 0) <= x"000000" & av_temperature(15 downto 8);
                end if;
                --
            when X"2A" =>
                if(ch < 4) then
                    av_ctrl.readdata(31 downto 0) <= av_rx_data_parallel(32*ch+31 downto 32*ch);
                elsif(ch = 6) then
                    av_ctrl.readdata(31 downto 0) <= x"000000" & av_lvds_data;
                else
                    av_ctrl.readdata(31 downto 0) <= (others => '0');
                end if;
            when X"2B" =>
                if(ch<4) then
                    av_ctrl.readdata(31 downto 0) <= x"0000000" & av_rx_datak(4*ch+3 downto ch*4);
                else
                    av_ctrl.readdata(31 downto 0) <= (others => '0');
                end if;
            when X"2C" =>
                av_ctrl.readdata(31 downto 0) <= (others => '0'); --rx(ch).Gbit;
                --
            when X"2F" =>
                av_ctrl.readdata(0) <= rx_seriallpbken(ch);
                if ( av_ctrl.write = '1' ) then rx_seriallpbken(ch) <= av_ctrl.writedata(0); end if;
                --
            when others =>
                av_ctrl.readdata <= X"CCCCCCCC";
                --
            end case;
        end if;

    end if; -- rising_edge
    end process;

    -- avalon control block
    b_avs : block
        signal av_ctrl_cs : std_logic;
        signal avs_waitrequest_i : std_logic;
    begin
        av_ctrl_cs <= '1' when ( i_avs_address(i_avs_address'left downto 8) = "000000" ) else '0';
        av_ctrl.address(i_avs_address'range) <= i_avs_address;
        av_ctrl.writedata <= i_avs_writedata;

        o_avs_waitrequest <= avs_waitrequest_i;

        process(i_clk, i_reset_156_n)
        begin
        if ( i_reset_156_n = '0' ) then
            avs_waitrequest_i <= '1';
            av_ctrl.read <= '0';
            av_ctrl.write <= '0';
            --
        elsif rising_edge(i_clk) then
            avs_waitrequest_i <= '1';

            if ( i_avs_read /= i_avs_write and avs_waitrequest_i = '1' ) then
                if ( av_ctrl_cs = '1' ) then
                    if ( av_ctrl.read = av_ctrl.write ) then
                        av_ctrl.read <= i_avs_read;
                        av_ctrl.write <= i_avs_write;
                    elsif ( av_ctrl.waitrequest = '0' ) then
                        o_avs_readdata <= av_ctrl.readdata;
                        avs_waitrequest_i <= '0';
                        av_ctrl.read <= '0';
                        av_ctrl.write <= '0';
                    end if;
                else
                    o_avs_readdata <= X"CCCCCCCC";
                    avs_waitrequest_i <= '0';
                end if;
            end if;
            --
        end if;
        end process;
    end block;

--------------------------------------------------
-- Sync FIFO's
--------------------------------------------------
    sync_fifo1 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4,
        DATA_WIDTH  => 8,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr    => lvds_fifo_reset,
        data    => lvds_8b10b_out,
        rdclk   => i_clk_lvds,
        rdreq   => '1',
        wrclk   => lvds_rx_clk,
        wrreq   => '1',
        q       => lvds_8b10b_out_in_clk125_global--,
    );

    sync_fifo2 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4,
        DATA_WIDTH  => 220,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => not i_lvds_align_reset_n,--'0',
        data            =>  disperr
                            & errdetect & syncstatus
                            & rx_is_lockedtodata & rx_is_lockedtoref 
                            & locked(3 downto 0) & x"00"--tx_ready 
                            & rx_data_parallel & rx_datak,
        rdclk           => i_clk,
        rdreq           => '1',
        wrclk           => rx_clk(0),
        wrreq           => '1',
        q(15 downto 0)      => av_rx_datak,
        q(143 downto 16)    => av_rx_data_parallel,
        q(151 downto 144)   => open,--av_tx_ready,
        q(155 downto 152)   => av_locked(3 downto 0),
        q(163 downto 156)   => av_rx_is_lockedtoref,
        q(171 downto 164)   => av_rx_is_lockedtodata,
        q(187 downto 172)   => av_syncstatus,
        q(203 downto 188)   => av_errdetect,
        q(219 downto 204)   => av_disperr--,
    );

    sync_fifo3 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
        DATA_WIDTH  => 280,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            =>  alarms & vcc
									 & tx_ready & rx_ready
                            & opt_rx_power & temperature
                            & rx_analogreset & rx_digitalreset 
                            & tx_analogreset2 & tx_analogreset1 
                            & tx_digitalreset2 & tx_digitalreset1,
        rdclk           => i_clk,
        rdreq           => '1',
        wrclk           => i_sysclk,
        wrreq           => '1',
        q(7 downto 0)       => tx_digitalreset,
        q(15 downto 8)      => tx_analogreset,
        q(19 downto 16)     => av_rx_digitalreset,
        q(23 downto 20)     => av_rx_analogreset,
        q(39 downto 24)     => av_temperature,
        q(167 downto 40)    => av_opt_rx_power,
        q(175 downto 168)   => av_rx_ready,
        q(183 downto 176)   => av_tx_ready,
		  q(215 downto 184)	 => av_vcc,
		  q(279 downto 216)	 => av_alarms
    );
	 
	 o_pwr 	<= av_opt_rx_power;
	 o_temp 	<= av_temperature;
	 o_vcc	<= av_vcc;
	 o_alarm <= av_alarms;

    sync_fifo4 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
        DATA_WIDTH  => 9,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            => lvds_8b10b_out_in_clk125_global & lvds_o_ready,
        rdclk           => i_clk,
        rdreq           => '1',
        wrclk           => i_clk_lvds,
        wrreq           => '1',
        q(0)            => av_locked(6),
        q(8 downto 1)   => av_lvds_data--,
    );
end rtl;
