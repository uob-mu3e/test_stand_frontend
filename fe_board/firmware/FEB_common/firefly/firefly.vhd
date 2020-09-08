----------------------------------------------------------------------------
-- Entity to talk to Firefly transceivers on V2 Frontent board 
-- opt. data 8TX, 4RX + alignment
-- LVDS data 2RX + alignment + 8b10b decoder
-- I2C reading of firefly regs
--
-- Martin Mueller muellem@uni-mainz.de
--
-- Transceiver plan:
-- one 4-channel TX-only (all ffly2_tx)
-- one 4-channel RX/TX (ffly1_rx0/tx0, ffly2rx0/ffly1tx1, ffly2rx1/ffly1tx2, ffly2rx2/ffly1tx3)
-- one 2-channel LVDS-RX
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.transceiver_components.all;
use work.firefly_constants.all;
use work.daq_constants.all;

entity firefly is
    generic(
        STARTADDR_g             : positive := 1;
        I2C_DELAY_g             : positive := 50000000--;
    );
    port(
        i_clk                   : in    std_logic;
        i_sysclk                : in    std_logic;
        i_clk_i2c               : in    std_logic;
        o_clk_reco              : out   std_logic;
        i_clk_lvds              : in    std_logic;
        i_reset_n               : in    std_logic;
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
        
        --I2C
        i_i2c_enable            : in    std_logic;
        o_Mod_Sel_n             : out   std_logic_vector(1 downto 0);
        o_Rst_n                 : out   std_logic_vector(1 downto 0);
        io_scl                  : inout std_logic;
        io_sda                  : inout std_logic;
        i_int_n                 : in    std_logic_vector(1 downto 0);
        i_modPrs_n              : in    std_logic_vector(1 downto 0);
        
        o_testclkout            : out   std_logic;
        o_testout               : out   std_logic--;
    );
end entity firefly;


architecture rtl of firefly is

-- fast rx transceiver status signals ----------------------
signal tx_clk           : std_logic_vector(3 downto 0):= (others => '0');
signal rx_clk           : std_logic_vector(3 downto 0):= (others => '0');
signal tx_clk2          : std_logic_vector(3 downto 0);

signal datak_not_aligned: std_logic_vector(15 downto 0);
signal data_not_aligned : std_logic_vector(127 downto 0);
signal enapatternalign  : std_logic_vector(3 downto 0);
signal syncstatus       : std_logic_vector(15 downto 0);
signal patterndetect    : std_logic_vector(15 downto 0);
signal errdetect        : std_logic_vector(15 downto 0);
signal disperr          : std_logic_vector(15 downto 0);

signal tx_analogreset   : std_logic_vector(3 downto 0):= (others => '0');
signal tx_digitalreset  : std_logic_vector(3 downto 0):= (others => '0');
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

signal rx_is_lockedtodata   : std_logic_vector(  3 downto 0):= (others => '0');

-- i2c signals --------------------------------------------
signal i2c_rw               : std_logic;
signal i2c_ena              : std_logic;
signal i2c_busy             : std_logic;
signal i2c_busy_prev        : std_logic;
signal i2c_addr             : std_logic_vector(6 downto 0);
signal i2c_data_rd          : std_logic_vector(7 downto 0);
signal i2c_data_wr          : std_logic_vector(7 downto 0);
type   i2c_state_type         is (idle, waiting1, i2cffly1, waiting2, i2cffly2);
signal i2c_state            : i2c_state_type;
signal i2c_data             : std_logic_vector(23 downto 0);
signal i2c_counter          : unsigned(31 downto 0);

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

-- lvds data signals
signal lvds_in_10b                      : std_logic_vector(9 downto 0);
signal lvds_8b10b_in                    : std_logic_vector(9 downto 0);
signal lvds_8b10b_out                   : std_logic_vector(7 downto 0);
signal lvds_rx_clk                      : std_logic;
signal lvds_8b10b_out_in_clk125_global  : std_logic_vector(7 downto 0);

begin

    o_Rst_n     <= (others => '1');--DO NOT DO THIS: (others => i_reset_n); !!! Phase will be not fixed
    o_testclkout<= lvds_rx_clk;

--------------------------------------------------
-- transceiver (2)
--------------------------------------------------

-- 4-channel RX/TX
    xcvr: entity work.ip_altera_xcvr_native_av
    port map(
        --clocks
        tx_pll_refclk(0)        => i_clk,
        rx_cdr_refclk(0)        => i_clk,
        tx_std_coreclkin        => tx_clk,
        rx_std_coreclkin        => rx_clk,
        tx_std_clkout           => tx_clk,
        rx_std_clkout           => rx_clk,
        
        --resets
        tx_analogreset          => tx_analogreset,
        tx_digitalreset         => tx_digitalreset,
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
        rx_is_lockedtoref       => open,
        rx_is_lockedtodata      => rx_is_lockedtodata,
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
        rx_seriallpbken         => (others => '0'),
        rx_std_wa_patternalign  => enapatternalign,
        
        -- reconfig
        reconfig_to_xcvr        => reconfig_to_xcvr_r,
        reconfig_from_xcvr      => reconfig_from_xcvr_r,
        
        unused_tx_parallel_data => (others => '0'),
        unused_rx_parallel_data => open--,
    );
    
-- 4-channel TX
    xcvr2: entity work.fastlink_small
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
            o_data                  => o_data_fast_parallel(31+I*32 downto I*32),
            o_datak                 => o_datak(3+I*4 downto I*4),

            o_locked                => open,

            i_data                  => data_not_aligned(31+I*32 downto I*32),
            i_datak                 => datak_not_aligned(3+I*4 downto I*4),
            
            i_syncstatus            => syncstatus(3+I*4 downto I*4),
            i_patterndetect         => patterndetect(3+I*4 downto I*4),
            o_enapatternalign       => enapatternalign(I),

            i_errdetect             => errdetect(3+I*4 downto I*4),
            i_disperr               => disperr(3+I*4 downto I*4),

            i_reset_n               => i_reset_n,
            i_clk                   => rx_clk(I)--,
        );
    end generate g_rx_align;

--------------------------------------------------
-- reset controller (2)
--------------------------------------------------

    reset_controller: entity work.ip_altera_xcvr_reset_control
    port map(
        clock                   => i_sysclk,
        reset                   => not i_reset_n,
        pll_powerdown           => pll_powerdown,
        tx_analogreset          => tx_analogreset,
        tx_digitalreset         => tx_digitalreset,
        tx_ready                => open,
        pll_locked              => pll_locked,
        pll_select              => (others => '0'),
        tx_cal_busy             => tx_cal_busy,
        rx_analogreset          => rx_analogreset,
        rx_digitalreset         => rx_digitalreset,
        rx_ready                => open,
        rx_is_lockedtodata      => rx_is_lockedtodata,
        rx_cal_busy             => rx_cal_busy--,
    );
    
    reset_controller2: entity work.native_reset_tx
    port map(
        clock                   => i_sysclk,
        reset                   => not i_reset_n,
        pll_powerdown           => pll_powerdown2,
        tx_analogreset          => tx_analogreset2,
        tx_digitalreset         => tx_digitalreset2,
        tx_ready                => open,
        pll_locked              => pll_locked2,
        pll_select              => (others => '0'),
        tx_cal_busy             => tx_cal_busy2--,
    );

--------------------------------------------------
-- reconfig controller (2)
--------------------------------------------------

    reconfig_controller: entity work.ip_alt_xcvr_reconfig
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
        i_reset_n           => i_lvds_align_reset_n,--i_reset_n,
        i_data              => lvds_8b10b_out_in_clk125_global, -- feed alignment with 8b10b decoded data in global clk domain
        i_cda_max           => lvds_cda_max,
        i_dpa_locked        => lvds_dpa_locked,
        i_rx_locked         => lvds_rx_locked,
        o_ready             => open,
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

    udec_8b10b : entity work.dec_8b10b 
    port map(
        RESET => not i_reset_n,
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

    e_fifo8b : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
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
        q       => lvds_8b10b_out_in_clk125_global
    );

    o_data_lvds_parallel(7 downto 0)    <= lvds_8b10b_out_in_clk125_global;

--------------------------------------------------
-- I2C reading
--------------------------------------------------

    firefly_i2c: entity work.i2c_master
    generic map(
        input_clk   => 50_000_000,  --input clock speed from user logic in Hz
        bus_clk     => 400_000--,   --speed the i2c bus (scl) will run at in Hz
    )
    port map(
        clk         => i_clk_i2c,
        reset_n     => i_reset_n,
        ena         => i2c_ena,
        addr        => i2c_addr,
        rw          => i2c_rw,
        data_wr     => i2c_data_wr,
        busy        => i2c_busy,
        data_rd     => i2c_data_rd,
        ack_error   => open,
        sda         => io_sda,
        scl         => io_scl--,
    );

    process(i_clk_i2c, i_reset_n)
    variable busy_cnt           : integer := 0;
    begin
        if(i_reset_n = '0') then
            i2c_state       <= idle;
            i2c_ena         <= '0';
            o_Mod_Sel_n     <= "11";
            i2c_rw          <= '1';
            i2c_busy_prev   <= '0';
            i2c_counter     <= (others => '0');
            
        elsif(rising_edge(i_clk_i2c)) then
            case i2c_state is
                when idle =>
                    o_Mod_Sel_n     <= "11";
                    i2c_counter     <= (others => '0');
                    if(i_i2c_enable = '1') then 
                        i2c_state       <= waiting1;
                    end if;
                when waiting1 =>
                    o_Mod_Sel_n(0)  <= '0'; -- want to talk to firefly 1 (active low)
                    o_Mod_Sel_n(1)  <= '1';
                    i2c_counter     <= i2c_counter + 1;
                    
                    if(i2c_counter = I2C_DELAY_g) then -- wait for assert time of mod_sel (a few hundred ms)
                        i2c_state       <= i2cffly1;
                    end if;
                    
                when i2cffly1 => -- i2c transaction with firefly 1
                    i2c_busy_prev   <= i2c_busy;
                    i2c_counter     <= (others => '0');
                    if(i2c_busy_prev = '0' AND i2c_busy = '1') then
                        busy_cnt := busy_cnt + 1;
                    end if;
                    
                    case busy_cnt is
                        when 0 =>
                            i2c_ena     <= '1';
                            i2c_addr    <= FFLY_DEV_ADDR_7;
                            i2c_rw      <= '0'; -- 0: write, 1: read
                            i2c_data_wr <= ADDR_TEMPERATURE;
                        when 1 =>
                            i2c_rw      <= '1';
                        when 2 =>
                            i2c_rw      <= '0';
                            i2c_data_wr <= RX1_PWR_1;
                            if(i2c_busy = '0') then
                                i2c_data(7 downto 0) <= i2c_data_rd; -- read data from busy_cnt = 1
                            end if;
                        when 3 =>
                            i2c_rw      <= '1';
                        when 4 =>
                            i2c_rw      <= '0';
                            i2c_data_wr <= RX1_PWR_2;
                            if(i2c_busy = '0') then
                                i2c_data(15 downto 8) <= i2c_data_rd; -- read data from busy_cnt = 1
                            end if;
                        when 5 =>
                            i2c_rw      <= '1';
                        when 6 =>
                            i2c_ena     <= '0';
                            if(i2c_busy = '0') then
                                i2c_data(23 downto 16)  <= i2c_data_rd;
                                busy_cnt                := 0;
                                i2c_state               <= waiting2;
                            end if;
                        when others => null;
                    end case;
                    
                when waiting2 =>
                    o_Mod_Sel_n(1)  <= '0';
                    o_Mod_Sel_n(0)  <= '1';
                    i2c_state       <= i2cffly2;
                    i2c_counter     <= i2c_counter + 1;
                    
                    if(i2c_counter = I2C_DELAY_g) then -- wait for assert time of mod_sel (a few hundred ms)
                        i2c_state       <= i2cffly1;
                    end if;
                when i2cffly2 => -- i2c transaction with firefly 2
                --todo: insert same thing when ffly1 is working
                    i2c_state       <= idle;
                    i2c_counter     <= (others => '0');
                    
                when others =>
                    i2c_state       <= idle;
            end case;
        end if;
    end process;


    dnca: entity work.doNotCompileAwayMux
    generic map(
        WIDTH_g   => 31--,
    )
    port map(
        i_clk               => i_clk_i2c,
        i_reset_n           => i_reset_n,
        i_doNotCompileAway  => i2c_data & lvds_8b10b_out_in_clk125_global,
        o_led               => o_testout--,
    );

end rtl;