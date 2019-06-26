library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity xcvr_s4 is
generic (
    NUMBER_OF_CHANNELS_g : positive := 4;
    CHANNEL_WIDTH_g : positive := 32;
    INPUT_CLOCK_FREQUENCY_g : positive := 125000000;
    DATA_RATE_g : positive := 5000;
    K_g : std_logic_vector(7 downto 0) := work.util.D28_5;
    CLK_MHZ_g : positive := 50--;
);
port (
    -- avalon slave interface
    i_avs_address       : in    std_logic_vector(13 downto 0);
    i_avs_read          : in    std_logic;
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic;
    i_avs_writedata     : in    std_logic_vector(31 downto 0);
    o_avs_waitrequest   : out   std_logic;

    i_tx_data       : in    std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    i_tx_datak      : in    std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    o_rx_data       : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    o_rx_datak      : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);

    o_tx_clkout     : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    i_tx_clkin      : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    o_rx_clkout     : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    i_rx_clkin      : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

--    o_tx_ready      : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
--    o_rx_ready      : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    o_tx_serial     : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    i_rx_serial     : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    i_pll_clk       : in    std_logic;
    i_cdr_clk       : in    std_logic;

    i_reset         : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of xcvr_s4 is

    signal reset_n : std_logic;

    signal ch : integer range NUMBER_OF_CHANNELS_g-1 downto 0;

    signal av_ctrl : work.mu3e.avalon_t;

    signal rx_data              :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    signal rx_datak             :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);

    signal tx_rst_n, rx_rst_n   :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    signal pll_powerdown        :   std_logic_vector(0 downto 0);
    signal pll_cal_busy         :   std_logic_vector(0 downto 0);
    signal pll_locked           :   std_logic_vector(0 downto 0);

    signal tx_analogreset       :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal tx_digitalreset      :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_analogreset       :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_digitalreset      :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    signal tx_ready             :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_ready             :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_is_lockedtoref    :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_is_lockedtodata   :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    signal tx_fifo_error        :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_fifo_error        :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_errdetect         :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    signal rx_disperr           :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);

    signal rx_syncstatus        :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    signal rx_patterndetect     :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    signal rx_enapatternalign   :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    signal rx_seriallpbken      :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    signal reconfig_togxb       :   std_logic_vector(3 downto 0);
    signal reconfig_fromgxb     :   std_logic_vector(16 downto 0);
    signal reconfig_busy        :   std_logic;
    signal reconfig_error       :   std_logic;
    signal reconfig_rst_n       :   std_logic;
    signal reconfig_clk         :   std_logic;



    type rx_t is record
        data    :   std_logic_vector(CHANNEL_WIDTH_g-1 downto 0);
        datak   :   std_logic_vector(CHANNEL_WIDTH_g/8-1 downto 0);
        locked  :   std_logic;
        rst_n   :   std_logic;

        -- Gbit counter
        Gbit    :   std_logic_vector(23 downto 0);
        -- loss-of-lock counter
        LoL_cnt :   std_logic_vector(7 downto 0);
        -- error counter
        err_cnt :   std_logic_vector(15 downto 0);

        syncstatus      :   std_logic_vector(CHANNEL_WIDTH_g/8-1 downto 0);
        patterndetect   :   std_logic_vector(CHANNEL_WIDTH_g/8-1 downto 0);
        errdetect       :   std_logic_vector(CHANNEL_WIDTH_g/8-1 downto 0);
        disperr         :   std_logic_vector(CHANNEL_WIDTH_g/8-1 downto 0);
    end record;
    type rx_vector_t is array (natural range <>) of rx_t;
    signal rx : rx_vector_t(NUMBER_OF_CHANNELS_g-1 downto 0);

begin

    reset_n <= not i_reset;

    gen_rx_data : for i in NUMBER_OF_CHANNELS_g-1 downto 0 generate
    begin
        o_rx_data(CHANNEL_WIDTH_g-1 + CHANNEL_WIDTH_g*i downto CHANNEL_WIDTH_g*i) <= rx(i).data;
        o_rx_datak(CHANNEL_WIDTH_g/8-1 + CHANNEL_WIDTH_g/8*i downto CHANNEL_WIDTH_g/8*i) <= rx(i).datak;
        rx(i).syncstatus <= rx_syncstatus(CHANNEL_WIDTH_g/8-1 + CHANNEL_WIDTH_g/8*i downto CHANNEL_WIDTH_g/8*i);
        rx(i).patterndetect <= rx_patterndetect(CHANNEL_WIDTH_g/8-1 + CHANNEL_WIDTH_g/8*i downto CHANNEL_WIDTH_g/8*i);
        rx(i).errdetect <= rx_errdetect(CHANNEL_WIDTH_g/8-1 + CHANNEL_WIDTH_g/8*i downto CHANNEL_WIDTH_g/8*i);
        rx(i).disperr <= rx_disperr(CHANNEL_WIDTH_g/8-1 + CHANNEL_WIDTH_g/8*i downto CHANNEL_WIDTH_g/8*i);
    end generate;

    g_rx_align : for i in NUMBER_OF_CHANNELS_g-1 downto 0 generate
    begin
        e_rx_rst_n : entity work.reset_sync
        port map ( rstout_n => rx(i).rst_n, arst_n => rx_ready(i), clk => i_rx_clkin(i) );

        e_rx_align : entity work.rx_align
        generic map (
            CHANNEL_WIDTH_g => CHANNEL_WIDTH_g,
            K_g => K_g--,
        )
        port map (
            o_data      => rx(i).data,
            o_datak     => rx(i).datak,

            o_locked    => rx(i).locked,

            i_data      => rx_data(CHANNEL_WIDTH_g-1 + CHANNEL_WIDTH_g*i downto CHANNEL_WIDTH_g*i),
            i_datak     => rx_datak(CHANNEL_WIDTH_g/8-1 + CHANNEL_WIDTH_g/8*i downto CHANNEL_WIDTH_g/8*i),

            i_syncstatus        => rx(i).syncstatus,
            i_patterndetect     => rx(i).patterndetect,
            o_enapatternalign   => rx_enapatternalign(i),

            i_errdetect => rx(i).errdetect,
            i_disperr   => rx(i).disperr,

            i_reset_n   => rx(i).rst_n,
            i_clk       => i_rx_clkin(i)--,
        );

        -- data counter
        e_rx_Gbit : entity work.counter
        generic map ( W => rx(i).Gbit'length, DIV => 2**30/32 )
        port map (
            cnt => rx(i).Gbit, ena => '1',
            reset => not rx(i).rst_n, clk => i_rx_clkin(i)
        );

        -- Loss-of-Lock (LoL) counter
        e_rx_LoL_cnt : entity work.counter
        generic map ( W => rx(i).LoL_cnt'length, EDGE => -1 ) -- falling edge
        port map (
            cnt => rx(i).LoL_cnt, ena => rx(i).locked,
            reset => not rx(i).rst_n, clk => i_rx_clkin(i)
        );

        -- 8b10b error counter
        e_rx_err_cnt : entity work.counter
        generic map ( W => rx(i).err_cnt'length )
        port map (
            cnt => rx(i).err_cnt,
            ena => work.util.to_std_logic( rx(i).errdetect /= 0 or rx(i).disperr /= 0 ),
            reset => not rx(i).rst_n, clk => i_rx_clkin(i)
        );
    end generate;

    -- av_ctrl process, avalon iface
    p_av_ctrl : process(i_clk, reset_n)
    begin
    if ( reset_n = '0' ) then
        av_ctrl.waitrequest <= '1';
        ch <= 0;
        rx_seriallpbken <= (others => '0');
        tx_rst_n <= (others => '1');
        rx_rst_n <= (others => '1');
        --
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
                if ( av_ctrl.write = '1' and av_ctrl.writedata(7 downto 0) < NUMBER_OF_CHANNELS_g ) then
                    ch <= to_integer(unsigned(av_ctrl.writedata(7 downto 0)));
                end if;
                --
            when X"01" =>
                av_ctrl.readdata(7 downto 0) <= std_logic_vector(to_unsigned(NUMBER_OF_CHANNELS_g, 8));
            when X"02" =>
                av_ctrl.readdata(7 downto 0) <= std_logic_vector(to_unsigned(CHANNEL_WIDTH_g, 8));
            when X"10" =>
                -- tx reset
                av_ctrl.readdata(0) <= tx_analogreset(ch);
                av_ctrl.readdata(4) <= tx_digitalreset(ch);
                if ( av_ctrl.write = '1' ) then tx_rst_n(ch) <= not av_ctrl.writedata(0); end if;
                --
            when X"11" =>
                -- tx status
                av_ctrl.readdata(0) <= tx_ready(ch);
                --
            when X"12" =>
                -- tx errors
                av_ctrl.readdata(8) <= tx_fifo_error(ch);
                --
            when X"20" =>
                -- rx reset
                av_ctrl.readdata(0) <= rx_analogreset(ch);
                av_ctrl.readdata(4) <= rx_digitalreset(ch);
                if ( av_ctrl.write = '1' ) then rx_rst_n(ch) <= not av_ctrl.writedata(0); end if;
                --
            when X"21" =>
                -- rx status
                av_ctrl.readdata(0) <= rx_ready(ch);
                av_ctrl.readdata(1) <= rx_is_lockedtoref(ch);
                av_ctrl.readdata(2) <= rx_is_lockedtodata(ch);
--                av_ctrl.readdata(11 downto 8) <= (others => '1');
                av_ctrl.readdata(CHANNEL_WIDTH_g/8-1 + 8 downto 8) <= rx(ch).syncstatus;
                av_ctrl.readdata(12) <= rx(ch).locked;
                --
            when X"22" =>
                -- rx errors
                av_ctrl.readdata(CHANNEL_WIDTH_g/8-1 + 0 downto 0) <= rx(ch).errdetect;
                av_ctrl.readdata(CHANNEL_WIDTH_g/8-1 + 4 downto 4) <= rx(ch).disperr;
                av_ctrl.readdata(8) <= rx_fifo_error(ch);
                --
            when X"23" =>
                av_ctrl.readdata(rx(ch).LoL_cnt'range) <= rx(ch).LoL_cnt;
            when X"24" =>
                av_ctrl.readdata(rx(ch).err_cnt'range) <= rx(ch).err_cnt;
                --
            when X"2A" =>
                av_ctrl.readdata(rx(ch).data'range) <= rx(ch).data;
            when X"2B" =>
                av_ctrl.readdata(rx(ch).datak'range) <= rx(ch).datak;
            when X"2C" =>
                av_ctrl.readdata(rx(ch).Gbit'range) <= rx(ch).Gbit;
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

        process(i_clk, reset_n)
        begin
        if ( reset_n = '0' ) then
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

    e_phy : entity work.altgx_generic
    generic map (
        NUMBER_OF_CHANNELS_g => NUMBER_OF_CHANNELS_g,
        CHANNEL_WIDTH_g => CHANNEL_WIDTH_g,
        INPUT_CLOCK_FREQUENCY_g => INPUT_CLOCK_FREQUENCY_g,
        DATA_RATE_g => DATA_RATE_g--,
    )
    port map (
        cal_blk_clk => i_clk,

        tx_dataout  => o_tx_serial,
        rx_datain   => i_rx_serial,

        pll_inclk       => i_pll_clk,
        pll_powerdown   => pll_powerdown,
        pll_locked      => pll_locked,

        -- analog reset => reset PMA/CDR (phys medium attachment, clock data recovery)
        -- digital reset => reset PCS (phys coding sublayer)
        tx_digitalreset => tx_digitalreset,
        rx_analogreset  => rx_analogreset,
        rx_digitalreset => rx_digitalreset,

        rx_pll_locked   => rx_is_lockedtoref,
        -- When asserted, indicates that the RX CDR is locked to incoming data. This signal is optional.
        rx_freqlocked   => rx_is_lockedtodata,

        tx_phase_comp_fifo_error => tx_fifo_error,
        rx_phase_comp_fifo_error => rx_fifo_error,
        -- When asserted, indicates that a received 10-bit code group has an 8B/10B code violation or disparity error.
        rx_errdetect => rx_errdetect,
        -- When asserted, indicates that the received 10-bit code or data group has a disparity error.
        rx_disperr => rx_disperr,

        rx_syncstatus => rx_syncstatus,
        rx_patterndetect => rx_patterndetect,
        rx_enapatternalign => rx_enapatternalign,

        tx_datain       => i_tx_data,
        tx_ctrlenable   => i_tx_datak,
        rx_dataout      => rx_data,
        rx_ctrldetect   => rx_datak,

        tx_clkout       => o_tx_clkout,
        tx_coreclk      => i_tx_clkin,
        rx_clkout       => o_rx_clkout,
        rx_coreclk      => i_rx_clkin,

        rx_seriallpbken => rx_seriallpbken,

        reconfig_togxb      => reconfig_togxb,
        reconfig_fromgxb    => reconfig_fromgxb,
        reconfig_clk        => reconfig_clk--,
    );

    g_reconfig_clk : if ( CLK_MHZ_g <= 50 ) generate
        reconfig_clk <= i_clk; -- Frequency Range (MHz) : 37.5 to 50
    end generate;

    -- generate reconfig_clk = 50 MHz
    g_reconfig_clk_altpll : if ( CLK_MHZ_g > 50 ) generate
        e_reconfig_clk : entity work.ip_altpll
        generic map (
            DIV => CLK_MHZ_g,
            MUL => 50--,
        )
        port map (
            c0 => reconfig_clk,
            locked => open,
            areset => i_reset,
            inclk0 => i_clk--,
        );
    end generate;

    e_reconfig_rst_n : entity work.reset_sync
    port map ( rstout_n => reconfig_rst_n, arst_n => reset_n, clk => reconfig_clk );

    e_reconfig : entity work.ip_altgx_reconfig
    port map (
        busy    => reconfig_busy,
        error   => reconfig_error,

        reconfig_togxb      => reconfig_togxb,
        reconfig_fromgxb    => reconfig_fromgxb,
        reconfig_reset      => not reconfig_rst_n,
        reconfig_clk        => reconfig_clk--,
    );

    --
    --
    --
    e_tx_reset : entity work.tx_reset
    generic map (
        NUMBER_OF_CHANNELS_g => NUMBER_OF_CHANNELS_g,
        NUMBER_OF_PLLS_g => 1,
        CLK_MHZ_g => CLK_MHZ_g--,
    )
    port map (
        o_analogreset => tx_analogreset,
        o_digitalreset => tx_digitalreset,

        o_ready => tx_ready,

        o_pll_powerdown => pll_powerdown,
        i_pll_locked => pll_locked,

        i_areset_n => reset_n and work.util.and_reduce(tx_rst_n),
        i_clk => i_clk--,
    );

    e_rx_reset : entity work.rx_reset
    generic map (
        NUMBER_OF_CHANNELS_g => NUMBER_OF_CHANNELS_g,
        CLK_MHZ_g => CLK_MHZ_g--,
    )
    port map (
        o_analogreset => rx_analogreset,
        o_digitalreset => rx_digitalreset,

        o_ready => rx_ready,

        i_freqlocked => rx_is_lockedtodata,

        i_reconfig_busy => reconfig_busy,

        i_areset_n => reset_n and work.util.and_reduce(rx_rst_n),
        i_clk => i_clk--,
    );

end architecture;
