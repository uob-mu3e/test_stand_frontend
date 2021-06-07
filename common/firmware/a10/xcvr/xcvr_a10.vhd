--
-- author : Alexandr Kozlinskiy
-- date : 2017-02-24
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity xcvr_a10 is
generic (
    NUMBER_OF_CHANNELS_g : positive := 4;
    CHANNEL_WIDTH_g : positive := 32;
    INPUT_CLOCK_FREQUENCY_g : positive := 125000000;
    DATA_RATE_g : positive := 5000;
    K_g : std_logic_vector(7 downto 0) := work.util.D28_5;
    CLK_MHZ_g : positive := 50--;
);
port (
    i_rx_serial         : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    o_tx_serial         : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    i_pll_clk           : in    std_logic;
    i_cdr_clk           : in    std_logic;

    o_rx_data           : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    o_rx_datak          : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    i_tx_data           : in    std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    i_tx_datak          : in    std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);

    o_rx_clkout         : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    i_rx_clkin          : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    o_tx_clkout         : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    i_tx_clkin          : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    -- avalon slave interface
    -- # address units words
    -- # read latency 0
    i_avs_address       : in    std_logic_vector(13 downto 0) := (others => '0');
    i_avs_read          : in    std_logic := '0';
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic := '0';
    i_avs_writedata     : in    std_logic_vector(31 downto 0) := (others => '0');
    o_avs_waitrequest   : out   std_logic;

    -- TODO: rename to i_reset_n
    i_reset             : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of xcvr_a10 is

    signal reset, reset_n : std_logic;

    signal ch : integer range 0 to NUMBER_OF_CHANNELS_g-1 := 0;

    signal av_ctrl : work.util.avalon_t;
    signal av_phy, av_pll : work.util.avalon_t;

    signal rx_data              :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    signal rx_datak             :   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);

    signal tx_rst_n, rx_rst_n   :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    signal pll_powerdown        :   std_logic_vector(0 downto 0);
    signal pll_cal_busy         :   std_logic_vector(0 downto 0);
    signal pll_locked           :   std_logic_vector(0 downto 0);

    signal tx_serial_clk        :   std_logic;

    signal tx_analogreset       :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal tx_digitalreset      :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_analogreset       :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_digitalreset      :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

    signal tx_cal_busy          :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    signal rx_cal_busy          :   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);

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

    reset <= i_reset;
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
        port map ( o_reset_n => rx(i).rst_n, i_reset_n => reset_n and rx_rst_n(i), i_clk => i_rx_clkin(i) );

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
        generic map ( DIV => 2**30/32, W => rx(i).Gbit'length )
        port map (
            o_cnt => rx(i).Gbit, i_ena => '1',
            i_reset_n => rx(i).rst_n, i_clk => i_rx_clkin(i)
        );

        -- Loss-of-Lock (LoL) counter
        e_rx_LoL_cnt : entity work.counter
        generic map ( EDGE => -1, W => rx(i).LoL_cnt'length ) -- falling edge
        port map (
            o_cnt => rx(i).LoL_cnt, i_ena => rx(i).locked,
            i_reset_n => rx(i).rst_n, i_clk => i_rx_clkin(i)
        );

        -- 8b10b error counter
        e_rx_err_cnt : entity work.counter
        generic map ( W => rx(i).err_cnt'length )
        port map (
            o_cnt => rx(i).err_cnt,
            i_ena => rx(i).locked and work.util.to_std_logic( rx(i).errdetect /= 0 or rx(i).disperr /= 0 ),
            i_reset_n => rx(i).rst_n, i_clk => i_rx_clkin(i)
        );
    end generate;

    -- av_ctrl process, avalon iface
    process(i_clk, reset_n)
    begin
    if ( reset_n = '0' ) then
        av_ctrl.waitrequest <= '1';
        ch <= 0;
        rx_seriallpbken <= (others => '0');
        tx_rst_n <= (others => '0');
        rx_rst_n <= (others => '0');
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
        signal av_phy_cs, av_pll_cs : std_logic;
        signal avs_waitrequest : std_logic;
    begin
        av_ctrl_cs <= '1' when ( i_avs_address(i_avs_address'left downto 8) = "000000" ) else '0';
        av_ctrl.address(i_avs_address'range) <= i_avs_address;
        av_ctrl.writedata <= i_avs_writedata;

        -- (alt_u32*)BASE + 0x1000
        av_phy_cs <= '1' when ( i_avs_address(i_avs_address'left downto 10) = "0100" ) else '0';
        av_phy.address(i_avs_address'range) <= i_avs_address;
        av_phy.writedata <= i_avs_writedata;

        -- (alt_u32*)BASE + 0x2000
        av_pll_cs <= '1' when ( i_avs_address(i_avs_address'left downto 10) = "1000" ) else '0';
        av_pll.address(i_avs_address'range) <= i_avs_address;
        av_pll.writedata <= i_avs_writedata;

        o_avs_waitrequest <= avs_waitrequest;

        process(i_clk, reset_n)
        begin
        if ( reset_n = '0' ) then
            avs_waitrequest <= '1';
            av_ctrl.read <= '0';
            av_ctrl.write <= '0';
            av_phy.read <= '0';
            av_phy.write <= '0';
            av_pll.read <= '0';
            av_pll.write <= '0';
            --
        elsif rising_edge(i_clk) then
            avs_waitrequest <= '1';

            if ( i_avs_read /= i_avs_write and avs_waitrequest = '1' ) then
                if ( av_ctrl_cs = '1' ) then
                    if ( av_ctrl.read = av_ctrl.write ) then
                        av_ctrl.read <= i_avs_read;
                        av_ctrl.write <= i_avs_write;
                    elsif ( av_ctrl.waitrequest = '0' ) then
                        o_avs_readdata <= av_ctrl.readdata;
                        avs_waitrequest <= '0';
                        av_ctrl.read <= '0';
                        av_ctrl.write <= '0';
                    end if;
                elsif ( av_phy_cs = '1' ) then
                    if ( av_phy.read = av_phy.write ) then
                        av_phy.read <= i_avs_read;
                        av_phy.write <= i_avs_write;
                    elsif ( av_phy.waitrequest = '0' ) then
                        o_avs_readdata <= av_phy.readdata;
                        avs_waitrequest <= '0';
                        av_phy.read <= '0';
                        av_phy.write <= '0';
                    end if;
                elsif ( av_pll_cs = '1' ) then
                    if ( av_pll.read = av_pll.write ) then
                        av_pll.read <= i_avs_read;
                        av_pll.write <= i_avs_write;
                    elsif ( av_pll.waitrequest = '0' ) then
                        o_avs_readdata <= av_pll.readdata;
                        avs_waitrequest <= '0';
                        av_pll.read <= '0';
                        av_pll.write <= '0';
                    end if;
                else
                    o_avs_readdata <= X"CCCCCCCC";
                    avs_waitrequest <= '0';
                end if;
            end if;
            --
        end if;
        end process;
    end block;

    e_phy : component work.cmp.ip_xcvr_a10_phy
    port map (
        tx_serial_data  => o_tx_serial,
        rx_serial_data  => i_rx_serial,

        rx_cdr_refclk0  => i_cdr_clk,
        tx_serial_clk0  => (others => tx_serial_clk),

        -- analog reset => reset PMA/CDR (phys medium attachment, clock data recovery)
        -- digital reset => reset PCS (phys coding sublayer)
        tx_analogreset  => tx_analogreset,
        tx_digitalreset => tx_digitalreset,
        rx_analogreset  => rx_analogreset,
        rx_digitalreset => rx_digitalreset,

        tx_cal_busy     => tx_cal_busy,
        rx_cal_busy     => rx_cal_busy,

        rx_is_lockedtoref => rx_is_lockedtoref,
        -- When asserted, indicates that the RX CDR is locked to incoming data. This signal is optional.
        rx_is_lockedtodata => rx_is_lockedtodata,

        -- When asserted, indicates that a received 10-bit code group has an 8B/10B code violation or disparity error.
        rx_errdetect => rx_errdetect,
        -- When asserted, indicates that the received 10-bit code or data group has a disparity error.
        rx_disperr => rx_disperr,
        rx_runningdisp => open,

        rx_syncstatus => rx_syncstatus,
        rx_patterndetect => rx_patterndetect,

        tx_parallel_data    => i_tx_data,
        tx_datak            => i_tx_datak,
        rx_parallel_data    => rx_data,
        rx_datak            => rx_datak,

        tx_clkout       => o_tx_clkout,
        tx_coreclkin    => i_tx_clkin,
        rx_clkout       => o_rx_clkout,
        rx_coreclkin    => i_rx_clkin,

        rx_seriallpbken => rx_seriallpbken,

        unused_tx_parallel_data => (others => '0'),
        unused_rx_parallel_data => open,

        reconfig_address        => std_logic_vector(to_unsigned(ch, work.util.vector_width(NUMBER_OF_CHANNELS_g))) & av_phy.address(9 downto 0),
        reconfig_read(0)        => av_phy.read,
        reconfig_readdata       => av_phy.readdata,
        reconfig_write(0)       => av_phy.write,
        reconfig_writedata      => av_phy.writedata,
        reconfig_waitrequest(0) => av_phy.waitrequest,
        reconfig_reset(0)       => reset,
        reconfig_clk(0)         => i_clk--,
    );

    e_fpll : component work.cmp.ip_xcvr_a10_fpll
    port map (
        pll_refclk0     => i_pll_clk,
        pll_powerdown   => pll_powerdown(0),
        pll_cal_busy    => pll_cal_busy(0),
        pll_locked      => pll_locked(0),
        tx_serial_clk   => tx_serial_clk,

        reconfig_address0       => av_pll.address(9 downto 0),
        reconfig_read0          => av_pll.read,
        reconfig_readdata0      => av_pll.readdata,
        reconfig_write0         => av_pll.write,
        reconfig_writedata0     => av_pll.writedata,
        reconfig_waitrequest0   => av_pll.waitrequest,
        reconfig_reset0         => reset,
        reconfig_clk0           => i_clk--,
    );

    --
    --
    --
    e_reset : component work.cmp.ip_xcvr_a10_reset
    port map (
        tx_analogreset => tx_analogreset,
        tx_digitalreset => tx_digitalreset,
        rx_analogreset => rx_analogreset,
        rx_digitalreset => rx_digitalreset,

        tx_cal_busy => tx_cal_busy,
        rx_cal_busy => rx_cal_busy,

        tx_ready => tx_ready,
        rx_ready => rx_ready,

        rx_is_lockedtodata => rx_is_lockedtodata,

        pll_powerdown => pll_powerdown,
        pll_cal_busy => pll_cal_busy,
        pll_locked => pll_locked,

        pll_select => (others => '0'),

        reset => reset,
        clock => i_clk--,
    );

end architecture;
