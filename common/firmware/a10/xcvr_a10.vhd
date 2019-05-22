library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity xcvr_a10 is
    generic (
        -- number of channels
        Nch : positive := 4;
        K : std_logic_vector(7 downto 0) := work.util.D28_5--;
    );
    port (
        -- avalon slave interface
        avs_address     :   in  std_logic_vector(13 downto 0);
        avs_read        :   in  std_logic;
        avs_readdata    :   out std_logic_vector(31 downto 0);
        avs_write       :   in  std_logic;
        avs_writedata   :   in  std_logic_vector(31 downto 0);
        avs_waitrequest :   out std_logic;

        tx3_data    :   in  std_logic_vector(31 downto 0);
        tx3_datak   :   in  std_logic_vector(3 downto 0);
        rx3_data    :   out std_logic_vector(31 downto 0);
        rx3_datak   :   out std_logic_vector(3 downto 0);

        tx2_data    :   in  std_logic_vector(31 downto 0);
        tx2_datak   :   in  std_logic_vector(3 downto 0);
        rx2_data    :   out std_logic_vector(31 downto 0);
        rx2_datak   :   out std_logic_vector(3 downto 0);

        tx1_data    :   in  std_logic_vector(31 downto 0);
        tx1_datak   :   in  std_logic_vector(3 downto 0);
        rx1_data    :   out std_logic_vector(31 downto 0);
        rx1_datak   :   out std_logic_vector(3 downto 0);

        tx0_data    :   in  std_logic_vector(31 downto 0);
        tx0_datak   :   in  std_logic_vector(3 downto 0);
        rx0_data    :   out std_logic_vector(31 downto 0);
        rx0_datak   :   out std_logic_vector(3 downto 0);

        tx_clkout   :   out std_logic_vector(Nch-1 downto 0);
        tx_clkin    :   in  std_logic_vector(Nch-1 downto 0);
        rx_clkout   :   out std_logic_vector(Nch-1 downto 0);
        rx_clkin    :   in  std_logic_vector(Nch-1 downto 0);

        tx_p        :   out std_logic_vector(Nch-1 downto 0);
        rx_p        :   in  std_logic_vector(Nch-1 downto 0);

        pll_refclk  :   in  std_logic;
        cdr_refclk  :   in  std_logic;

        reset   :   in  std_logic;
        clk     :   in  std_logic--;
    );
end entity;

architecture arch of xcvr_a10 is

    signal rst_n : std_logic;

    signal ch : integer range Nch-1 downto 0;

    signal av_ctrl : work.mu3e.avalon_t;
    signal av_phy, av_pll : work.mu3e.avalon_t;

    signal rx_data_i            :   std_logic_vector(32*Nch-1 downto 0);
    signal rx_datak_i           :   std_logic_vector(4*Nch-1 downto 0);

    signal tx_rst_n, rx_rst_n   :   std_logic_vector(Nch-1 downto 0);

    signal pll_powerdown        :   std_logic_vector(0 downto 0);
    signal pll_cal_busy         :   std_logic_vector(0 downto 0);
    signal pll_locked           :   std_logic_vector(0 downto 0);

    signal tx_serial_clk        :   std_logic;

    signal tx_analogreset       :   std_logic_vector(Nch-1 downto 0);
    signal tx_digitalreset      :   std_logic_vector(Nch-1 downto 0);
    signal rx_analogreset       :   std_logic_vector(Nch-1 downto 0);
    signal rx_digitalreset      :   std_logic_vector(Nch-1 downto 0);

    signal tx_cal_busy          :   std_logic_vector(Nch-1 downto 0);
    signal rx_cal_busy          :   std_logic_vector(Nch-1 downto 0);

    signal tx_ready             :   std_logic_vector(Nch-1 downto 0);
    signal rx_ready             :   std_logic_vector(Nch-1 downto 0);
    signal rx_is_lockedtoref    :   std_logic_vector(Nch-1 downto 0);
    signal rx_is_lockedtodata   :   std_logic_vector(Nch-1 downto 0);

    signal tx_fifo_error        :   std_logic_vector(Nch-1 downto 0);
    signal rx_fifo_error        :   std_logic_vector(Nch-1 downto 0);
    signal rx_errdetect         :   std_logic_vector(4*Nch-1 downto 0);
    signal rx_disperr           :   std_logic_vector(4*Nch-1 downto 0);

    signal rx_syncstatus        :   std_logic_vector(4*Nch-1 downto 0);
    signal rx_patterndetect     :   std_logic_vector(4*Nch-1 downto 0);
    signal rx_enapatternalign   :   std_logic_vector(Nch-1 downto 0);

    signal rx_seriallpbken      :   std_logic_vector(Nch-1 downto 0);



    type rx_t is record
        data    :   std_logic_vector(31 downto 0);
        datak   :   std_logic_vector(3 downto 0);
        lock    :   std_logic;
        rst_n   :   std_logic;

        -- loss-of-lock counter
        LoL_cnt :   std_logic_vector(7 downto 0);
        -- error counter
        err_cnt :   std_logic_vector(15 downto 0);
    end record;
    type rx_vector_t is array (natural range <>) of rx_t;
    signal rx : rx_vector_t(Nch-1 downto 0);

begin

    rst_n <= not reset;

    rx3_data <= rx(3).data;
    rx3_datak <= rx(3).datak;
    rx2_data <= rx(2).data;
    rx2_datak <= rx(2).datak;
    rx1_data <= rx(1).data;
    rx1_datak <= rx(1).datak;
    rx0_data <= rx(0).data;
    rx0_datak <= rx(0).datak;

    g_rx_align : for i in rx_clkin'range generate
    begin
        i_rx_rst_n : entity work.reset_sync
        port map ( rstout_n => rx(i).rst_n, arst_n => rx_ready(i), clk => rx_clkin(i) );

        i_rx_align : entity work.rx_align
        generic map ( K => K )
        port map (
            data    => rx(i).data,
            datak   => rx(i).datak,

            lock    => rx(i).lock,

            datain  => rx_data_i(31 + 32*i downto 32*i),
            datakin => rx_datak_i(3 + 4*i downto 4*i),

            syncstatus      => rx_syncstatus(3 + 4*i downto 4*i),
            patterndetect   => rx_patterndetect(3 + 4*i downto 4*i),
            enapatternalign => rx_enapatternalign(i),

            errdetect => rx_errdetect(3 + 4*i downto 4*i),
            disperr => rx_disperr(3 + 4*i downto 4*i),

            rst_n   => rx(i).rst_n,
            clk     => rx_clkin(i)--,
        );

        i_rx_LoL_cnt : entity work.counter
        generic map ( W => rx(i).LoL_cnt'length, EDGE => -1 )
        port map ( cnt => rx(i).LoL_cnt, ena => rx(i).lock, reset => not rx(i).rst_n, clk => rx_clkin(i) );

        i_rx_err_cnt : entity work.counter
        generic map ( W => rx(i).err_cnt'length )
        port map (
            cnt => rx(i).err_cnt,
            ena => work.util.to_std_logic( rx_errdetect(3 + 4*i downto 4*i) /= 0 or rx_disperr(3 + 4*i downto 4*i) /= 0 ),
            reset => not rx(i).rst_n, clk => rx_clkin(i)
        );
    end generate;

    -- av_ctrl process, avalon iface
    p_av_ctrl : process(clk, rst_n)
    begin
    if ( rst_n = '0' ) then
        av_ctrl.waitrequest <= '1';
        ch <= 0;
        rx_seriallpbken <= (others => '0');
        tx_rst_n <= (others => '1');
        rx_rst_n <= (others => '1');
        --
    elsif rising_edge(clk) then
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
                if ( av_ctrl.write = '1' and av_ctrl.writedata(7 downto 0) < Nch ) then
                    ch <= to_integer(unsigned(av_ctrl.writedata(7 downto 0)));
                end if;
                --
            when X"01" =>
                av_ctrl.readdata(7 downto 0) <= std_logic_vector(to_unsigned(Nch, 8));
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
                av_ctrl.readdata(11 downto 8) <= rx_syncstatus(3 + 4*ch downto 4*ch);
                av_ctrl.readdata(12) <= rx(ch).lock;
                --
            when X"22" =>
                -- rx errors
                av_ctrl.readdata(3 downto 0) <= rx_errdetect(3 + 4*ch downto 4*ch);
                av_ctrl.readdata(7 downto 4) <= rx_disperr(3 + 4*ch downto 4*ch);
                av_ctrl.readdata(8) <= rx_fifo_error(ch);
                --
            when X"23" =>
                av_ctrl.readdata(rx(ch).LoL_cnt'range) <= rx(ch).LoL_cnt;
            when X"24" =>
                av_ctrl.readdata(rx(ch).err_cnt'range) <= rx(ch).err_cnt;
                --
            when X"2A" =>
                av_ctrl.readdata <= rx(ch).data;
            when X"2B" =>
                av_ctrl.readdata(3 downto 0) <= rx(ch).datak;
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

    b_avs : block
        signal av_ctrl_cs : std_logic;
        signal av_phy_cs, av_pll_cs : std_logic;
        signal avs_waitrequest_i : std_logic;
    begin
        av_ctrl_cs <= '1' when ( avs_address(avs_address'left downto 8) = "000000" ) else '0';
        av_ctrl.address(avs_address'range) <= avs_address;
        av_ctrl.writedata <= avs_writedata;

        av_phy_cs <= '1' when ( avs_address(avs_address'left downto 10) = "0100" ) else '0';
        av_phy.address(avs_address'range) <= avs_address;
        av_phy.writedata <= avs_writedata;

        av_pll_cs <= '1' when ( avs_address(avs_address'left downto 10) = "1000" ) else '0';
        av_pll.address(avs_address'range) <= avs_address;
        av_pll.writedata <= avs_writedata;

        avs_waitrequest <= avs_waitrequest_i;

        process(clk, rst_n)
        begin
        if ( rst_n = '0' ) then
            avs_waitrequest_i <= '1';
            av_ctrl.read <= '0';
            av_ctrl.write <= '0';
            av_phy.read <= '0';
            av_phy.write <= '0';
            av_pll.read <= '0';
            av_pll.write <= '0';
            --
        elsif rising_edge(clk) then
            avs_waitrequest_i <= '1';

            if ( avs_read /= avs_write and avs_waitrequest_i = '1' ) then
                if ( av_ctrl_cs = '1' ) then
                    if ( av_ctrl.read = av_ctrl.write ) then
                        av_ctrl.read <= avs_read;
                        av_ctrl.write <= avs_write;
                    elsif ( av_ctrl.waitrequest = '0' ) then
                        avs_readdata <= av_ctrl.readdata;
                        avs_waitrequest_i <= '0';
                        av_ctrl.read <= '0';
                        av_ctrl.write <= '0';
                    end if;
                elsif ( av_phy_cs = '1' ) then
                    if ( av_phy.read = av_phy.write ) then
                        av_phy.read <= avs_read;
                        av_phy.write <= avs_write;
                    elsif ( av_phy.waitrequest = '0' ) then
                        avs_readdata <= av_phy.readdata;
                        avs_waitrequest_i <= '0';
                        av_phy.read <= '0';
                        av_phy.write <= '0';
                    end if;
                elsif ( av_pll_cs = '1' ) then
                    if ( av_pll.read = av_pll.write ) then
                        av_pll.read <= avs_read;
                        av_pll.write <= avs_write;
                    elsif ( av_pll.waitrequest = '0' ) then
                        avs_readdata <= av_pll.readdata;
                        avs_waitrequest_i <= '0';
                        av_pll.read <= '0';
                        av_pll.write <= '0';
                    end if;
                else
                    avs_readdata <= X"CCCCCCCC";
                    avs_waitrequest_i <= '0';
                end if;
            end if;
            --
        end if;
        end process;
    end block;

    i_phy : component work.cmp.ip_xcvr_phy
    port map (
        tx_serial_data  => tx_p,
        rx_serial_data  => rx_p,

        rx_cdr_refclk0  => cdr_refclk,
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

        tx_parallel_data(127 downto 96) => tx3_data,
        tx_datak(15 downto 12)          => tx3_datak,
        tx_parallel_data(95 downto 64)  => tx2_data,
        tx_datak(11 downto 8)           => tx2_datak,
        tx_parallel_data(63 downto 32)  => tx1_data,
        tx_datak(7 downto 4)            => tx1_datak,
        tx_parallel_data(31 downto 0)   => tx0_data,
        tx_datak(3 downto 0)            => tx0_datak,
        rx_parallel_data => rx_data_i,
        rx_datak => rx_datak_i,

        tx_clkout       => tx_clkout,
        tx_coreclkin    => tx_clkin,
        rx_clkout       => rx_clkout,
        rx_coreclkin    => rx_clkin,

        rx_seriallpbken => rx_seriallpbken,

        unused_tx_parallel_data => (others => '0'),
        unused_rx_parallel_data => open,

        reconfig_address        => std_logic_vector(to_unsigned(ch, 2)) & av_phy.address(9 downto 0),
        reconfig_read(0)        => av_phy.read,
        reconfig_readdata       => av_phy.readdata,
        reconfig_write(0)       => av_phy.write,
        reconfig_writedata      => av_phy.writedata,
        reconfig_waitrequest(0) => av_phy.waitrequest,
        reconfig_reset(0)       => reset,
        reconfig_clk(0)         => clk--,
    );

    i_fpll : component work.cmp.ip_xcvr_fpll
    port map (
        pll_refclk0     => pll_refclk,
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
        reconfig_clk0           => clk--,
    );

    --
    --
    --
    i_reset : component work.cmp.ip_xcvr_reset
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
        clock => clk--,
    );

end architecture;
