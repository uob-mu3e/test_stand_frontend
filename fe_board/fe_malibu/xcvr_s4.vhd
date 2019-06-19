library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity xcvr_s4 is
    generic (
        -- number of channels
        Nch : positive := 4;
        data_rate : positive := 5000; -- Gbps
        pll_freq : real := 125.0; -- MHz
        K : std_logic_vector(7 downto 0) := work.util.D28_5;
        CLK_MHZ : positive := 50--;
    );
    port (
        -- avalon slave interface
        avs_address     :   in  std_logic_vector(13 downto 0);
        avs_read        :   in  std_logic;
        avs_readdata    :   out std_logic_vector(31 downto 0);
        avs_write       :   in  std_logic;
        avs_writedata   :   in  std_logic_vector(31 downto 0);
        avs_waitrequest :   out std_logic;

        tx_data     :   in  std_logic_vector(32*Nch-1 downto 0);
        tx_datak    :   in  std_logic_vector(4*Nch-1 downto 0);
        rx_data     :   out std_logic_vector(32*Nch-1 downto 0);
        rx_datak    :   out std_logic_vector(4*Nch-1 downto 0);

        tx_clkout   :   out std_logic_vector(Nch-1 downto 0);
        tx_clkin    :   in  std_logic_vector(Nch-1 downto 0);
        rx_clkout   :   out std_logic_vector(Nch-1 downto 0);
        rx_clkin    :   in  std_logic_vector(Nch-1 downto 0);
        -- TODO: tx/rx_rstout_n

        tx_p        :   out std_logic_vector(Nch-1 downto 0);
        rx_p        :   in  std_logic_vector(Nch-1 downto 0);

        pll_refclk  :   in  std_logic;
        cdr_refclk  :   in  std_logic;

        reset   :   in  std_logic;
        clk     :   in  std_logic--;
    );
end entity;

architecture arch of xcvr_s4 is

    signal rst_n : std_logic;

    signal ch : integer range Nch-1 downto 0;

    signal av_ctrl : work.mu3e.avalon_t;

    signal rx_data_i            :   std_logic_vector(32*Nch-1 downto 0);
    signal rx_datak_i           :   std_logic_vector(4*Nch-1 downto 0);

    signal tx_rst_n, rx_rst_n   :   std_logic_vector(Nch-1 downto 0);

    signal pll_powerdown        :   std_logic_vector(0 downto 0);
    signal pll_cal_busy         :   std_logic_vector(0 downto 0);
    signal pll_locked           :   std_logic_vector(0 downto 0);

    signal tx_analogreset       :   std_logic_vector(Nch-1 downto 0);
    signal tx_digitalreset      :   std_logic_vector(Nch-1 downto 0);
    signal rx_analogreset       :   std_logic_vector(Nch-1 downto 0);
    signal rx_digitalreset      :   std_logic_vector(Nch-1 downto 0);

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

    signal reconfig_togxb       :   std_logic_vector(3 downto 0);
    signal reconfig_fromgxb     :   std_logic_vector(16 downto 0);
    signal reconfig_busy        :   std_logic;
    signal reconfig_error       :   std_logic;
    signal reconfig_rst_n       :   std_logic;
    signal reconfig_clk         :   std_logic;



    type rx_t is record
        data    :   std_logic_vector(31 downto 0);
        datak   :   std_logic_vector(3 downto 0);
        lock    :   std_logic;
        rst_n   :   std_logic;

        -- Gbit counter
        Gbit    :   std_logic_vector(23 downto 0);
        -- loss-of-lock counter
        LoL_cnt :   std_logic_vector(7 downto 0);
        -- error counter
        err_cnt :   std_logic_vector(15 downto 0);
    end record;
    type rx_vector_t is array (natural range <>) of rx_t;
    signal rx : rx_vector_t(Nch-1 downto 0);

begin

    rst_n <= not reset;

    gen_rx_data : for i in 0 to Nch-1 generate
    begin
        rx_data(31 + 32*i downto 32*i) <= rx(i).data;
        rx_datak(3 + 4*i downto 4*i) <= rx(i).datak;
    end generate;

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

        i_rx_Gbit : entity work.counter
        generic map ( W => rx(i).Gbit'length, DIV => 2**30/32 )
        port map ( cnt => rx(i).Gbit, ena => '1', reset => not rx(i).rst_n, clk => rx_clkin(i) );

        i_rx_LoL_cnt : entity work.counter
        generic map ( W => rx(i).LoL_cnt'length, EDGE => -1 )
        port map ( cnt => rx(i).LoL_cnt, ena => rx(i).lock, reset => not rx(i).rst_n, clk => rx_clkin(i) );

        i_rx_err_cnt : entity work.counter
        generic map ( W => rx(i).err_cnt'length )
        port map (
            cnt => rx(i).err_cnt,
            ena => work.util.to_std_logic( rx_errdetect(3 + 4*i downto 4*i) /= "0000" or rx_disperr(3 + 4*i downto 4*i) /= "0000" ),
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

    b_avs : block
        signal av_ctrl_cs : std_logic;
        signal avs_waitrequest_i : std_logic;
    begin
        av_ctrl_cs <= '1' when ( avs_address(avs_address'left downto 8) = "000000" ) else '0';
        av_ctrl.address(avs_address'range) <= avs_address;
        av_ctrl.writedata <= avs_writedata;

        avs_waitrequest <= avs_waitrequest_i;

        process(clk, rst_n)
        begin
        if ( rst_n = '0' ) then
            avs_waitrequest_i <= '1';
            av_ctrl.read <= '0';
            av_ctrl.write <= '0';
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
                else
                    avs_readdata <= X"CCCCCCCC";
                    avs_waitrequest_i <= '0';
                end if;
            end if;
            --
        end if;
        end process;
    end block;

    i_phy : component work.cmp.ip_altgx
    generic map (
        effective_data_rate => data_rate,
        input_clock_frequency => real'image(pll_freq),
        input_clock_period => integer(1000000.0 / pll_freq),
        m_divider => integer(real(data_rate) / pll_freq / 2.0)--,
    )
    port map (
        cal_blk_clk => clk,

        tx_dataout  => tx_p,
        rx_datain   => rx_p,

        pll_inclk       => pll_refclk,
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

        tx_datain       => tx_data,
        tx_ctrlenable   => tx_datak,
        rx_dataout      => rx_data_i,
        rx_ctrldetect   => rx_datak_i,

        tx_clkout       => tx_clkout,
        tx_coreclk      => tx_clkin,
        rx_clkout       => rx_clkout,
        rx_coreclk      => rx_clkin,

        rx_seriallpbken => rx_seriallpbken,

        reconfig_togxb      => reconfig_togxb,
        reconfig_fromgxb    => reconfig_fromgxb,
        reconfig_clk        => reconfig_clk--,
    );

    reconfig_clk <= clk; -- Frequency Range (MHz) : 37.5 to 50

    i_reconfig_rst_n : entity work.reset_sync
    port map ( rstout_n => reconfig_rst_n, arst_n => rst_n, clk => reconfig_clk );

    i_reconfig : component work.cmp.ip_altgx_reconfig
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
    i_tx_reset : entity work.tx_reset
    generic map (
        Nch => 4,
        Npll => 1,
        CLK_MHZ => CLK_MHZ--,
    )
    port map (
        analogreset => tx_analogreset,
        digitalreset => tx_digitalreset,

        ready => tx_ready,

        pll_powerdown => pll_powerdown,
        pll_locked => pll_locked,

        arst_n => rst_n and work.util.and_reduce(tx_rst_n),
        clk => clk--,
    );

    i_rx_reset : entity work.rx_reset
    generic map (
        Nch => 4,
        CLK_MHZ => CLK_MHZ--,
    )
    port map (
        analogreset => rx_analogreset,
        digitalreset => rx_digitalreset,

        ready => rx_ready,

        freqlocked => rx_is_lockedtodata,

        reconfig_busy => reconfig_busy,

        arst_n => rst_n and work.util.and_reduce(rx_rst_n),
        clk => clk--,
    );

end architecture;
