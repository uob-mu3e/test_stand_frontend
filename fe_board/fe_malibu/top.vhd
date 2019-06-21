library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    -- FE.A
    malibu_ck_fpga_0    : out   std_logic; -- pin 36, 38 -- malibu.CK_FPGA_0_N/P
    malibu_pll_reset    : out   std_logic; -- pin 42, 44 -- malibu.PLL_reset_P/N
    malibu_spi_sck      : out   std_logic; -- pin 54 -- malibu.SPI_SCK_P
    malibu_spi_sdi      : inout std_logic; -- pin 50 -- malibu.SPI_SDI_P
    malibu_spi_sdo      : inout std_logic; -- pin 52 -- malibu.SPI_SDO_N
    malibu_chip_reset   : out   std_logic; -- pin 48 -- malibu.chip_reset

    -- FE.B
    malibu_ck_fpga_1    : out   std_logic; -- pin 36, 38 -- malibu.CK_FPGA_1_P/N
    malibu_pll_test     : out   std_logic; -- pin 42, 44 -- malibu.PLL_TEST_N/P
    malibu_i2c_scl      : out   std_logic; -- pin 54 -- malibu.i2c_SCL
    malibu_i2c_sda      : inout std_logic; -- pin 56 -- malibu.i2c_SDA
    malibu_i2c_int_n    : inout std_logic; -- pin 52 -- malibu.I2C_INTn
    malibu_spi_sdo_cec  : in    std_logic; -- pin 48 -- malibu.SPI_SDO_CEC

    malibu_data         : in    std_logic_vector(13 downto 0);



    -- SI45

    si45_oe_n       : out   std_logic; -- <= '0'
    si45_rst_n      : out   std_logic; -- reset
    si45_spi_out    : in    std_logic; -- slave data out
    si45_spi_in     : out   std_logic; -- slave data in
    si45_spi_sclk   : out   std_logic; -- clock
    si45_spi_cs_n   : out   std_logic; -- chip select



    -- QSFP

    qsfp_pll_clk    : in    std_logic; -- 156.25 MHz

    QSFP_ModSel_n   : out   std_logic; -- module select (i2c)
    QSFP_Rst_n      : out   std_logic;
    QSFP_LPM        : out   std_logic; -- Low Power Mode

    qsfp_tx         : out   std_logic_vector(3 downto 0);
    qsfp_rx         : in    std_logic_vector(3 downto 0);



    -- POD

--    pod_pll_clk     : in    std_logic;
--
--    pod_tx_reset    : out   std_logic;
--    pod_rx_reset    : out   std_logic;
--
--    pod_tx          : out   std_logic_vector(3 downto 0);
--    pod_rx          : in    std_logic_vector(3 downto 0);



    --

    led_n       : out   std_logic_vector(15 downto 0);

    reset_n     : in    std_logic;
    -- 125 MHz
    clk_aux     : in    std_logic--;
);
end entity;

architecture arch of top is

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal led : std_logic_vector(led_n'range);

    signal nios_clk, nios_reset_n : std_logic;
    signal nios_pio : std_logic_vector(31 downto 0);

    signal i2c_scl_in, i2c_scl_oe, i2c_sda_in, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(1 downto 0);

    signal malibu_clk : std_logic;
    signal malibu_rx_data_clk : std_logic;
    signal malibu_rx_data : std_logic_vector(15 downto 0);
    signal malibu_rx_datak : std_logic_vector(1 downto 0);
    signal malibu_word : std_logic_vector(47 downto 0);

    signal avm_pod, avm_qsfp : work.mu3e.avalon_t;

    signal qsfp_tx_data : std_logic_vector(127 downto 0);
    signal qsfp_tx_datak : std_logic_vector(15 downto 0);

    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);

    signal qsfp_reset_n : std_logic;

    signal avm_sc : work.mu3e.avalon_t;

    signal ram_addr_a : std_logic_vector(15 downto 0);
    signal ram_rdata_a : std_logic_vector(31 downto 0);
    signal ram_wdata_a : std_logic_vector(31 downto 0);
    signal ram_we_a : std_logic;

    signal data_to_fifo : std_logic_vector(35 downto 0);
    signal data_to_fifo_we : std_logic;
    signal data_from_fifo : std_logic_vector(35 downto 0);
    signal data_from_fifo_re : std_logic;
    signal data_from_fifo_empty : std_logic;

    signal sc_to_fifo : std_logic_vector(35 downto 0);
    signal sc_to_fifo_we : std_logic;
    signal sc_from_fifo : std_logic_vector(35 downto 0);
    signal sc_from_fifo_re : std_logic;
    signal sc_from_fifo_empty : std_logic;

begin

    led_n <= not led;

    -- 125 MHz
    i_clk_aux_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(15), rst_n => reset_n, clk => clk_aux );

    -- 156.25 MHz
    i_clk_qsfp_hz : entity work.clkdiv
    generic map ( P => 156250000 )
    port map ( clkout => led(14), rst_n => reset_n, clk => qsfp_pll_clk );

    ----------------------------------------------------------------------------
    -- NIOS

    -- 50 MHz
    i_nios_clk : entity work.ip_altpll
    generic map (
        DIV => 5,
        MUL => 2--,
    )
    port map (
        c0 => nios_clk,
        locked => open,
        areset => '0',
        inclk0 => clk_aux--,
    );

    i_nios_reset_n : entity work.reset_sync
    port map ( rstout_n => nios_reset_n, arst_n => reset_n, clk => nios_clk );

    i_qsfp_reset_n : entity work.reset_sync
    port map ( rstout_n => qsfp_reset_n, arst_n => reset_n, clk => qsfp_pll_clk );

    led(12) <= nios_pio(7);

    i_nios : component work.cmp.nios
    port map (
        avm_qsfp_address        => avm_qsfp.address(15 downto 0),
        avm_qsfp_read           => avm_qsfp.read,
        avm_qsfp_readdata       => avm_qsfp.readdata,
        avm_qsfp_write          => avm_qsfp.write,
        avm_qsfp_writedata      => avm_qsfp.writedata,
        avm_qsfp_waitrequest    => avm_qsfp.waitrequest,

        avm_pod_address         => avm_pod.address(15 downto 0),
        avm_pod_read            => avm_pod.read,
        avm_pod_readdata        => avm_pod.readdata,
        avm_pod_write           => avm_pod.write,
        avm_pod_writedata       => avm_pod.writedata,
        avm_pod_waitrequest     => avm_pod.waitrequest,

        avm_sc_address          => avm_sc.address(15 downto 0),
        avm_sc_read             => avm_sc.read,
        avm_sc_readdata         => avm_sc.readdata,
        avm_sc_write            => avm_sc.write,
        avm_sc_writedata        => avm_sc.writedata,
        avm_sc_waitrequest      => avm_sc.waitrequest,

        sc_clk_clk          => qsfp_pll_clk,
        sc_reset_reset_n    => qsfp_reset_n,

        --
        -- nios base
        --

        i2c_scl_in => i2c_scl_in,
        i2c_scl_oe => i2c_scl_oe,
        i2c_sda_in => i2c_sda_in,
        i2c_sda_oe => i2c_sda_oe,

        spi_miso => spi_miso,
        spi_mosi => spi_mosi,
        spi_sclk => spi_sclk,
        spi_ss_n => spi_ss_n,

        pio_export => nios_pio,

        rst_reset_n => nios_reset_n,
        clk_clk => nios_clk--,
    );

    si45_oe_n <= '0';
    si45_rst_n <= '1';
    si45_spi_in <= spi_mosi;
--    spi_miso <= si45_spi_out;
    si45_spi_sclk <= spi_sclk;
    si45_spi_cs_n <= spi_ss_n(0);



    -- I2C
    i2c_scl_in <= not i2c_scl_oe;
    i2c_sda_in <=
        malibu_i2c_sda and
        '1';
    malibu_i2c_scl <= ZERO when i2c_scl_oe = '1' else 'Z';
    malibu_i2c_sda <= ZERO when i2c_sda_oe = '1' else 'Z';

    -- SPI
    malibu_spi_sdi <= spi_mosi;
--    spi_miso <= malibu_spi_sdo;
    malibu_spi_sck <= spi_sclk;

    spi_miso <= si45_spi_out when spi_ss_n(0) = '0' else
                malibu_spi_sdo when spi_ss_n(1) = '0' else '0';

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- MALIBU

    -- 160 MHz
    i_malibu_clk : entity work.ip_altpll
    generic map (
        DIV => 25,
        MUL => 32--,
    )
    port map (
        c0 => malibu_clk,
        locked => open,
        areset => '0',
        inclk0 => clk_aux--,
    );

    malibu_ck_fpga_1 <= '0';
    malibu_pll_reset <= '0';
    malibu_ck_fpga_0 <= malibu_clk;

    -- reset tdc and digital part
    malibu_chip_reset <= nios_pio(16);

    e_test_pulse : entity work.clkdiv
    generic map ( P => 125 )
    port map ( clkout => malibu_pll_test, rst_n => reset_n, clk => clk_aux );

    i_malibu : entity work.malibu_dec
    generic map (
        N => 2--,
    )
    port map (
        data_clk    => malibu_rx_data_clk,
        data        => malibu_rx_data,
        datak       => malibu_rx_datak,

        rx_data => malibu_data(1 downto 0),
        rx_clk  => malibu_clk,

        reset   => not reset_n--,
    );

    i_frame_rcv : entity work.frame_rcv
    port map (
        i_rst => not reset_n,
        i_clk => malibu_rx_data_clk,
        i_data => malibu_rx_data(7 downto 0),
        i_byteisk => malibu_rx_datak(0),
        i_dser_no_sync => '0',

        o_frame_number => open,
        o_frame_info => open,
        o_frame_info_ready => open,
        o_new_frame => open,
        o_word => malibu_word,
        o_new_word => open,

        o_end_of_frame => open,
        o_crc_error => open,
        o_crc_err_count => open--,
    );

    i_mutrig_datapath : entity work.mutrig_datapath
    port map (
        i_rst => not reset_n,
        i_stic_txd => malibu_data(0 downto 0),
        i_refclk_125 => clk_aux,

        --interface to asic fifos
        i_clk_core => '0',
        o_fifo_empty => open,
        o_fifo_data => open,
        i_fifo_rd => '1',
        --slow control
        i_SC_disable_dec => '0',
        i_SC_mask => (others => '0'),
        i_SC_datagen_enable => '0',
        i_SC_datagen_shortmode => '0',
        i_SC_datagen_count => (others => '0'),
        --monitors
        o_receivers_usrclk => open,
        o_receivers_pll_lock => open,
        o_receivers_dpa_lock=> open,
        o_receivers_ready => open,
        o_frame_desync => open,
        o_buffer_full => open--,
    );

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- QSFP

    QSFP_ModSel_n <= '1';
    QSFP_Rst_n <= '1';
    QSFP_LPM <= '0';

    i_qsfp : entity work.xcvr_s4
    generic map (
        data_rate => 6250,
        pll_freq => 156.25--,
    )
    port map (
        -- avalon slave interface
        avs_address     => avm_qsfp.address(15 downto 2),
        avs_read        => avm_qsfp.read,
        avs_readdata    => avm_qsfp.readdata,
        avs_write       => avm_qsfp.write,
        avs_writedata   => avm_qsfp.writedata,
        avs_waitrequest => avm_qsfp.waitrequest,

        tx_data     => qsfp_tx_data,
        tx_datak    => qsfp_tx_datak,

        rx_data     => qsfp_rx_data,
        rx_datak    => qsfp_rx_datak,

        tx_clkout   => open,
        tx_clkin    => (others => qsfp_pll_clk),
        rx_clkout   => open,
        rx_clkin    => (others => qsfp_pll_clk),

        tx_p        => qsfp_tx,
        rx_p        => qsfp_rx,

        pll_refclk  => qsfp_pll_clk,
        cdr_refclk  => qsfp_pll_clk,

        reset   => not nios_reset_n,
        clk     => nios_clk--,
    );

    qsfp_tx_data(127 downto 32) <=
          X"03CAFE" & work.util.D28_5
        & X"02BABE" & work.util.D28_5
        & X"01DEAD" & work.util.D28_5;

    qsfp_tx_datak(15 downto 4) <=
          "0001"
        & "0001"
        & "0001";

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- SLOW CONTROL

    i_sc_ram : entity work.ip_ram
    generic map (
        ADDR_WIDTH => 14,
        DATA_WIDTH => 32--,
    )
    port map (
        address_b => avm_sc.address(15 downto 2),
        q_b => avm_sc.readdata,
        wren_b => avm_sc.write,
        data_b => avm_sc.writedata,
        clock_b => qsfp_pll_clk,

        address_a => ram_addr_a(13 downto 0),
        q_a => ram_rdata_a,
        wren_a => ram_we_a,
        data_a => ram_wdata_a,
        clock_a => qsfp_pll_clk--,
    );
    avm_sc.waitrequest <= '0';

    i_sc : entity work.sc_s4
    port map (
        clk => qsfp_pll_clk,
        reset_n => reset_n,
        enable => '1',

        mem_data_in => ram_rdata_a,

        link_data_in => qsfp_rx_data(31 downto 0),
        link_data_in_k => qsfp_rx_datak(3 downto 0),

        fifo_data_out => sc_to_fifo,
        fifo_we => sc_to_fifo_we,

        mem_data_out => ram_wdata_a,
        mem_addr_out => ram_addr_a,
        mem_wren => ram_we_a,

        stateout => open--,
    );

    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- data gen
    
    i_data_gen : entity work.data_generator
    port map (
        clk => qsfp_pll_clk,
        reset => not reset_n,
        enable_pix => '1',
        --enable_sc:         	in  std_logic;
        random_seed => (others => '1'),
        data_pix_generated => data_to_fifo,
        --data_sc_generated:   	out std_logic_vector(31 downto 0);
        data_pix_ready => data_to_fifo_we,
        --data_sc_ready:      	out std_logic;
        start_global_time => (others => '0')--,
              -- TODO: add some rate control
    );

    i_merger : entity work.data_merger
    port map (
        clk                     => qsfp_pll_clk,
        reset                   => not reset_n,
        fpga_ID_in              => (5=>'1',others => '0'),
        FEB_type_in             => "111010",
        state_idle              => '1',
        state_run_prepare       => '0',
        state_sync              => '0',
        state_running           => '0',
        state_terminating       => '0',
        state_link_test         => '0',
        state_sync_test         => '0',
        state_reset             => '0',
        state_out_of_DAQ        => '0',
        data_out                => qsfp_tx_data(31 downto 0),
        data_is_k               => qsfp_tx_datak(3 downto 0),
        data_in                 => data_from_fifo,
        data_in_slowcontrol     => sc_from_fifo,
        slowcontrol_fifo_empty  => sc_from_fifo_empty,
        data_fifo_empty         => '1',--data_from_fifo_empty,
        slowcontrol_read_req    => sc_from_fifo_re,
        data_read_req           => data_from_fifo_re,
        terminated              => open,
        override_data_in        => (others => '0'),
        override_data_is_k_in   => (others => '0'),
        override_req            => '0',
        override_granted        => open,
        data_priority           => '0',
        leds                    => open -- debug
    );

    i_data_fifo : entity work.mergerfifo
    generic map (
        DEVICE => "Stratix IV"--,
    )
    port map (
        data    => data_to_fifo,
        rdclk   => qsfp_pll_clk,
        rdreq   => data_from_fifo_re,
        wrclk   => qsfp_pll_clk,
        wrreq   => data_to_fifo_we,
        q       => data_from_fifo,
        rdempty => data_from_fifo_empty,
        wrfull  => open--,
    );

    i_sc_fifo : entity work.mergerfifo -- ip_fifo
    generic map (
--        ADDR_WIDTH => 11,
--        DATA_WIDTH => 36,
        DEVICE => "Stratix IV"--,
    )
    port map (
        data    => sc_to_fifo,
        rdclk   => qsfp_pll_clk,
        rdreq   => sc_from_fifo_re,
        wrclk   => qsfp_pll_clk,
        wrreq   => sc_to_fifo_we,
        q       => sc_from_fifo,
        rdempty => sc_from_fifo_empty,
        wrfull  => open--,
    );

    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- POD

--    pod_tx_reset <= '0';
--    pod_tx_reset <= '0';
--
--    i_pod : entity work.xcvr_s4
--    generic map (
--        data_rate => 5000,
--        pll_freq => 125--,
--    )
--    port map (
--        -- avalon slave interface
--        avs_address     => avm_pod.address(15 downto 2),
--        avs_read        => avm_pod.read,
--        avs_readdata    => avm_pod.readdata,
--        avs_write       => avm_pod.write,
--        avs_writedata   => avm_pod.writedata,
--        avs_waitrequest => avm_pod.waitrequest,
--
--        tx_data     => X"02CAFE" & work.util.D28_5
--                     & X"02BABE" & work.util.D28_5
--                     & X"01DEAD" & work.util.D28_5
--                     & X"00BEEF" & work.util.D28_5,
--        tx_datak    => "0001"
--                     & "0001"
--                     & "0001"
--                     & "0001",
--
--        rx_data => open,
--        rx_datak => open,
--
--        tx_clkout   => open,
--        tx_clkin    => (others => pod_pll_clk),
--        rx_clkout   => open,
--        rx_clkin    => (others => pod_pll_clk),
--
--        tx_p        => pod_tx,
--        rx_p        => pod_rx,
--
--        pll_refclk  => pod_pll_clk,
--        cdr_refclk  => pod_pll_clk,
--
--        reset   => not nios_reset_n,
--        clk     => nios_clk--,
--    );

    ----------------------------------------------------------------------------

end architecture;
