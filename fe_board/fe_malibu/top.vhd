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

    pod_pll_clk     : in    std_logic;

    pod_tx_reset    : out   std_logic;
    pod_rx_reset    : out   std_logic;

    pod_tx          : out   std_logic_vector(3 downto 0);
    pod_rx          : in    std_logic_vector(3 downto 0);


    -- MSCB

    mscb_data_in    : in    std_logic;
    mscb_data_out   : out   std_logic;
    mscb_oe         : out   std_logic;



    --

    led_n       : out   std_logic_vector(15 downto 0);

    PushButton  : in    std_logic_vector(1 downto 0);



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

    signal led : std_logic_vector(led_n'range) := (others => '0');

    signal nios_clk, nios_reset_n : std_logic;
    signal nios_pio : std_logic_vector(31 downto 0);

    signal i2c_scl_in, i2c_scl_oe, i2c_sda_in, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(1 downto 0);



    signal malibu_clk : std_logic;
    signal malibu_rx_data_clk : std_logic;

    signal fifo_data : std_logic_vector(35 downto 0);
    signal fifo_data_empty, fifo_data_read : std_logic;



    signal av_pod, av_qsfp : work.util.avalon_t;

    signal qsfp_tx_data : std_logic_vector(127 downto 0);
    signal qsfp_tx_datak : std_logic_vector(15 downto 0);

    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);

    signal qsfp_reset_n : std_logic;



    signal av_sc : work.util.avalon_t;

    signal mscb_to_nios_parallel_in : std_logic_vector(11 downto 0);
    signal mscb_from_nios_parallel_out : std_logic_vector(11 downto 0);
    signal mscb_counter_in : unsigned(15 downto 0);



    signal av_test : work.util.avalon_t;

begin

    led_n <= not led;

    -- 125 MHz
    e_clk_aux_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(15), rst_n => reset_n, clk => clk_aux );

    -- 156.25 MHz
    e_clk_qsfp_hz : entity work.clkdiv
    generic map ( P => 156250000 )
    port map ( clkout => led(14), rst_n => reset_n, clk => qsfp_pll_clk );

    -- 125 MHz
    e_clk_pod_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(13), rst_n => reset_n, clk => pod_pll_clk );

    ----------------------------------------------------------------------------
    -- NIOS

    nios_clk <= clk_aux;

    e_nios_reset_n : entity work.reset_sync
    port map ( rstout_n => nios_reset_n, arst_n => reset_n, clk => nios_clk );

    e_qsfp_reset_n : entity work.reset_sync
    port map ( rstout_n => qsfp_reset_n, arst_n => reset_n, clk => qsfp_pll_clk );

    led(12) <= nios_pio(7);

    e_nios : component work.cmp.nios
    port map (
        avm_qsfp_address        => av_qsfp.address(15 downto 0),
        avm_qsfp_read           => av_qsfp.read,
        avm_qsfp_readdata       => av_qsfp.readdata,
        avm_qsfp_write          => av_qsfp.write,
        avm_qsfp_writedata      => av_qsfp.writedata,
        avm_qsfp_waitrequest    => av_qsfp.waitrequest,

        avm_pod_address         => av_pod.address(15 downto 0),
        avm_pod_read            => av_pod.read,
        avm_pod_readdata        => av_pod.readdata,
        avm_pod_write           => av_pod.write,
        avm_pod_writedata       => av_pod.writedata,
        avm_pod_waitrequest     => av_pod.waitrequest,

        avm_sc_address          => av_sc.address(17 downto 0),
        avm_sc_read             => av_sc.read,
        avm_sc_readdata         => av_sc.readdata,
        avm_sc_write            => av_sc.write,
        avm_sc_writedata        => av_sc.writedata,
        avm_sc_waitrequest      => av_sc.waitrequest,

        avm_test_address        => av_test.address(15 downto 0),
        avm_test_read           => av_test.read,
        avm_test_readdata       => av_test.readdata,
        avm_test_write          => av_test.write,
        avm_test_writedata      => av_test.writedata,
        avm_test_waitrequest    => av_test.waitrequest,

        avm_clk_clk          => qsfp_pll_clk,
        avm_reset_reset_n    => qsfp_reset_n,

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

        -- mscb
        parallel_mscb_in_export => mscb_to_nios_parallel_in,
        parallel_mscb_out_export => mscb_from_nios_parallel_out,
        counter_in_export => std_logic_vector(mscb_counter_in),

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
    e_malibu_clk : entity work.ip_altpll
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

    -- 16 MHz
    e_malibu_rx_data_clk_hz : entity work.clkdiv
    generic map ( P => 16000000 )
    port map ( clkout => led(8), rst_n => reset_n, clk => malibu_rx_data_clk );

    e_mutrig_datapath : entity work.mutrig_datapath
    generic map (
        N_ASICS => 1,
        LVDS_PLL_FREQ => 160.0,
        LVDS_DATA_RATE => 160--,
    )
    port map (
        i_rst => not reset_n,
        i_stic_txd => malibu_data(0 downto 0),
        i_refclk_125 => malibu_clk,
        i_ts_clk => malibu_clk,
        i_ts_rst => not reset_n,

        --interface to asic fifos
        i_clk_core => qsfp_pll_clk,
        o_fifo_empty => fifo_data_empty,
        o_fifo_data => fifo_data,
        i_fifo_rd => fifo_data_read,

        --slow control
        i_SC_disable_dec => '0',
        i_SC_mask => (others => '0'),
        i_SC_datagen_enable => '0',
        i_SC_datagen_shortmode => '0',
        i_SC_datagen_count => (others => '0'),
        --monitors
        o_receivers_usrclk => malibu_rx_data_clk,
        o_receivers_pll_lock => open,
        o_receivers_dpa_lock=> open,
        o_receivers_ready(0) => led(9),
        o_frame_desync => led(10),
        o_buffer_full => led(11)--,
    );

    ----------------------------------------------------------------------------



    e_data_sc_path : entity work.data_sc_path
    port map (
        i_sc_address        => av_sc.address(17 downto 2),
        i_sc_read           => av_sc.read,
        o_sc_readdata       => av_sc.readdata,
        i_sc_write          => av_sc.write,
        i_sc_writedata      => av_sc.writedata,
        o_sc_waitrequest    => av_sc.waitrequest,

        i_fifo_data         => fifo_data,
        i_fifo_data_empty   => fifo_data_empty,
        o_fifo_data_read    => fifo_data_read,

        i_link_data         => qsfp_rx_data(31 downto 0),
        i_link_datak        => qsfp_rx_datak(3 downto 0),

        o_link_data         => qsfp_tx_data(31 downto 0),
        o_link_datak        => qsfp_tx_datak(3 downto 0),

        i_reset             => reset_n,
        i_clk               => qsfp_pll_clk--,
    );



    ----------------------------------------------------------------------------
    -- MSCB

    i_mscb : entity work.mscb
    port map (
        nios_clk                    => nios_clk,
        reset                       => not nios_reset_n,
        mscb_to_nios_parallel_in    => mscb_to_nios_parallel_in,
        mscb_from_nios_parallel_out => mscb_from_nios_parallel_out,
        mscb_data_in                => mscb_data_in,
        mscb_data_out               => mscb_data_out,
        mscb_oe                     => mscb_oe,
        mscb_counter_in             => mscb_counter_in--,
    );

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- QSFP
    -- (data and slow_control)

    QSFP_ModSel_n <= '1';
    QSFP_Rst_n <= '1';
    QSFP_LPM <= '0';

    e_qsfp : entity work.xcvr_s4
    generic map (
        NUMBER_OF_CHANNELS_g => 4,
        CHANNEL_WIDTH_g => 32,
        INPUT_CLOCK_FREQUENCY_g => 156250000,
        DATA_RATE_g => 6250,
        CLK_MHZ_g => 125--,
    )
    port map (
        i_tx_data   => qsfp_tx_data,
        i_tx_datak  => qsfp_tx_datak,

        o_rx_data   => qsfp_rx_data,
        o_rx_datak  => qsfp_rx_datak,

        o_tx_clkout => open,
        i_tx_clkin  => (others => qsfp_pll_clk),
        o_rx_clkout => open,
        i_rx_clkin  => (others => qsfp_pll_clk),

        o_tx_serial => qsfp_tx,
        i_rx_serial => qsfp_rx,

        i_pll_clk   => qsfp_pll_clk,
        i_cdr_clk   => qsfp_pll_clk,

        i_avs_address     => av_qsfp.address(15 downto 2),
        i_avs_read        => av_qsfp.read,
        o_avs_readdata    => av_qsfp.readdata,
        i_avs_write       => av_qsfp.write,
        i_avs_writedata   => av_qsfp.writedata,
        o_avs_waitrequest => av_qsfp.waitrequest,

        i_reset     => not nios_reset_n,
        i_clk       => nios_clk--,
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
    -- POD
    -- (reset system)

    pod_tx_reset <= '0';
    pod_rx_reset <= '0';

    e_pod : entity work.xcvr_s4
    generic map (
        NUMBER_OF_CHANNELS_g => 4,
        CHANNEL_WIDTH_g => 8,
        INPUT_CLOCK_FREQUENCY_g => 125000000,
        DATA_RATE_g => 1250,
        CLK_MHZ_g => 125--,
    )
    port map (
        -- avalon slave interface
        i_avs_address     => av_pod.address(15 downto 2),
        i_avs_read        => av_pod.read,
        o_avs_readdata    => av_pod.readdata,
        i_avs_write       => av_pod.write,
        i_avs_writedata   => av_pod.writedata,
        o_avs_waitrequest => av_pod.waitrequest,

        i_tx_data   => work.util.D28_5
                     & work.util.D28_5
                     & work.util.D28_5
                     & work.util.D28_5,
        i_tx_datak  => "1"
                     & "1"
                     & "1"
                     & "1",

        o_rx_data   => open,
        o_rx_datak  => open,

        o_tx_clkout => open,
        i_tx_clkin  => (others => pod_pll_clk),
        o_rx_clkout => open,
        i_rx_clkin  => (others => pod_pll_clk),

        o_tx_serial => pod_tx,
        i_rx_serial => pod_rx,

        i_pll_clk   => pod_pll_clk,
        i_cdr_clk   => pod_pll_clk,

        i_reset     => not nios_reset_n,
        i_clk       => nios_clk--,
    );

    ----------------------------------------------------------------------------

end architecture;
