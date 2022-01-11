library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    LED                         : out   std_logic_vector(3 downto 0);

    FLASH_A                     : out   std_logic_vector(26 downto 1);
    FLASH_D                     : inout std_logic_vector(31 downto 0);
    FLASH_OE_n                  : inout std_logic;
    FLASH_WE_n                  : out   std_logic;
    FLASH_CE_n                  : out   std_logic_vector(1 downto 0);
    FLASH_ADV_n                 : out   std_logic;
    FLASH_CLK                   : out   std_logic;
    FLASH_RESET_n               : out   std_logic;

    FAN_I2C_SCL                 : inout std_logic;
    FAN_I2C_SDA                 : inout std_logic;
    POWER_MONITOR_I2C_SCL       : inout std_logic;
    POWER_MONITOR_I2C_SDA       : inout std_logic;
    TEMP_I2C_SCL                : inout std_logic;
    TEMP_I2C_SDA                : inout std_logic;

--    QSFPB_INTERRUPT_n           : in    std_logic;
    QSFPB_LP_MODE               : out   std_logic;
--    QSFPB_MOD_PRS_n             : in    std_logic;
    QSFPB_MOD_SEL_n             : out   std_logic;
    QSFPB_REFCLK_p              : in    std_logic;
    QSFPB_RST_n                 : out   std_logic;
--    QSFPB_SCL                   : out   std_logic;
--    QSFPB_SDA                   : inout std_logic;
    QSFPB_TX_p                  : out   std_logic_vector(3 downto 0);
    QSFPB_RX_p                  : in    std_logic_vector(3 downto 0);

--    QSFPA_INTERRUPT_n           : in    std_logic;
    QSFPA_LP_MODE               : out   std_logic;
--    QSFPA_MOD_PRS_n             : in    std_logic;
    QSFPA_MOD_SEL_n             : out   std_logic;
    QSFPA_REFCLK_p              : in    std_logic;
    QSFPA_RST_n                 : out   std_logic;
--    QSFPA_SCL                   : out   std_logic;
--    QSFPA_SDA                   : inout std_logic;
    QSFPA_TX_p                  : out   std_logic_vector(3 downto 0);
    QSFPA_RX_p                  : in    std_logic_vector(3 downto 0);

    SMA_CLKIN                   : in    std_logic;
    SMA_CLKOUT                  : out   std_logic;

    CPU_RESET_n                 : in    std_logic;
    CLK_50_B2J                  : in    std_logic--;
);
end entity;

architecture rtl of top is

    signal clk_50               : std_logic;
    signal reset_50_n           : std_logic;
    signal clk_125              : std_logic;
    signal reset_125_n          : std_logic;

    signal flash_cs_n           : std_logic;
    signal flash_rst_n          : std_logic;
    signal nios_reset_n         : std_logic;

    signal nios_i2c_scl         : std_logic;
    signal nios_i2c_scl_oe      : std_logic;
    signal nios_i2c_sda         : std_logic;
    signal nios_i2c_sda_oe      : std_logic;
    signal nios_i2c_mask        : std_logic_vector(31 downto 0);

    signal nios_pio             : std_logic_vector(31 downto 0);

    signal av_xcvr0             : work.util.avalon_t;

begin

    clk_50 <= CLK_50_B2J;

    e_reset_50_n : entity work.reset_sync
    port map ( o_reset_n => reset_50_n, i_reset_n => CPU_RESET_n, i_clk => clk_50 );

    -- 50 MHz -> 1 Hz
    e_clk_50_hz : entity work.clkdiv
    generic map ( P => 50000000 )
    port map ( o_clk => LED(0), i_reset_n => reset_50_n, i_clk => clk_50 );



    e_iopll_50to125 : component work.cmp.ip_pll_50to125
    port map (
        refclk => clk_50,
        outclk_0 => SMA_CLKOUT,
        rst => not reset_50_n--,
    );

    -- SMA input -> clk_125
    e_clkctrl : component work.cmp.ip_clkctrl
    port map (
        inclk => SMA_CLKIN,
        outclk => clk_125--,
    );

    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => CPU_RESET_n, i_clk => clk_125 );

    -- 125 MHz -> 1 Hz
    e_clk_125_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( o_clk => LED(1), i_reset_n => reset_125_n, i_clk => clk_125 );



    -- generate reset sequence for flash and nios
    e_nios_reset_n : entity work.debouncer
    generic map (
        W => 2,
        N => integer(50.0e6 * 0.100) -- 100ms
    )
    port map (
        i_d(0)          => '1',
        o_q(0)          => flash_rst_n,
        i_d(1)          => flash_rst_n,
        o_q(1)          => nios_reset_n,

        i_reset_n       => reset_50_n,
        i_clk           => clk_50--,
    );
    flash_reset_n <= flash_rst_n;



    e_nios : component work.cmp.nios
    port map (
        flash_tcm_address_out(27 downto 2) => FLASH_A,
        flash_tcm_data_out => FLASH_D,
        flash_tcm_read_n_out(0) => FLASH_OE_n,
        flash_tcm_write_n_out(0) => FLASH_WE_n,
        flash_tcm_chipselect_n_out(0) => flash_cs_n,

        i2c_scl_in              => nios_i2c_scl,
        i2c_scl_oe              => nios_i2c_scl_oe,
        i2c_sda_in              => nios_i2c_sda,
        i2c_sda_oe              => nios_i2c_sda_oe,
        i2c_mask_export         => nios_i2c_mask,

        pio_export              => nios_pio,

        avm_xcvr0_address       => av_xcvr0.address(17 downto 0),
        avm_xcvr0_read          => av_xcvr0.read,
        avm_xcvr0_readdata      => av_xcvr0.readdata,
        avm_xcvr0_write         => av_xcvr0.write,
        avm_xcvr0_writedata     => av_xcvr0.writedata,
        avm_xcvr0_waitrequest   => av_xcvr0.waitrequest,

        avm_xcvr0_reset_reset_n => reset_50_n,
        avm_xcvr0_clock_clk     => clk_50,

        rst_reset_n             => nios_reset_n,
        clk_clk                 => clk_50--,
    );

    FLASH_CE_n <= (flash_cs_n, flash_cs_n);
    FLASH_ADV_n <= '0';
    FLASH_CLK <= '0';

    LED(3) <= nios_pio(7);



    e_i2c_mux : entity work.i2c_mux
    port map (
        io_scl(0)       => FAN_I2C_SCL,
        io_sda(0)       => FAN_I2C_SDA,
        io_scl(1)       => TEMP_I2C_SCL,
        io_sda(1)       => TEMP_I2C_SDA,
        io_scl(2)       => POWER_MONITOR_I2C_SCL,
        io_sda(2)       => POWER_MONITOR_I2C_SDA,

        o_scl           => nios_i2c_scl,
        i_scl_oe        => nios_i2c_scl_oe,
        o_sda           => nios_i2c_sda,
        i_sda_oe        => nios_i2c_sda_oe,
        i_mask          => nios_i2c_mask--,
    );



    QSFPB_LP_MODE <= '0';
    QSFPB_MOD_SEL_n <= '1';
    QSFPB_RST_n <= '1';

    QSFPA_LP_MODE <= '0';
    QSFPA_MOD_SEL_n <= '1';
    QSFPA_RST_n <= '1';

    e_xcvr0_block : entity work.xcvr_block
    generic map (
--        g_MODE => "enh",
        g_XCVR_N => 2,
        g_CHANNELS => 4,
        g_REFCLK_MHZ => 125.0,
        g_RATE_MBPS => 5000,
        g_CLK_MHZ => 50.0--,
    )
    port map (
        o_rx_data               => open,
        o_rx_datak              => open,

        i_tx_data(7)            => X"07CAFEBC",
        i_tx_data(6)            => X"06BABEBC",
        i_tx_data(5)            => X"05DEADBC",
        i_tx_data(4)            => X"04BEEFBC",
        i_tx_data(3)            => X"03CAFEBC",
        i_tx_data(2)            => X"02BABEBC",
        i_tx_data(1)            => X"01DEADBC",
        i_tx_data(0)            => X"00BEEFBC",
        i_tx_datak(7)           => "0001",
        i_tx_datak(6)           => "0001",
        i_tx_datak(5)           => "0001",
        i_tx_datak(4)           => "0001",
        i_tx_datak(3)           => "0001",
        i_tx_datak(2)           => "0001",
        i_tx_datak(1)           => "0001",
        i_tx_datak(0)           => "0001",

        i_tx_clk                => (others => clk_125),
        i_rx_clk                => (others => clk_125),

        i_rx_serial(7 downto 4) => QSFPB_RX_p,
        i_rx_serial(3 downto 0) => QSFPA_RX_p,
        o_tx_serial(7 downto 4) => QSFPB_TX_p,
        o_tx_serial(3 downto 0) => QSFPA_TX_p,

        i_refclk                => (others => clk_125),

        i_avs_address           => av_xcvr0.address(17 downto 0),
        i_avs_read              => av_xcvr0.read,
        o_avs_readdata          => av_xcvr0.readdata,
        i_avs_write             => av_xcvr0.write,
        i_avs_writedata         => av_xcvr0.writedata,
        o_avs_waitrequest       => av_xcvr0.waitrequest,

        i_reset_n               => reset_50_n,
        i_clk                   => clk_50--,
    );

end architecture;
