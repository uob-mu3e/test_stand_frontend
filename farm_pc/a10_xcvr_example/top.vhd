library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    BUTTON      : in    std_logic_vector(3 downto 0);
    SW          : in    std_logic_vector(1 downto 0);
    LED         : out   std_logic_vector(3 downto 0);
    LED_BRACKET : out   std_logic_vector(3 downto 0);
    HEX0_D      : out   std_logic_vector(6 downto 0);
    HEX0_DP     : out   std_logic;
    HEX1_D      : out   std_logic_vector(6 downto 0);
    HEX1_DP     : out   std_logic;

    FLASH_A         : out   std_logic_vector(26 downto 1);
    FLASH_D         : inout std_logic_vector(31 downto 0);
    FLASH_OE_n      : inout std_logic;
    FLASH_WE_n      : out   std_logic;
    FLASH_CE_n      : out   std_logic_vector(1 downto 0);
    FLASH_ADV_n     : out   std_logic;
    FLASH_CLK       : out   std_logic;
    FLASH_RESET_n   : out   std_logic;

    FAN_I2C_SCL             : out   std_logic;
    FAN_I2C_SDA             : inout std_logic;
    POWER_MONITOR_I2C_SCL   : out   std_logic;
    POWER_MONITOR_I2C_SDA   : inout std_logic;
    TEMP_I2C_SCL            : out   std_logic;
    TEMP_I2C_SDA            : inout std_logic;

--    QSFPA_INTERRUPT_n   : in    std_logic;
    QSFPA_LP_MODE       : out   std_logic;
--    QSFPA_MOD_PRS_n     : in    std_logic;
    QSFPA_MOD_SEL_n     : out   std_logic;
    QSFPA_REFCLK_p      : in    std_logic;
    QSFPA_RST_n         : out   std_logic;
--    QSFPA_SCL           : out   std_logic;
--    QSFPA_SDA           : inout std_logic;
    QSFPA_TX_p          : out   std_logic_vector(3 downto 0);
    QSFPA_RX_p          : in    std_logic_vector(3 downto 0);

    SMA_CLKIN       : in    std_logic;
    SMA_CLKOUT      : out   std_logic;

    CPU_RESET_n     : in    std_logic;
    CLK_50_B2J      : in    std_logic--;
);
end entity;

architecture rtl of top is

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal sw_q : std_logic_vector(1 downto 0);
    signal button_q : std_logic_vector(3 downto 0);

    signal i2c_scl_in   : std_logic;
    signal i2c_scl_oe   : std_logic;
    signal i2c_sda_in   : std_logic;
    signal i2c_sda_oe   : std_logic;

    signal nios_clk      : std_logic;
    signal nios_rst_n    : std_logic;
    signal flash_rst_n  : std_logic;

    signal refclk_125   : std_logic;

    signal clk_250      : std_logic;

    signal wd_rst_n     : std_logic;

    signal nios_pio_i : std_logic_vector(31 downto 0);

    signal flash_ce_n_i : std_logic;

    signal clk_125_cnt : unsigned(31 downto 0);
    signal hex1, hex0 : std_logic_vector(3 downto 0);

    signal avm_qsfp : work.mu3e.avalon_t;

begin

    -- SMA input -> refclk_125
    i_clkctrl : component work.cmp.ip_clkctrl
    port map (
        inclk => SMA_CLKIN,
        outclk => refclk_125--,
    );

    i_iopll_125 : component work.cmp.ip_iopll_125
    port map (
        refclk => CLK_50_B2J,
        outclk_0 => SMA_CLKOUT,
        rst => not CPU_RESET_n--,
    );



    QSFPA_LP_MODE <= '0';
    QSFPA_MOD_SEL_n <= '1';
    QSFPA_RST_n <= '1';



    nios_clk <= refclk_125;

    i_debouncer : entity work.debouncer
    generic map (
        W => 6,
        N => 125 * 10**3 -- 1ms
    )
    port map (
        d(1 downto 0) => SW(1 downto 0),
        q(1 downto 0) => sw_q(1 downto 0),
        d(5 downto 2) => BUTTON(3 downto 0),
        q(5 downto 2) => button_q(3 downto 0),
        arst_n => CPU_RESET_n,
        clk => nios_clk--,
    );



    -- generate reset sequence for flash and nios
    i_reset_ctrl : entity work.reset_ctrl
    generic map (
        W => 2,
        N => 125 * 10**5 -- 100ms
    )
    port map (
        rstout_n(1) => flash_rst_n,
        rstout_n(0) => nios_rst_n,

--        rst_n => CPU_RESET_n and wd_rst_n and SW(1),
        rst_n => CPU_RESET_n,
        clk => nios_clk--,
    );
    LED(0) <= not flash_rst_n;
    LED(1) <= not nios_rst_n;

    watchdog_i : entity work.watchdog
    generic map (
        W => 4,
        N => 125 * 10**6 -- 1s
    )
    port map (
        d => nios_pio_i(3 downto 0),

        rstout_n => wd_rst_n,

        rst_n => CPU_RESET_n,
        clk => nios_clk--,
    );
    LED(2) <= not wd_rst_n;



    i_nios : component work.cmp.nios
    port map (
        avm_qsfp_address        => avm_qsfp.address(15 downto 0),
        avm_qsfp_read           => avm_qsfp.read,
        avm_qsfp_readdata       => avm_qsfp.readdata,
        avm_qsfp_write          => avm_qsfp.write,
        avm_qsfp_writedata      => avm_qsfp.writedata,
        avm_qsfp_waitrequest    => avm_qsfp.waitrequest,

        flash_tcm_address_out(27 downto 2) => FLASH_A,
        flash_tcm_data_out => FLASH_D,
        flash_tcm_read_n_out(0) => FLASH_OE_n,
        flash_tcm_write_n_out(0) => FLASH_WE_n,
        flash_tcm_chipselect_n_out(0) => flash_ce_n_i,

        i2c_scl_in  => i2c_scl_in,
        i2c_scl_oe  => i2c_scl_oe,
        i2c_sda_in  => i2c_sda_in,
        i2c_sda_oe  => i2c_sda_oe,

        spi_MISO    => '-',
        spi_MOSI    => open,
        spi_SCLK    => open,
        spi_SS_n    => open,

        pio_export => nios_pio_i,

        rst_reset_n => nios_rst_n,
        clk_clk     => nios_clk--,
    );

    FLASH_CE_n <= (flash_ce_n_i, flash_ce_n_i);
    FLASH_ADV_n <= '0';
    FLASH_CLK <= '0';
    FLASH_RESET_n <= flash_rst_n;



    -- I2C clock
    i2c_scl_in <= not i2c_scl_oe;
    FAN_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';
    TEMP_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';
    POWER_MONITOR_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';

    -- I2C data
    i2c_sda_in <= FAN_I2C_SDA and
                  TEMP_I2C_SDA and
                  POWER_MONITOR_I2C_SDA and
                  '1';
    FAN_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';
    TEMP_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';
    POWER_MONITOR_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';



    process(refclk_125)
    begin
    if rising_edge(refclk_125) then
        clk_125_cnt <= clk_125_cnt + 1;
    end if; -- rising_edge
    end process;

--    hex1 <= clk_125_cnt(31 downto 28);
    hex1 <= nios_pio_i(7 downto 4);
    hex0 <= std_logic_vector(clk_125_cnt)(27 downto 24);
    HEX1_DP <= '1';
    HEX0_DP <= '1';

    i_seg7_hex1 : entity work.seg7_lut
    port map (
        hex => hex1,
        seg => HEX1_D--,
    );
    i_seg7_hex0 : entity work.seg7_lut
    port map (
        hex => hex0,
        seg => HEX0_D--,
    );



    i_qsfp : entity work.xcvr_a10
    port map (
        -- avalon slave interface
        avs_address     => avm_qsfp.address(15 downto 2),
        avs_read        => avm_qsfp.read,
        avs_readdata    => avm_qsfp.readdata,
        avs_write       => avm_qsfp.write,
        avs_writedata   => avm_qsfp.writedata,
        avs_waitrequest => avm_qsfp.waitrequest,

        tx3_data    => X"03CAFEBC",
        tx2_data    => X"02BABEBC",
        tx1_data    => X"01DEADBC",
        tx0_data    => X"00BEEFBC",
        tx3_datak   => "0001",
        tx2_datak   => "0001",
        tx1_datak   => "0001",
        tx0_datak   => "0001",

        rx3_data    => open,
        rx2_data    => open,
        rx1_data    => open,
        rx0_data    => open,
        rx3_datak   => open,
        rx2_datak   => open,
        rx1_datak   => open,
        rx0_datak   => open,

        tx_clkout   => open,
        tx_clkin    => (others => refclk_125),
        rx_clkout   => open,
        rx_clkin    => (others => refclk_125),

        tx_p        => QSFPA_TX_p,
        rx_p        => QSFPA_RX_p,

        pll_refclk  => refclk_125,
        cdr_refclk  => refclk_125,

        reset   => not nios_rst_n,
        clk     => nios_clk--,
    );

end architecture;
