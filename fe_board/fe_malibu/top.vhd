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



    -- Si5342
    si42_oe_n       : out   std_logic; -- <= '0'
    si42_rst_n      : out   std_logic; -- reset
    si42_spi_out    : in    std_logic; -- slave data out
    si42_spi_in     : out   std_logic; -- slave data in
    si42_spi_sclk   : out   std_logic; -- clock
    si42_spi_cs_n   : out   std_logic; -- chip select

    -- Si5345
    si45_oe_n       : out   std_logic; -- <= '0'
    si45_rst_n      : out   std_logic; -- reset
    si45_spi_out    : in    std_logic; -- slave data out
    si45_spi_in     : out   std_logic; -- slave data in
    si45_spi_sclk   : out   std_logic; -- clock
    si45_spi_cs_n   : out   std_logic; -- chip select



    -- QSFP

    -- si5345 out2 (156.25 MHz)
    qsfp_pll_clk    : in    std_logic;

    QSFP_ModSel_n   : out   std_logic; -- module select (i2c)
    QSFP_Rst_n      : out   std_logic;
    QSFP_LPM        : out   std_logic; -- Low Power Mode

    qsfp_tx         : out   std_logic_vector(3 downto 0);
    qsfp_rx         : in    std_logic_vector(3 downto 0);



    -- POD

    -- si5345 out0 (125 MHz)
    pod_pll_clk     : in    std_logic;

    pod_tx_reset_n  : out   std_logic;
    pod_rx_reset_n  : out   std_logic;

    pod_tx          : out   std_logic_vector(3 downto 0);
    pod_rx          : in    std_logic_vector(3 downto 0);



    -- MSCB

    mscb_data_in    : in    std_logic;
    mscb_data_out   : out   std_logic;
    mscb_oe         : out   std_logic;



    --

    led_n       : out   std_logic_vector(15 downto 0);

    PushButton  : in    std_logic_vector(1 downto 0);



    -- si5345 out8 (625 MHz)
    clk_625     : in    std_logic;



    si42_clk_40 : in    std_logic;
    si42_clk_80 : in    std_logic;

    reset_n     : in    std_logic;

    -- 125 MHz
    clk_aux     : in    std_logic--;
);
end entity;

architecture arch of top is

    signal fifo_rempty : std_logic;
    signal fifo_rack : std_logic;
    signal fifo_rdata : std_logic_vector(35 downto 0);

    signal malibu_reg, scifi_reg, mupix_reg : work.util.rw_t;

    signal led : std_logic_vector(led_n'range) := (others => '0');

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(15 downto 0);

begin

    ----------------------------------------------------------------------------
    -- MALIBU

    malibu_ck_fpga_1 <= clk_625;
    malibu_pll_reset <= '0';

    e_malibu_block : entity work.malibu_block
    generic map (
        N_g => 1--,
    )
    port map (
        i_reg_addr      => malibu_reg.addr(3 downto 0),
        i_reg_re        => malibu_reg.re,
        o_reg_rdata     => malibu_reg.rdata,
        i_reg_we        => malibu_reg.we,
        i_reg_wdata     => malibu_reg.wdata,

        o_ck_fpga_0     => malibu_ck_fpga_0,
        o_chip_reset    => malibu_chip_reset,
        o_pll_test      => malibu_pll_test,
        i_data          => malibu_data(0 downto 0),

        o_fifo_rempty   => fifo_rempty,
        i_fifo_rack     => fifo_rack,
        o_fifo_rdata    => fifo_rdata,

        i_reset         => not reset_n,
        i_clk           => qsfp_pll_clk--,
    );

    ----------------------------------------------------------------------------



    led_n <= not led;

    -- enable Si5342
    si42_oe_n <= '0';
    si42_rst_n <= '1';

    -- enable Si5345
    si45_oe_n <= '0';
    si45_rst_n <= '1';

    -- enable QSFP
    QSFP_ModSel_n <= '1';
    QSFP_Rst_n <= '1';
    QSFP_LPM <= '0';

    -- enable PID
    pod_tx_reset_n <= '1';
    pod_rx_reset_n <= '1';



    ----------------------------------------------------------------------------
    -- I2C

    i2c_scl <= not i2c_scl_oe;
    i2c_sda <=
        malibu_i2c_sda and
        '1';
    malibu_i2c_scl <= ZERO when i2c_scl_oe = '1' else 'Z';
    malibu_i2c_sda <= ZERO when i2c_sda_oe = '1' else 'Z';

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- SPI

    malibu_spi_sdi <= spi_mosi;
    malibu_spi_sck <= spi_sclk when spi_ss_n(1) = '0' else '0';

    spi_miso <=
        malibu_spi_sdo when spi_ss_n(1) = '0' else
        '0';

    ----------------------------------------------------------------------------



    e_fe_block : entity work.fe_block
    generic map (
        FPGA_ID_g => X"FEB0",
        -- mutrig FEB type
        FEB_type_in => "111000"--,
    )
    port map (
        i_i2c_scl       => i2c_scl,
        o_i2c_scl_oe    => i2c_scl_oe,
        i_i2c_sda       => i2c_sda,
        o_i2c_sda_oe    => i2c_sda_oe,

        i_spi_miso      => spi_miso,
        o_spi_mosi      => spi_mosi,
        o_spi_sclk      => spi_sclk,
        o_spi_ss_n      => spi_ss_n,

        i_spi_si_miso   => si45_spi_out,
        o_spi_si_mosi   => si45_spi_in,
        o_spi_si_sclk   => si45_spi_sclk,
        o_spi_si_ss_n   => si45_spi_cs_n,

        i_qsfp_rx       => qsfp_rx,
        o_qsfp_tx       => qsfp_tx,

        i_pod_rx        => pod_rx,
        o_pod_tx        => pod_tx,

        i_fifo_rempty   => fifo_rempty,
        o_fifo_rack     => fifo_rack,
        i_fifo_rdata    => fifo_rdata,

        i_mscb_data     => mscb_data_in,
        o_mscb_data     => mscb_data_out,
        o_mscb_oe       => mscb_oe,

        o_malibu_reg_addr   => malibu_reg.addr(7 downto 0),
        o_malibu_reg_re     => malibu_reg.re,
        i_malibu_reg_rdata  => malibu_reg.rdata,
        o_malibu_reg_we     => malibu_reg.we,
        o_malibu_reg_wdata  => malibu_reg.wdata,

        i_nios_clk_startup => clk_aux,
        i_nios_clk_main => clk_aux,
        o_nios_clk_mon  => led(15),
        i_clk_156       => qsfp_pll_clk,
        o_clk_156_mon   => led(14),
        i_clk_125       => pod_pll_clk,
        o_clk_125_mon   => led(13),

        i_areset_n      => reset_n--,
    );

end architecture;
