library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

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



    -- POD

    -- Si5345 out0 (125 MHz)
    pod_clk_left        : in    std_logic;
    -- Si5345 out1 (125 MHz)
--    pod_clk_right       : in    std_logic;

    pod_tx_reset_n  : out   std_logic;
    pod_rx_reset_n  : out   std_logic;

    pod_tx          : out   std_logic_vector(3 downto 0);
    pod_rx          : in    std_logic_vector(3 downto 0);



    -- QSFP

    -- Si5345 out2 (156.25 MHz)
    qsfp_clk        : in    std_logic;

    QSFP_ModSel_n   : out   std_logic; -- module select (i2c)
    QSFP_Rst_n      : out   std_logic;
    QSFP_LPM        : out   std_logic; -- Low Power Mode

    qsfp_tx         : out   std_logic_vector(3 downto 0);
    qsfp_rx         : in    std_logic_vector(3 downto 0);



    -- Si5345 out3 (125 MHz, right)
    lvds_clk_A          : in    std_logic;
    -- Si5345 out6 (125 MHz, left)
    lvds_clk_B          : in    std_logic;

    -- Si5345 out7 (125 MHz)
    clk_125_bottom      : in    std_logic; -- global 125 MHz clock
    -- Si5345 out8 (125 MHz)
    clk_125_top         : in    std_logic;



    -- MSCB

    mscb_data_in    : in    std_logic;
    mscb_data_out   : out   std_logic;
    mscb_oe         : out   std_logic;



    --

    led_n       : out   std_logic_vector(15 downto 0);

    PushButton  : in    std_logic_vector(1 downto 0);



    -- Si5345 out0 (125 MHz)
    si42_clk_125        : in    std_logic;
    -- Si5345 out1 (50 MHz)
    si42_clk_50         : in    std_logic;



    clk_aux     : in    std_logic;

    reset_n     : in    std_logic--;
);
end entity;

architecture arch of top is

    signal led : std_logic_vector(led_n'range) := (others => '0');

    signal fifo_rempty : std_logic;
    signal fifo_rack : std_logic;
    signal fifo_rdata : std_logic_vector(35 downto 0);

    signal malibu_reg, scifi_reg, mupix_reg : work.util.rw_t;

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(15 downto 0);

    signal run_state_125 : run_state_t;
    signal s_run_state_all_done : std_logic;

begin

    ----------------------------------------------------------------------------
    -- MALIBU

    -- lvds clock (156.25 MHz)
    malibu_ck_fpga_0 <= qsfp_clk;
    -- timestamp clock (625 MHz, conf from nios)
    malibu_ck_fpga_1 <= lvds_clk_A;

    malibu_pll_reset <= '0';

    e_malibu_block : entity work.scifi_path
    generic map (
        N_MODULES => 1,
        N_ASICS => 1,
        LVDS_PLL_FREQ => 156.25,
        LVDS_DATA_RATE => 156.25--,
    )
    port map (
        i_reg_addr      => malibu_reg.addr(3 downto 0),
        i_reg_re        => malibu_reg.re,
        o_reg_rdata     => malibu_reg.rdata,
        i_reg_we        => malibu_reg.we,
        i_reg_wdata     => malibu_reg.wdata,

        o_chip_reset(0) => malibu_chip_reset,
        o_pll_test      => malibu_pll_test,
        i_data          => malibu_data(0 downto 0),

        o_fifoA_rempty  => fifo_rempty,
        i_fifoA_rack    => fifo_rack,
        o_fifoA_rdata   => fifo_rdata,

        o_fifoB_rempty  => open,
        i_fifoB_rack    => '0',
        o_fifoB_rdata   => open,

        i_run_state     => run_state_125,
        o_run_state_all_done => s_run_state_all_done,

        o_MON_rxrdy     => open,

        i_clk_core      => qsfp_clk,
        i_clk_g125      => clk_125_bottom,
        i_clk_ref_A     => qsfp_clk,
        i_clk_ref_B     => qsfp_clk,

        i_reset         => not reset_n--,
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

    -- enable POD
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
        NIOS_CLK_MHZ_g => 50.0--,
    )
    port map (
        i_fpga_id       => X"FEB0",
        -- mutrig FEB type
        i_fpga_type     => "111000",

        i_i2c_scl       => i2c_scl,
        o_i2c_scl_oe    => i2c_scl_oe,
        i_i2c_sda       => i2c_sda,
        o_i2c_sda_oe    => i2c_sda_oe,

        i_spi_miso      => spi_miso,
        o_spi_mosi      => spi_mosi,
        o_spi_sclk      => spi_sclk,
        o_spi_ss_n      => spi_ss_n,

        i_spi_si_miso(1)    => si42_spi_out,
        o_spi_si_mosi(1)    => si42_spi_in,
        o_spi_si_sclk(1)    => si42_spi_sclk,
        o_spi_si_ss_n(1)    => si42_spi_cs_n,
        i_spi_si_miso(0)    => si45_spi_out,
        o_spi_si_mosi(0)    => si45_spi_in,
        o_spi_si_sclk(0)    => si45_spi_sclk,
        o_spi_si_ss_n(0)    => si45_spi_cs_n,

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

        -- reset system
        o_run_state_125 => run_state_125,
        i_can_terminate => s_run_state_all_done,

        -- clocks
        i_nios_clk      => si42_clk_50,
        o_nios_clk_mon  => led(15),
        i_clk_156       => qsfp_clk,
        o_clk_156_mon   => led(14),
        i_clk_125       => pod_clk_left,
        o_clk_125_mon   => led(13),

        i_areset_n      => reset_n--,
    );

end architecture;
