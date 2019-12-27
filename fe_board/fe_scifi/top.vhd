library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;

entity top is
port (
    -- FE.Ports
    i_fee_rxd       : in    std_logic_vector(2*4 - 1 downto 0); -- data inputs from ASICs
    o_fee_spi_CSn   : out   std_logic_vector(2*4 - 1 downto 0); -- CSn signals to ASICs (one per ASIC)
    o_fee_spi_MOSI  : out   std_logic_vector(2 - 1 downto 0);   -- MOSI signals to ASICs (one per board)
    i_fee_spi_MISO  : in    std_logic_vector(2 - 1 downto 0);   -- MISO signals from ASICs (one per board)
    o_fee_spi_SCK   : out   std_logic_vector(2 - 1 downto 0);   -- SCK signals to ASICs (one per board)

    o_fee_ext_trig  : out   std_logic_vector(2 - 1 downto 0);   -- external trigger (data validation) signals to ASICs (one per board)
    o_fee_chip_rst  : out   std_logic_vector(2 - 1 downto 0);   -- chip reset signals to ASICs (one per board)



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
    FPGA_Test   : out   std_logic_vector(7 downto 0);
    PushButton  : in    std_logic_vector(1 downto 0);



    -- Si5345 out0 (125 MHz)
    si42_clk_125        : in    std_logic;
    -- Si5345 out1 (50 MHz)
    si42_clk_50         : in    std_logic;

    lvds_clk_A      : in    std_logic; -- 125 MHz base clock for LVDS PLLs - right  // SI5345 OUT3
    lvds_clk_B      : in    std_logic; -- 125 MHz base clock for LVDS PLLs - left   // SI5345 OUT6



    clk_aux     : in    std_logic;

    reset_n     : in    std_logic--;
);
end entity;

architecture arch of top is

    signal fifoA_rempty, fifoB_rempty : std_logic;
    signal fifoA_rack,   fifoB_rack   : std_logic;
    signal fifoA_rdata,  fifoB_rdata  : std_logic_vector(35 downto 0);

    signal malibu_reg, scifi_reg, mupix_reg : work.util.rw_t;

    signal led : std_logic_vector(led_n'range) := (others => '0');

    signal nios_clk : std_logic;

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    -- i2c interface (external, not used)
    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    -- spi interface (external, spi_ss_n[4*N_SCIFI_BOARDS] is rewired to siXX45 chip, miso is also rewired if corresponding cs is low)
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(15 downto 0);

    signal s_fee_chip_rst : std_logic_vector(1 downto 0);
    signal s_FPGA_test : std_logic_vector (7 downto 0) := (others => '0');
    signal s_MON_rxrdy : std_logic_vector (7 downto 0);

    -- reset system
    signal run_state_125 : run_state_t;
    signal s_run_state_all_done : std_logic;

begin

    ----------------------------------------------------------------------------



    --fee assignments
    o_fee_ext_trig <= (others =>'0');
    o_fee_chip_rst <= s_fee_chip_rst;

    ----------------------------------------------------------------------------
    -- SciFi FE board

    e_scifi_path : entity work.scifi_path
    generic map (
        N_m => 2
    )
    port map (
        i_reg_addr      => scifi_reg.addr(3 downto 0),
        i_reg_re        => scifi_reg.re,
        o_reg_rdata     => scifi_reg.rdata,
        i_reg_we        => scifi_reg.we,
        i_reg_wdata     => scifi_reg.wdata,

        o_chip_reset    => s_fee_chip_rst,
        o_pll_test      => open,
        i_data          => i_fee_rxd,

        o_fifoA_rempty   => fifoA_rempty,
        i_fifoA_rack     => fifoA_rack,
        o_fifoA_rdata    => fifoA_rdata,

        o_fifoB_rempty   => fifoB_rempty,
        i_fifoB_rack     => fifoB_rack,
        o_fifoB_rdata    => fifoB_rdata,

        i_reset         => not reset_n,
        i_clk_core      => qsfp_pll_clk,
        i_clk_g125      => clk_125_bottom,
        i_clk_ref_A     => lvds_clk_A,
        i_clk_ref_B     => lvds_clk_B,

        i_run_state     => run_state_125,

        o_run_state_all_done => s_run_state_all_done,

        o_MON_rxrdy     => s_MON_rxrdy
    );

    ----------------------------------------------------------------------------



    -- LED maps:
    -- 15: si42_clk_50 (80MHz -> 1Hz)
    -- 14: clk_qsfp (156MHz -> 1Hz)
    -- 13: clk_pod (125MHz -> 1Hz)
    -- 11: fee_chip_reset (niosclk)
    -- x..0 : CSn to SciFi boards

    led(11) <= s_fee_chip_rst(0);
    --led(7 downto 0) <= s_MON_rxrdy;

    -- test outputs
    FPGA_Test <= s_FPGA_test;
    --s_FPGA_test(2 downto 0) <= s_fee_chip_rst(2 downto 0);
    led(8 downto 0) <= run_state_125(8 downto 0);
    s_FPGA_test(4 downto 0) <= run_state_125(4 downto 0); 



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
    -- I2C (currently unused, simulating empty bus)

    i2c_scl <= not i2c_scl_oe;
    i2c_sda <= not i2c_sda_oe;
    --i2c_scl_in <= not i2c_scl_oe;
    --i2c_sda_in <= io_fee_i2c_sda
    --io_fee_i2c_scl <= ZERO when i2c_scl_oe = '1' else 'Z';
    --io_fee_i2c_sda <= ZERO when i2c_sda_oe = '1' else 'Z';

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- SPI
    o_fee_spi_MOSI <= (others => spi_mosi);
    o_fee_spi_SCK  <= (others => spi_sclk);
    o_fee_spi_CSn <=  spi_ss_n(o_fee_spi_CSn'range);

    spi_miso <=
        si45_spi_out when spi_ss_n(0) = '0' else
        i_fee_spi_MISO(0) when spi_ss_n(3 downto 0)/="1111" else
        i_fee_spi_MISO(1) when spi_ss_n(7 downto 4)/="1111" else
--        i_fee_spi_MISO(2) when spi_ss_n(11 downto 8)/="1111" else
--        i_fee_spi_MISO(3) when spi_ss_n(15 downto 12)/="1111" else
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

        i_fifo_rempty   => fifoA_rempty,
        o_fifo_rack     => fifoA_rack,
        i_fifo_rdata    => fifoA_rdata,

        i_secondary_fifo_rempty   => fifoB_rempty,
        o_secondary_fifo_rack     => fifoB_rack,
        i_secondary_fifo_rdata    => fifoB_rdata,

        i_mscb_data     => mscb_data_in,
        o_mscb_data     => mscb_data_out,
        o_mscb_oe       => mscb_oe,

        o_scifi_reg_addr    => scifi_reg.addr(7 downto 0),
        o_scifi_reg_re      => scifi_reg.re,
        i_scifi_reg_rdata   => scifi_reg.rdata,
        o_scifi_reg_we      => scifi_reg.we,
        o_scifi_reg_wdata   => scifi_reg.wdata,



        -- reset system
        o_run_state_125 => run_state_125,
        i_can_terminate => s_run_state_all_done,



        i_nios_clk      => si42_clk_50,
        o_nios_clk_mon  => led(15),
        i_clk_156       => qsfp_pll_clk,
        o_clk_156_mon   => led(14),
        i_clk_125       => pod_pll_clk,
        o_clk_125_mon   => led(13),

        i_areset_n      => reset_n--,
    );

end architecture;
