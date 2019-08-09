library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    -- FE.A
    -- chip dacs
    CTRL_SDO_A



    -- SI45

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



    reset_n     : in    std_logic;

    -- 125 MHz
    clk_aux     : in    std_logic--;
);
end entity;

architecture arch of top is

    signal fifo_rempty : std_logic;
    signal fifo_rack : std_logic;
    signal fifo_rdata : std_logic_vector(35 downto 0);

    signal sc_reg_addr : std_logic_vector(7 downto 0);
    signal sc_reg_re : std_logic;
    signal sc_reg_rdata : std_logic_vector(31 downto 0);
    signal sc_reg_we : std_logic;
    signal sc_reg_wdata : std_logic_vector(31 downto 0);

    signal led : std_logic_vector(led_n'range) := (others => '0');

    signal nios_clk, nios_reset_n : std_logic;
    signal qsfp_reset_n : std_logic;

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(15 downto 0);

begin

    ----------------------------------------------------------------------------
    -- MUPIX

    e_mupix_block : entity work.mupix_block
    generic map( NCHIPS => 1);
    port map(
   
        -- chip dacs
        i_CTRL_SDO_A            => ,
        o_CTRL_SDI_A            => ,
        o_CTRL_SCK1_A           => ,
        o_CTRL_SCK2_A           => ,
        o_CTRL_Load_A           => ,
        o_CTRL_RB_A             => ,
        i_data_chip_dacs        => ,
        o_add_chip_dacs         => ,
        
        -- board dacs
        i_SPI_DOUT_ADC_0_A      => ,
        o_SPI_DIN0_A            => ,
        o_SPI_CLK_A             => ,
        o_SPI_LD_ADC_A          => ,
        o_SPI_LD_TEMP_DAC_A     => ,
        o_SPI_LD_DAC_A          => ,
        o_add_board_dacs        => ,
        i_data_board_dacs       => ,
        o_data_board_dacs       => ,
        o_wen_data_board_dacs   => ,
        
        i_ckdiv                 => (others => '0'),

        i_reset                 => not reset_n,
        -- 156.25 MHz
        i_clk                   => qsfp_pll_clk,
        i_clk125                => clk_aux,
    );

    ----------------------------------------------------------------------------



    led_n <= not led;

    si45_oe_n <= '0';
    si45_rst_n <= '1';

    QSFP_ModSel_n <= '1';
    QSFP_Rst_n <= '1';
    QSFP_LPM <= '0';

    pod_tx_reset_n <= '1';
    pod_rx_reset_n <= '1';



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

    nios_clk <= clk_aux;

    e_nios_reset_n : entity work.reset_sync
    port map ( rstout_n => nios_reset_n, arst_n => reset_n, clk => nios_clk );

    e_qsfp_reset_n : entity work.reset_sync
    port map ( rstout_n => qsfp_reset_n, arst_n => reset_n, clk => qsfp_pll_clk );



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

    si45_spi_in <= spi_mosi;
    si45_spi_sclk <= spi_sclk when spi_ss_n(0) = '0' else '0';
    si45_spi_cs_n <= spi_ss_n(0);

    malibu_spi_sdi <= spi_mosi;
    malibu_spi_sck <= spi_sclk when spi_ss_n(1) = '0' else '0';

    spi_miso <=
        si45_spi_out when spi_ss_n(0) = '0' else
        malibu_spi_sdo when spi_ss_n(1) = '0' else
        '0';

    ----------------------------------------------------------------------------



    e_fe_block : entity work.fe_block
    generic map (
        FPGA_ID_g => X"FEB0"--,
    )
    port map (
        i_nios_clk      => nios_clk,
        i_nios_reset_n  => nios_reset_n,

        i_i2c_scl       => i2c_scl,
        o_i2c_scl_oe    => i2c_scl_oe,
        i_i2c_sda       => i2c_sda,
        o_i2c_sda_oe    => i2c_sda_oe,

        i_spi_miso      => spi_miso,
        o_spi_mosi      => spi_mosi,
        o_spi_sclk      => spi_sclk,
        o_spi_ss_n      => spi_ss_n,

        i_mscb_data     => mscb_data_in,
        o_mscb_data     => mscb_data_out,
        o_mscb_oe       => mscb_oe,

        i_qsfp_rx       => qsfp_rx,
        o_qsfp_tx       => qsfp_tx,
        i_qsfp_refclk   => qsfp_pll_clk,

        i_fifo_rempty   => fifo_rempty,
        o_fifo_rack     => fifo_rack,
        i_fifo_rdata    => fifo_rdata,

        i_pod_rx        => pod_rx,
        o_pod_tx        => pod_tx,
        i_pod_refclk    => pod_pll_clk,

        o_sc_reg_addr   => sc_reg_addr,
        o_sc_reg_re     => sc_reg_re,
        i_sc_reg_rdata  => sc_reg_rdata,
        o_sc_reg_we     => sc_reg_we,
        o_sc_reg_wdata  => sc_reg_wdata,

        i_reset_n       => qsfp_reset_n,
        i_clk           => qsfp_pll_clk--,
    );

end architecture;
