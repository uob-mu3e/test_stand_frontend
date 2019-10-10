library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    -- FE.Ports
    i_fee_rxd		: in  std_logic_vector (4*1 - 1 downto 0); --data inputs from ASICs
    o_fee_spi_CSn	: out std_logic_vector (4*1 - 1 downto 0); --CSn signals to ASICs (one per ASIC)
    o_fee_spi_MOSI	: out std_logic_vector (1 - 1 downto 0);   --MOSI signals to ASICs (one per board)
    i_fee_spi_MISO	: in  std_logic_vector (1 - 1 downto 0);   --MISO signals from ASICs (one per board)
    o_fee_spi_SCK	: out std_logic_vector (1 - 1 downto 0);   --SCK signals to ASICs (one per board)

    o_fee_ext_trig	: out std_logic_vector (1 - 1 downto 0);   --external trigger (data validation) signals to ASICs (one per board)
    o_fee_chip_rst	: out std_logic_vector (1 - 1 downto 0);   --chip reset signals to ASICs (one per board)



    -- SI5345

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

    signal sc_reg : work.util.rw_t;
    signal malibu_reg : work.util.rw_t;
    signal scifi_reg : work.util.rw_t;

    signal led : std_logic_vector(led_n'range) := (others => '0');

    signal nios_clk, nios_reset_n : std_logic;
    signal qsfp_reset_n : std_logic;

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    -- i2c interface (external, not used)
    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    -- spi interface (external, spi_ss_n[4*N_SCIFI_BOARDS] is rewired to siXX45 chip, miso is also rewired if corresponding cs is low)
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(15 downto 0);

    signal s_fee_chip_rst_auxclk_sync : std_logic_vector(1 downto 0);
    signal s_fee_chip_rst_niosclk : std_logic;

begin

    -- malibu regs : 0x40-0x4F
    malibu_reg.addr <= sc_reg.addr;
    malibu_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 4) = X"4" ) else '0';
    malibu_reg.we <= sc_reg.we when ( sc_reg.addr(7 downto 4) = X"4" ) else '0';
    malibu_reg.wdata <= sc_reg.wdata;

    -- scifi regs : 0x60-0x6F
    scifi_reg.addr <= sc_reg.addr;
    scifi_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 4) = X"6" ) else '0';
    scifi_reg.we <= sc_reg.we when ( sc_reg.addr(7 downto 4) = X"6" ) else '0';
    scifi_reg.wdata <= sc_reg.wdata;

    -- select valid rdata
    sc_reg.rdata <=
        malibu_reg.rdata when ( malibu_reg.rvalid = '1' ) else
        scifi_reg.rdata when ( scifi_reg.rvalid = '1' ) else
        X"CCCCCCCC";

    process(qsfp_pll_clk)
    begin
    if rising_edge(qsfp_pll_clk) then
        malibu_reg.rdata <= X"CCCCCCCC";
        malibu_reg.rvalid <= malibu_reg.re;
--        scifi_reg.rdata <= X"CCCCCCCC";
        scifi_reg.rvalid <= scifi_reg.re;
    end if;
    end process;



    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    --generation of reset signal synchronized to aux clock (125MHz, nios as source is running at 156MHz).
    --using simple two-ff synchronizer, assuming the reset pulse is longer than 2 cc in the nios domain
    p_fee_reset_sync: process(clk_aux)
    begin
        if rising_edge(clk_aux) then
            s_fee_chip_rst_auxclk_sync <= s_fee_chip_rst_auxclk_sync(0) & s_fee_chip_rst_niosclk;
        end if;
    end process;
    o_fee_chip_rst <= ( others => s_fee_chip_rst_auxclk_sync(1) );

    --fee assignments
    o_fee_ext_trig <= (others =>'0');

    ----------------------------------------------------------------------------
    -- SciFi FE board

    e_scifi_path : entity work.scifi_path
    generic map (
        N_g => 4
    )
    port map (
        i_reg_addr      => scifi_reg.addr(3 downto 0),
        i_reg_re        => scifi_reg.re,
        o_reg_rdata     => scifi_reg.rdata,
        i_reg_we        => scifi_reg.we,
        i_reg_wdata     => scifi_reg.wdata,

        o_ck_fpga_0     => open,
        o_chip_reset    => s_fee_chip_rst_niosclk,
        o_pll_test      => open,
        i_data          => i_fee_rxd,

        o_fifo_rempty   => fifo_rempty,
        i_fifo_rack     => fifo_rack,
        o_fifo_rdata    => fifo_rdata,

        i_reset         => not reset_n,
        i_clk           => qsfp_pll_clk--,
    );

    ----------------------------------------------------------------------------

--LED maps:
-- 15: clk_aux  (125M -> 1Hz)
-- 14: clk_qsfp (156M -> 1Hz)
-- 13: clk_pod  (125M -> 1Hz)
-- 12: fee_chip_reset (niosclk)
-- x..0 : CSn to SciFi boards

    led(12) <= s_fee_chip_rst_niosclk;


    led_n <= not led;

    -- enable SI5345
    si45_oe_n <= '0';
    si45_rst_n <= '1';

    -- enable QSFP
    QSFP_ModSel_n <= '1';
    QSFP_Rst_n <= '1';
    QSFP_LPM <= '0';

    -- enable PID
    pod_tx_reset_n <= '1';
    pod_rx_reset_n <= '1';



    -- 125 MHz -> 1 Hz
    e_clk_aux_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(15), rst_n => reset_n, clk => clk_aux );

    -- 156.25 MHz -> 1 Hz
    e_clk_qsfp_hz : entity work.clkdiv
    generic map ( P => 156250000 )
    port map ( clkout => led(14), rst_n => reset_n, clk => qsfp_pll_clk );

    -- 125 MHz -> 1 Hz
    e_clk_pod_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(13), rst_n => reset_n, clk => pod_pll_clk );

    nios_clk <= clk_aux;

    e_nios_reset_n : entity work.reset_sync
    port map ( rstout_n => nios_reset_n, arst_n => reset_n, clk => nios_clk );

    e_qsfp_reset_n : entity work.reset_sync
    port map ( rstout_n => qsfp_reset_n, arst_n => reset_n, clk => qsfp_pll_clk );



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

    -- si chip assignments
    si45_spi_in <= spi_mosi;
    si45_spi_sclk <= spi_sclk when spi_ss_n(0) = '0' else '0';
    si45_spi_cs_n <= spi_ss_n(0);

    -- fee assignments
    o_fee_spi_MOSI <= (others => spi_mosi);
    o_fee_spi_SCK  <= (others => spi_sclk);
    o_fee_spi_CSn <=  spi_ss_n(o_fee_spi_CSn'length downto 1);

    -- MISO: multiplexing si chip / SciFi FEE
    spi_miso <=
        si45_spi_out when spi_ss_n(0) = '0' else
	i_fee_spi_MISO(0) when spi_ss_n(4 downto 1)/="1111" else
	--i_fee_spi_MISO(1) when spi_ss_n(8 downto 5)/="1111" else
	'0';
    --connect CSn lines to FEs to LEDs
    led(o_fee_spi_CSn'range)<=spi_ss_n(o_fee_spi_CSn'high+1 downto 1);

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

        o_sc_reg_addr   => sc_reg.addr(7 downto 0),
        o_sc_reg_re     => sc_reg.re,
        i_sc_reg_rdata  => sc_reg.rdata,
        o_sc_reg_we     => sc_reg.we,
        o_sc_reg_wdata  => sc_reg.wdata,

        i_reset_n       => qsfp_reset_n,
        i_clk           => qsfp_pll_clk--,
    );

end architecture;
