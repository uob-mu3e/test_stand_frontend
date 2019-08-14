library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

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

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal led : std_logic_vector(led_n'range) := (others => '0');

    signal nios_clk, nios_reset_n : std_logic;
    signal nios_pio : std_logic_vector(31 downto 0);

    --i2c interface (external, not used)
    signal i2c_scl_in, i2c_scl_oe, i2c_sda_in, i2c_sda_oe : std_logic;
    --spi interface (external, spi_ss_n[4*N_SCIFI_BOARDS] is rewired to siXX45 chip, miso is also rewired if corresponding cs is low)
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(4*1 downto 0);

    signal s_fee_chip_rst_auxclk_sync : std_logic_vector(1 downto 0);
    signal s_fee_chip_rst_niosclk : std_logic;



    signal fifo_data : std_logic_vector(35 downto 0);
    signal fifo_data_empty, fifo_data_read : std_logic;



    signal av_pod, av_qsfp : work.util.avalon_t;

    signal qsfp_tx_data : std_logic_vector(127 downto 0) :=
          X"03CAFE" & work.util.D28_5
        & X"02BABE" & work.util.D28_5
        & X"01DEAD" & work.util.D28_5
        & X"00BEEF" & work.util.D28_5;

    signal qsfp_tx_datak : std_logic_vector(15 downto 0) :=
          "0001"
        & "0001"
        & "0001"
        & "0001";

    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);

    signal qsfp_reset_n : std_logic;



    signal av_sc : work.util.avalon_t;

    signal mscb_to_nios_parallel_in : std_logic_vector(11 downto 0);
    signal mscb_from_nios_parallel_out : std_logic_vector(11 downto 0);
    signal mscb_counter_in : unsigned(15 downto 0);

    signal reset_bypass : std_logic_vector(11 downto 0);

    signal pod_rx_data : std_logic_vector(7 downto 0);

    signal run_state : feb_run_state;
    signal terminated : std_logic;

    signal av_test : work.util.avalon_t;

begin

    led_n <= not led;

    -- 125 MHz --> 1Hz
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
        avm_qsfp_address        => av_qsfp.address(13 downto 0),
        avm_qsfp_read           => av_qsfp.read,
        avm_qsfp_readdata       => av_qsfp.readdata,
        avm_qsfp_write          => av_qsfp.write,
        avm_qsfp_writedata      => av_qsfp.writedata,
        avm_qsfp_waitrequest    => av_qsfp.waitrequest,

        avm_pod_address         => av_pod.address(13 downto 0),
        avm_pod_read            => av_pod.read,
        avm_pod_readdata        => av_pod.readdata,
        avm_pod_write           => av_pod.write,
        avm_pod_writedata       => av_pod.writedata,
        avm_pod_waitrequest     => av_pod.waitrequest,

        avm_sc_address          => av_sc.address(15 downto 0),
        avm_sc_read             => av_sc.read,
        avm_sc_readdata         => av_sc.readdata,
        avm_sc_write            => av_sc.write,
        avm_sc_writedata        => av_sc.writedata,
        avm_sc_waitrequest      => av_sc.waitrequest,

        avm_test_address        => av_test.address(13 downto 0),
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

        --I2C interface is not connected to outside world (currently unused)
        i2c_scl_in => i2c_scl_in,
        i2c_scl_oe => i2c_scl_oe,
        i2c_sda_in => i2c_sda_in,
        i2c_sda_oe => i2c_sda_oe,

        --SPI interface connected to ASICs and SI clock chip
        spi_miso => spi_miso,
        spi_mosi => spi_mosi,
        spi_sclk => spi_sclk,
        spi_ss_n => spi_ss_n,

        pio_export => nios_pio,

        -- mscb
        parallel_mscb_in_export => mscb_to_nios_parallel_in,
        parallel_mscb_out_export => mscb_from_nios_parallel_out,
        counter_in_export => std_logic_vector(mscb_counter_in),

        -- reset bypass
        reset_bypass_out_export => reset_bypass,

        rst_reset_n => nios_reset_n,
        clk_clk => nios_clk--,
    );

    --si chip assignments
    si45_oe_n <= '0';
    si45_rst_n <= '1';
    --fee assignments
    o_fee_ext_trig <= (others =>'0');

    -- SPI
    ----------------------------------------------------------------------------
    --si chip assignments
    si45_spi_in <= spi_mosi;
    si45_spi_sclk <= spi_sclk;
    si45_spi_cs_n <= spi_ss_n(4);
    --fee assignments
    o_fee_spi_MOSI <= (others => spi_mosi);
    o_fee_spi_SCK  <= (others => spi_sclk);
    o_fee_spi_CSn <=  spi_ss_n(4-1 downto 0);
    --MISO: multiplexing si chip / SciFi FEE
    spi_miso <= si45_spi_out when spi_ss_n(4) = '0' else
        i_fee_spi_MISO(0); --TODO make working with multiple FEBs, if we need this

    led(4 downto 1)<=spi_ss_n(3 downto 0);

    -- I2C (currently unused, simulating empty bus)
    ----------------------------------------------------------------------------
    i2c_scl_in <= not i2c_scl_oe;
    i2c_sda_in <= not i2c_sda_oe;
    --i2c_scl_in <= not i2c_scl_oe;
    --i2c_sda_in <= io_fee_i2c_sda
    --io_fee_i2c_scl <= ZERO when i2c_scl_oe = '1' else 'Z';
    --io_fee_i2c_sda <= ZERO when i2c_sda_oe = '1' else 'Z';



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

    ----------------------------------------------------------------------------
    -- SciFi FE board


    e_scifi_path : entity work.scifi_path
    generic map (
        N_g => 4
    )
    port map (
        i_avs_address       => av_test.address(3 downto 0),
        i_avs_read          => av_test.read,
        o_avs_readdata      => av_test.readdata,
        i_avs_write         => av_test.write,
        i_avs_writedata     => av_test.writedata,
        o_avs_waitrequest   => av_test.waitrequest,

        o_ck_fpga_0         => open,
        o_chip_reset        => s_fee_chip_rst_niosclk,
        o_pll_test          => open,
        i_data              => i_fee_rxd(3 downto 0),

        o_fifo_data         => fifo_data,
        o_fifo_empty        => fifo_data_empty,
        i_fifo_rack         => fifo_data_read,

        i_reset             => not reset_n,
        i_clk               => qsfp_pll_clk--,
    );
    led(0)<=s_fee_chip_rst_niosclk;

    ----------------------------------------------------------------------------



    e_data_sc_path : entity work.data_sc_path
    port map (
        i_avs_address       => av_sc.address(15 downto 0),
        i_avs_read          => av_sc.read,
        o_avs_readdata      => av_sc.readdata,
        i_avs_write         => av_sc.write,
        i_avs_writedata     => av_sc.writedata,
        o_avs_waitrequest   => av_sc.waitrequest,

        i_fifo_data         => fifo_data,
        i_fifo_data_empty   => fifo_data_empty,
        o_fifo_data_read    => fifo_data_read,

        i_link_data         => qsfp_rx_data(31 downto 0),
        i_link_datak        => qsfp_rx_datak(3 downto 0),

        o_link_data         => qsfp_tx_data(31 downto 0),
        o_link_datak        => qsfp_tx_datak(3 downto 0),

        o_terminated        => terminated,
        i_run_state         => run_state,

        i_reset             => not reset_n,
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
    -- reset system

    e_reset_sys : entity work.resetsys
    port map (
        clk_reset_rx    => pod_pll_clk,
        clk_global      => clk_aux,
        clk_free        => clk_aux,
        reset_in        => not PushButton(0),
        resets_out      => open,
        phase_out       => open,
        data_in         => pod_rx_data,
        reset_bypass    => reset_bypass,
        state_out       => run_state,
        run_number_out  => open,
        fpga_id         => x"FEB0",
        terminated      => terminated,
        testout         => open--,
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

        i_avs_address     => av_qsfp.address(13 downto 0),
        i_avs_read        => av_qsfp.read,
        o_avs_readdata    => av_qsfp.readdata,
        i_avs_write       => av_qsfp.write,
        i_avs_writedata   => av_qsfp.writedata,
        o_avs_waitrequest => av_qsfp.waitrequest,

        i_reset     => not nios_reset_n,
        i_clk       => nios_clk--,
    );

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- POD
    -- (reset system)

    pod_tx_reset_n <= '1';
    pod_rx_reset_n <= '1';

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
        i_avs_address     => av_pod.address(13 downto 0),
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
