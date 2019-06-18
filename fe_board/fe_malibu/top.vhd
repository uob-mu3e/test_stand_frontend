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

    qsfp_pll_clk    : in    std_logic; -- 125 MHz for transceiver PLLs - QSFP

    QSFP_ModSel_n   : out   std_logic; -- module select (i2c)
    QSFP_Rst_n      : out   std_logic;
    QSFP_LPM        : out   std_logic; -- Low Power Mode

    qsfp_tx         : out   std_logic_vector(3 downto 0);
    qsfp_rx         : in    std_logic_vector(3 downto 0);

    -- POD

--    pod_pll_clk     : in    std_logic;
--
--    pod_tx_reset    : out   std_logic;
    pod_rx_reset    : out   std_logic;
--
--    pod_tx          : out   std_logic_vector(3 downto 0);
--    pod_rx          : in    std_logic_vector(3 downto 0);

	 pod_rx			  : in std_logic;

    -- mscb
	 mscb_data_in	  : in 	 std_logic;
	 mscb_data_out   : out 	 std_logic;
	 mscb_oe			  : out 	 std_logic;

    led_n       : out   std_logic_vector(15 downto 0);
	 PushButton	 : in 	std_logic_vector(1 downto 0);

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

    signal clk_125, rst_125_n : std_logic;

    signal nios_clk, nios_rst_n : std_logic;
    signal nios_pio : std_logic_vector(31 downto 0);

    signal i2c_scl_in, i2c_scl_oe, i2c_sda_in, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(1 downto 0);

    signal malibu_clk : std_logic;

    signal avm_pod, avm_qsfp : work.mu3e.avalon_t;

    signal qsfp_tx_clk : std_logic_vector(3 downto 0);
    signal qsfp_tx_data : std_logic_vector(127 downto 0);
    signal qsfp_tx_datak : std_logic_vector(15 downto 0);

    signal qsfp_rx_clk : std_logic_vector(3 downto 0);
    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);

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
	 
	 signal mscb_to_nios_parallel_in : std_logic_vector(11 downto 0);
	 signal mscb_from_nios_parallel_out : std_logic_vector(11 downto 0);
	 signal mscb_counter_in : unsigned(15 downto 0);
	 
	 signal clk_reset_recovered :	std_logic;
	 signal reset_link8b : std_logic_vector(7 downto 0);
	 signal reconfig_fromgxb_pod : STD_LOGIC_VECTOR (16 DOWNTO 0);
	 signal reconfig_togxb_pod : STD_LOGIC_VECTOR (3 DOWNTO 0);

begin

    led_n <= not led;
	 led(8) <= PushButton(0);
	 led(9) <= PushButton(1);
	 nios_rst_n <= (not PushButton(0)) and PushButton(1);

    -- 125 MHz
    i_aux_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(15), rst_n => reset_n, clk => clk_aux );

    -- 125 MHz
    i_si45_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(14), rst_n => reset_n, clk => qsfp_pll_clk );

    clk_125 <= clk_aux;

    i_rst_125_n : entity work.reset_sync
    port map ( rstout_n => rst_125_n, arst_n => reset_n, clk => clk_125 );



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
        inclk0 => clk_125--,
    );

--    i_nios_rst_n : entity work.reset_sync
--    port map ( rstout_n => nios_rst_n, arst_n => reset_n, clk => clk_125 );
    --nios_rst_n <= '1';

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

        sc_clk_clk          => qsfp_rx_clk(0),
        sc_reset_reset_n    => '1',

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
		  
		  parallel_mscb_in_export => mscb_to_nios_parallel_in,
		  parallel_mscb_out_export => mscb_from_nios_parallel_out,
		  counter_in_export => std_logic_vector(mscb_counter_in),
        rst_reset_n => nios_rst_n,
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
        areset => not reset_n,
        inclk0 => clk_125--,
    );

    malibu_ck_fpga_1 <= '0';
    malibu_pll_reset <= '0';
    malibu_ck_fpga_0 <= malibu_clk;

    -- reset tdc and digital part
    malibu_chip_reset <= nios_pio(16);

    e_test_pulse : entity work.clkdiv
    generic map ( P => 125 )
    port map ( clkout => malibu_pll_test, rst_n => rst_125_n, clk => clk_125 );

    i_malibu : entity work.malibu_dec
    generic map (
        N => 2--,
    )
    port map (
        data_clk    => open,
        data        => open,
        datak       => open,

        rx_data => malibu_data(1 downto 0),
        rx_clk  => malibu_clk,

        reset   => not reset_n--,
    );

    i_mutrig_datapath : entity work.mutrig_datapath
    port map (
        i_rst => not reset_n,
        i_stic_txd => malibu_data(0 downto 0),
        i_refclk_125 => clk_125,

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
        pll_freq => 125--,
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

        rx_data => qsfp_rx_data,
        rx_datak => qsfp_rx_datak,

        tx_clkout   => qsfp_tx_clk,
        tx_clkin    => qsfp_tx_clk,
        rx_clkout   => qsfp_rx_clk,
        rx_clkin    => qsfp_rx_clk,

        tx_p        => qsfp_tx,
        rx_p        => qsfp_rx,

        pll_refclk  => qsfp_pll_clk,
        cdr_refclk  => qsfp_pll_clk,

        reset   => not nios_rst_n,
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
        clock_b => qsfp_rx_clk(0),

        address_a => ram_addr_a(13 downto 0),
        q_a => ram_rdata_a,
        wren_a => ram_we_a,
        data_a => ram_wdata_a,
        clock_a => qsfp_rx_clk(0)--,
    );
    avm_sc.waitrequest <= '0';

    i_sc : entity work.sc_s4
    port map (
        clk => qsfp_rx_clk(0),
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
	 -- MSCB
	 i_mscb : entity work.mscb 
    port map(
			nios_clk								=> nios_clk,
			reset 								=> not nios_rst_n,
			mscb_to_nios_parallel_in		=> mscb_to_nios_parallel_in,
			mscb_from_nios_parallel_out   => mscb_from_nios_parallel_out,
			mscb_data_in						=> mscb_data_in,
			mscb_data_out						=> mscb_data_out,
			mscb_oe								=> mscb_oe,
			mscb_counter_in 					=> mscb_counter_in--,
    );
    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- data gen
    
    i_data_gen : entity work.data_generator
    port map (
        clk => qsfp_tx_clk(0),
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
        clk                     => qsfp_tx_clk(0),
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
        rdclk   => qsfp_tx_clk(0),
        rdreq   => data_from_fifo_re,
        wrclk   => qsfp_tx_clk(0),
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
        rdclk   => qsfp_tx_clk(0),
        rdreq   => sc_from_fifo_re,
        wrclk   => qsfp_rx_clk(0),
        wrreq   => sc_to_fifo_we,
        q       => sc_from_fifo,
        rdempty => sc_from_fifo_empty,
        wrfull  => open--,
    );

    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- POD

    pod_rx_reset <= '0';
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
--        reset   => not nios_rst_n,
--        clk     => nios_clk--,
--    );

----------------------------------------------------------------------------
----------------------------------------------------------------------------
--pod 8b
i_pod: entity work.ip_altgx8b
	port map
	(
		cal_blk_clk				=> clk_aux,
		reconfig_clk			=> clk_aux,
		reconfig_togxb			=> reconfig_togxb_pod,
		rx_analogreset(0)		=> not nios_rst_n,
		rx_cruclk(0)			=> clk_aux,
		rx_datain(0)			=> pod_rx,
		rx_digitalreset(0)	=> not nios_rst_n,
		reconfig_fromgxb		=> reconfig_fromgxb_pod,
		rx_clkout(0)			=> clk_reset_recovered,
		rx_ctrldetect			=> open,
		rx_dataout				=> reset_link8b,
		rx_syncstatus			=> open
	);
	
i_altgx_reconfig8b: entity work.ip_altgx_reconfig8b
	port map
	(
		reconfig_clk			=> clk_aux,
		reconfig_fromgxb		=> reconfig_fromgxb_pod,
		busy						=> open,
		reconfig_togxb			=> reconfig_togxb_pod
	);
	
led(7 downto 0) <= reset_link8b(7 downto 0);
end architecture;
