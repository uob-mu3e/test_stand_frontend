library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;

entity top is
port (

	-- FE.A

	clock_A				: out std_logic;
	data_in_A_0			: in std_logic_vector(3 downto 0);
	data_in_A_1			: in std_logic_vector(3 downto 0);
	fast_reset_A			: out std_logic;
	test_pulse_A			: out std_logic;

	CTRL_SDO_A				: in std_logic; -- A_ctrl_dout_front
	CTRL_SDI_A				: out std_logic; -- A_ctrl_din_front
	CTRL_SCK1_A			: out std_logic; -- A_ctrl_clk1_front
	CTRL_SCK2_A			: out std_logic; -- A_ctrl_clk2_front
	CTRL_RB_A				: out std_logic; -- A_ctrl_rb_front
	CTRL_Load_A			: out std_logic; -- A_ctrl_ld_front

	-- A_trig_front

	chip_reset_A			: out std_logic; -- is called trigger on adapter card!
	SPI_DIN0_A				: out std_logic; -- A_spi_din_front             -- AH9
	SPI_DIN1_A				: out std_logic; -- A_spi_din_back              --
	SPI_CLK_A				: out std_logic; -- A_spi_clk_front             -- AH8
	SPI_LD_DAC_A			: out std_logic; -- A_spi_ld_front              -- AW4
	SPI_LD_ADC_A			: out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
	SPI_LD_TEMP_DAC_A		: out std_logic; -- A_spi_ld_adc_front          -- AN7
	SPI_DOUT_ADC_0_A		: in std_logic; -- A_spi_dout_adc_front         -- AT9
	SPI_DOUT_ADC_1_A		: in std_logic; -- A_spi_dout_adc_back          --

	-- FE.B

	clock_B				: out std_logic;
	data_in_B_0			: in std_logic_vector(3 downto 0);
	data_in_B_1			: in std_logic_vector(3 downto 0);
	fast_reset_B			: out std_logic;
	test_pulse_B			: out std_logic;

	CTRL_SDO_B				: in std_logic; -- A_ctrl_dout_front
	CTRL_SDI_B				: out std_logic; -- A_ctrl_din_front
	CTRL_SCK1_B			: out std_logic; -- A_ctrl_clk1_front
	CTRL_SCK2_B			: out std_logic; -- A_ctrl_clk2_front
	CTRL_RB_B				: out std_logic; -- A_ctrl_rb_front
	CTRL_Load_B			: out std_logic; -- A_ctrl_ld_front

	-- B_trig_front

	chip_reset_B			: out std_logic; -- is called trigger on adapter card!
	SPI_DIN0_B				: out std_logic; -- A_spi_din_front             -- AH9
	SPI_DIN1_B				: out std_logic; -- A_spi_din_back              --
	SPI_CLK_B				: out std_logic; -- A_spi_clk_front             -- AH8
	SPI_LD_DAC_B			: out std_logic; -- A_spi_ld_front              -- AW4
	SPI_LD_ADC_B			: out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
	SPI_LD_TEMP_DAC_B		: out std_logic; -- A_spi_ld_adc_front          -- AN7
	SPI_DOUT_ADC_0_B		: in std_logic; -- A_spi_dout_adc_front         -- AT9
	SPI_DOUT_ADC_1_B		: in std_logic; -- A_spi_dout_adc_back          --

	-- FE.C

	clock_C				: out std_logic;
	data_in_C_0			: in std_logic_vector(3 downto 0);
	data_in_C_1			: in std_logic_vector(3 downto 0);
	fast_reset_C			: out std_logic;
	test_pulse_C			: out std_logic;

	CTRL_SDO_C				: in std_logic; -- A_ctrl_dout_front
	CTRL_SDI_C				: out std_logic; -- A_ctrl_din_front
	CTRL_SCK1_C			: out std_logic; -- A_ctrl_clk1_front
	CTRL_SCK2_C			: out std_logic; -- A_ctrl_clk2_front
	CTRL_RB_C				: out std_logic; -- A_ctrl_rb_front
	CTRL_Load_C			: out std_logic; -- A_ctrl_ld_front

	-- C_trig_front

	chip_reset_C			: out std_logic; -- is called trigger on adapter card!
	SPI_DIN0_C				: out std_logic; -- A_spi_din_front             -- AH9
	SPI_DIN1_C				: out std_logic; -- A_spi_din_back              --
	SPI_CLK_C				: out std_logic; -- A_spi_clk_front             -- AH8
	SPI_LD_DAC_C			: out std_logic; -- A_spi_ld_front              -- AW4
	SPI_LD_ADC_C			: out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
	SPI_LD_TEMP_DAC_C		: out std_logic; -- A_spi_ld_adc_front          -- AN7
	SPI_DOUT_ADC_0_C		: in std_logic; -- A_spi_dout_adc_front         -- AT9
	SPI_DOUT_ADC_1_C		: in std_logic; -- A_spi_dout_adc_back          --


	-- FE.E

	clock_E				: out std_logic;
	data_in_E_0			: in std_logic_vector(3 downto 0);
	data_in_E_1			: in std_logic_vector(3 downto 0);
	fast_reset_E			: out std_logic;
	test_pulse_E			: out std_logic;

	CTRL_SDO_E				: in std_logic; -- A_ctrl_dout_front
	CTRL_SDI_E				: out std_logic; -- A_ctrl_din_front
	CTRL_SCK1_E			: out std_logic; -- A_ctrl_clk1_front
	CTRL_SCK2_E			: out std_logic; -- A_ctrl_clk2_front
	CTRL_RB_E				: out std_logic; -- A_ctrl_rb_front
	CTRL_Load_E			: out std_logic; -- A_ctrl_ld_front

	-- E_trig_front

	chip_reset_E			: out std_logic; -- is called trigger on adapter card!
	SPI_DIN0_E				: out std_logic; -- A_spi_din_front             -- AH9
	SPI_DIN1_E				: out std_logic; -- A_spi_din_back              --
	SPI_CLK_E				: out std_logic; -- A_spi_clk_front             -- AH8
	SPI_LD_DAC_E			: out std_logic; -- A_spi_ld_front              -- AW4
	SPI_LD_ADC_E			: out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
	SPI_LD_TEMP_DAC_E		: out std_logic; -- A_spi_ld_adc_front          -- AN7
	SPI_DOUT_ADC_0_E		: in std_logic; -- A_spi_dout_adc_front         -- AT9
	SPI_DOUT_ADC_1_E		: in std_logic; -- A_spi_dout_adc_back          --



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
    FPGA_Test   : inout std_logic_vector(2 downto 0);
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

    signal nios_clk : std_logic;

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(15 downto 0);

    signal run_state_125 : run_state_t;

    signal sync_reset_cnt : std_logic;
    signal nios_clock : std_logic;

begin

    ----------------------------------------------------------------------------
    -- MUPIX

	e_mupix_block : entity work.mupix_block
	generic map (
		NCHIPS => 8,
		NCHIPS_SPI => 1,
		NLVDS  => 32,
		NINPUTS_BANK_A => 16,
		NINPUTS_BANK_B => 16--,
	)
	port map (

		-- chip dacs
		i_CTRL_SDO_A         => CTRL_SDO_A,
		o_CTRL_SDI_A         => CTRL_SDI_A,
		o_CTRL_SCK1_A        => CTRL_SCK1_A,
		o_CTRL_SCK2_A        => CTRL_SCK2_A,
		o_CTRL_Load_A        => CTRL_Load_A,
		o_CTRL_RB_A          => CTRL_RB_A,

		-- board dacs
		i_SPI_DOUT_ADC_0_A   => SPI_DOUT_ADC_0_A,
		o_SPI_DIN0_A         => SPI_DIN0_A,
		o_SPI_CLK_A          => SPI_CLK_A,
		o_SPI_LD_ADC_A       => SPI_LD_ADC_A,
		o_SPI_LD_TEMP_DAC_A  => SPI_LD_TEMP_DAC_A,
		o_SPI_LD_DAC_A       => SPI_LD_DAC_A,

		-- mupix dac regs
		i_reg_add               => mupix_reg.addr(7 downto 0),
		i_reg_re                => mupix_reg.re,
		o_reg_rdata             => mupix_reg.rdata,
		i_reg_we                => mupix_reg.we,
		i_reg_wdata             => mupix_reg.wdata,

		-- data
		o_fifo_rdata            => fifo_rdata,
		o_fifo_rempty           => fifo_rempty,
		i_fifo_rack             => fifo_rack,

		i_data_in_A_0           => data_in_A_0,
		i_data_in_A_1           => data_in_A_1,
		i_data_in_B_0           => data_in_B_0,
		i_data_in_B_1           => data_in_B_1,
		i_data_in_C_0           => data_in_C_0,
		i_data_in_C_1           => data_in_C_1,
		i_data_in_E_0           => data_in_E_0,
		i_data_in_E_1           => data_in_E_1,

		i_reset              => not reset_n,
		-- 156.25 MHz
		i_clk                => qsfp_clk,
		i_clk125             => clk_125_bottom,
		i_sync_reset_cnt    => sync_reset_cnt--,
	);

	clock_A <= pod_clk_left;
	clock_B <= pod_clk_left;
	clock_C <= pod_clk_left;
	clock_E <= pod_clk_left;

	process(pod_clk_left)
	begin
	if falling_edge(pod_clk_left) then
		if(run_state_125 = RUN_STATE_SYNC)then
			fast_reset_A <= '1';
			fast_reset_B <= '1';
			fast_reset_C <= '1';
			fast_reset_E <= '1';
			sync_reset_cnt	<= '1';
		else
			fast_reset_A <= '0';
			fast_reset_B <= '0';
			fast_reset_C <= '0';
			fast_reset_E <= '0';
			sync_reset_cnt	<= '0';
		end if;
	end if;
	end process;

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

--    i2c_scl <= not i2c_scl_oe;
--    i2c_sda <=
--        malibu_i2c_sda and
--        '1';
--    malibu_i2c_scl <= ZERO when i2c_scl_oe = '1' else 'Z';
--    malibu_i2c_sda <= ZERO when i2c_sda_oe = '1' else 'Z';

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- SPI
--
--    malibu_spi_sdi <= spi_mosi;
--    malibu_spi_sck <= spi_sclk when spi_ss_n(1) = '0' else '0';
--
--    spi_miso <=
--        malibu_spi_sdo when spi_ss_n(1) = '0' else
--        '0';

--    SPI_SCK_A <= SPI_CLK_A;
--    SPI_SDI_A <= SPI_DIN0_A;
--    SPI_DOUT_ADC_0_A <= SPI_SDO_A;
--    SPI_Load_A <= SPI_LD_ADC_A & SPI_LD_TEMP_DAC_A & SPI_LD_DAC_A;

    ----------------------------------------------------------------------------



    e_nios_clk : component work.cmp.clk_ctrl
    port map (
		inclk1x   => clk_aux,
		inclk0x   => si42_clk_50,
		clkselect => FPGA_Test(1),
		outclk    => nios_clk--,
    );
	-- use these two signals with a jumper to FPGA_Test(1) to select the clock
	FPGA_Test(0) <= '0';
	FPGA_Test(2) <= '1';



    e_fe_block : entity work.fe_block
    generic map (
        NIOS_CLK_MHZ_g => 50.0--,
    )
    port map (
        i_fpga_id       => X"FEB0",
        i_fpga_type     => "111010",

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

        o_mupix_reg_addr    => mupix_reg.addr(7 downto 0),
        o_mupix_reg_re      => mupix_reg.re,
        i_mupix_reg_rdata   => mupix_reg.rdata,
        o_mupix_reg_we      => mupix_reg.we,
        o_mupix_reg_wdata   => mupix_reg.wdata,

        -- reset system
        o_run_state_125 => run_state_125,


        -- clocks
        i_nios_clk      => si42_clk_50,
        -- i replaced nios_clk with si42_clk_50 here, M.Mueller, FPGA_Test jumper not used for testbeam version
        o_nios_clk_mon  => led(15),
        i_clk_156       => qsfp_clk,
        o_clk_156_mon   => led(14),
        i_clk_125       => pod_clk_left,
        o_clk_125_mon   => led(13),

        i_areset_n      => reset_n--,
    );

end architecture;
