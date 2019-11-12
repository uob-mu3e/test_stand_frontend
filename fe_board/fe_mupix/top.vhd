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
    signal mupix_reg : work.util.rw_t;

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

    signal run_state_125 : run_state_t;

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

    -- mupix regs : 0x80-0x9F
    mupix_reg.addr <= sc_reg.addr;
    mupix_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 4) = X"8" or sc_reg.addr(7 downto 4) = X"9" ) else '0';
    mupix_reg.we <= sc_reg.we when ( sc_reg.addr(7 downto 4) = X"8" or sc_reg.addr(7 downto 4) = X"9" ) else '0';
    mupix_reg.wdata <= sc_reg.wdata;

    -- select valid rdata
    sc_reg.rdata <=
        malibu_reg.rdata when ( malibu_reg.rvalid = '1' ) else
        scifi_reg.rdata when ( scifi_reg.rvalid = '1' ) else
        mupix_reg.rdata when ( mupix_reg.rvalid = '1' ) else
        X"CCCCCCCC";

    process(qsfp_pll_clk)
    begin
    if rising_edge(qsfp_pll_clk) then
--        malibu_reg.rdata <= X"CCCCCCCC";
        malibu_reg.rvalid <= malibu_reg.re;
        scifi_reg.rdata <= X"CCCCCCCC";
        scifi_reg.rvalid <= scifi_reg.re;
        mupix_reg.rvalid <= mupix_reg.re;
    end if;
    end process;



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
		i_clk                => qsfp_pll_clk,
		i_clk125             => clk_aux--,
    );

	clock_A <= clk_aux;
	clock_B <= clk_aux;
	clock_C <= clk_aux;
	clock_E <= clk_aux;

	process(clk_aux)
	begin
	if falling_edge(clk_aux) then
		if(run_state_125 = RUN_STATE_SYNC)then
			fast_reset_A <= '1';
			fast_reset_B <= '1';
			fast_reset_C <= '1';
			fast_reset_E <= '1';
		else
			fast_reset_A <= '0';
			fast_reset_B <= '0';
			fast_reset_C <= '0';
			fast_reset_E <= '0';
		end if;
	end if;
	end process;

    ----------------------------------------------------------------------------

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



    e_fe_block : entity work.fe_block
    generic map (
        FPGA_ID_g => X"FEB0",
        FEB_type_in => "111010"
    )
    port map (
        i_nios_clk_startup => clk_aux,
        i_nios_clk_main => clk_aux,
        i_nios_areset_n => reset_n,
        o_nios_clk_monitor => nios_clk,
        o_nios_clk_selected => led(10),

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
        i_clk           => qsfp_pll_clk,

        o_run_state_125	=> run_state_125
    );

end architecture;
