library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;

entity top is
port (

    -- FE.A
    clock_A                 : out std_logic;
    data_in_A_0             : in  std_logic_vector(3 downto 0);
    data_in_A_1             : in  std_logic_vector(3 downto 0);
    fast_reset_A            : out std_logic;
    test_pulse_A            : out std_logic;

    CTRL_SDO_A              : in  std_logic; -- A_ctrl_dout_front
    CTRL_SDI_A              : out std_logic; -- A_ctrl_din_front
    CTRL_SCK1_A             : out std_logic; -- A_ctrl_clk1_front
    CTRL_SCK2_A             : out std_logic; -- A_ctrl_clk2_front
    CTRL_RB_A               : out std_logic; -- A_ctrl_rb_front
    CTRL_Load_A             : out std_logic; -- A_ctrl_ld_front

    -- A_trig_front
    chip_reset_A            : out std_logic; -- is called trigger on adapter card!
    SPI_DIN0_A              : out std_logic; -- A_spi_din_front             -- AH9
    SPI_DIN1_A              : out std_logic; -- A_spi_din_back              --
    SPI_CLK_A               : out std_logic; -- A_spi_clk_front             -- AH8
    SPI_LD_DAC_A            : out std_logic; -- A_spi_ld_front              -- AW4
    SPI_LD_ADC_A            : out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
    SPI_LD_TEMP_DAC_A       : out std_logic; -- A_spi_ld_adc_front          -- AN7
    SPI_DOUT_ADC_0_A        : in  std_logic; -- A_spi_dout_adc_front        -- AT9
    SPI_DOUT_ADC_1_A        : in  std_logic; -- A_spi_dout_adc_back         --

    -- FE.B
    clock_B                 : out std_logic;
    data_in_B_0             : in  std_logic_vector(3 downto 0);
    data_in_B_1             : in  std_logic_vector(3 downto 0);
    fast_reset_B            : out std_logic;
    test_pulse_B            : out std_logic;

    CTRL_SDO_B              : in  std_logic; -- A_ctrl_dout_front
    CTRL_SDI_B              : out std_logic; -- A_ctrl_din_front
    CTRL_SCK1_B             : out std_logic; -- A_ctrl_clk1_front
    CTRL_SCK2_B             : out std_logic; -- A_ctrl_clk2_front
    CTRL_RB_B               : out std_logic; -- A_ctrl_rb_front
    CTRL_Load_B             : out std_logic; -- A_ctrl_ld_front

    -- B_trig_front
    chip_reset_B            : out std_logic; -- is called trigger on adapter card!
    SPI_DIN0_B              : out std_logic; -- A_spi_din_front             -- AH9
    SPI_DIN1_B              : out std_logic; -- A_spi_din_back              --
    SPI_CLK_B               : out std_logic; -- A_spi_clk_front             -- AH8
    SPI_LD_DAC_B            : out std_logic; -- A_spi_ld_front              -- AW4
    SPI_LD_ADC_B            : out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
    SPI_LD_TEMP_DAC_B       : out std_logic; -- A_spi_ld_adc_front          -- AN7
    SPI_DOUT_ADC_0_B        : in  std_logic; -- A_spi_dout_adc_front        -- AT9
    SPI_DOUT_ADC_1_B        : in  std_logic; -- A_spi_dout_adc_back         --

    -- FE.C
    clock_C                 : out std_logic;
    data_in_C_0             : in  std_logic_vector(3 downto 0);
    data_in_C_1             : in  std_logic_vector(3 downto 0);
    fast_reset_C            : out std_logic;
    test_pulse_C            : out std_logic;

    CTRL_SDO_C              : in  std_logic; -- A_ctrl_dout_front
    CTRL_SDI_C              : out std_logic; -- A_ctrl_din_front
    CTRL_SCK1_C             : out std_logic; -- A_ctrl_clk1_front
    CTRL_SCK2_C             : out std_logic; -- A_ctrl_clk2_front
    CTRL_RB_C               : out std_logic; -- A_ctrl_rb_front
    CTRL_Load_C             : out std_logic; -- A_ctrl_ld_front

    -- C_trig_front
    chip_reset_C            : out std_logic; -- is called trigger on adapter card!
    SPI_DIN0_C              : out std_logic; -- A_spi_din_front             -- AH9
    SPI_DIN1_C              : out std_logic; -- A_spi_din_back              --
    SPI_CLK_C               : out std_logic; -- A_spi_clk_front             -- AH8
    SPI_LD_DAC_C            : out std_logic; -- A_spi_ld_front              -- AW4
    SPI_LD_ADC_C            : out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
    SPI_LD_TEMP_DAC_C       : out std_logic; -- A_spi_ld_adc_front          -- AN7
    SPI_DOUT_ADC_0_C        : in  std_logic; -- A_spi_dout_adc_front        -- AT9
    SPI_DOUT_ADC_1_C        : in  std_logic; -- A_spi_dout_adc_back         --

    -- FE.E
    clock_E                 : out std_logic;
    data_in_E_0             : in  std_logic_vector(3 downto 0);
    data_in_E_1             : in  std_logic_vector(3 downto 0);
    fast_reset_E            : out std_logic;
    test_pulse_E            : out std_logic;

    CTRL_SDO_E              : in  std_logic; -- A_ctrl_dout_front
    CTRL_SDI_E              : out std_logic; -- A_ctrl_din_front
    CTRL_SCK1_E             : out std_logic; -- A_ctrl_clk1_front
    CTRL_SCK2_E             : out std_logic; -- A_ctrl_clk2_front
    CTRL_RB_E               : out std_logic; -- A_ctrl_rb_front
    CTRL_Load_E             : out std_logic; -- A_ctrl_ld_front

    -- E_trig_front
    chip_reset_E            : out std_logic; -- is called trigger on adapter card!
    SPI_DIN0_E              : out std_logic; -- A_spi_din_front             -- AH9
    SPI_DIN1_E              : out std_logic; -- A_spi_din_back              --
    SPI_CLK_E               : out std_logic; -- A_spi_clk_front             -- AH8
    SPI_LD_DAC_E            : out std_logic; -- A_spi_ld_front              -- AW4
    SPI_LD_ADC_E            : out std_logic; -- A_spi_ld_tmp_dac_front      -- AP7
    SPI_LD_TEMP_DAC_E       : out std_logic; -- A_spi_ld_adc_front          -- AN7
    SPI_DOUT_ADC_0_E        : in  std_logic; -- A_spi_dout_adc_front        -- AT9
    SPI_DOUT_ADC_1_E        : in  std_logic; -- A_spi_dout_adc_back         --

    -- Si5342
    si42_oe_n               : out std_logic; -- <= '0'
    si42_rst_n              : out std_logic; -- reset
    si42_spi_out            : in  std_logic; -- slave data out
    si42_spi_in             : out std_logic; -- slave data in
    si42_spi_sclk           : out std_logic; -- clock
    si42_spi_cs_n           : out std_logic; -- chip select

    -- Si5345
    si45_oe_n               : out std_logic; -- <= '0'
    si45_rst_n              : out std_logic; -- reset
    si45_spi_out            : in  std_logic; -- slave data out
    si45_spi_in             : out std_logic; -- slave data in
    si45_spi_sclk           : out std_logic; -- clock
    si45_spi_cs_n           : out std_logic; -- chip select

    -- POD

    -- Si5345 out0 (125 MHz)
    pod_clk_left            : in  std_logic;
    -- Si5345 out1 (125 MHz)
    -- pod_clk_right        : in  std_logic;

    pod_tx_reset_n          : out std_logic;
    pod_rx_reset_n          : out std_logic;

    pod_tx                  : out std_logic_vector(3 downto 0);
    pod_rx                  : in  std_logic_vector(3 downto 0);

    -- QSFP

    -- Si5345 out2 (156.25 MHz)
    qsfp_clk                : in  std_logic;

    QSFP_ModSel_n           : out std_logic; -- module select (i2c)
    QSFP_Rst_n              : out std_logic;
    QSFP_LPM                : out std_logic; -- Low Power Mode

    qsfp_tx                 : out std_logic_vector(3 downto 0);
    qsfp_rx                 : in  std_logic_vector(3 downto 0);

    -- Si5345 out3 (125 MHz, right)
    lvds_clk_A              : in  std_logic;
    -- Si5345 out6 (125 MHz, left)
    lvds_clk_B              : in  std_logic;

    -- Si5345 out7 (125 MHz)
    clk_125_bottom          : in  std_logic; -- global 125 MHz clock
    -- Si5345 out8 (125 MHz)
    clk_125_top             : in  std_logic;

    -- MSCB
    mscb_data_in            : in  std_logic;
    mscb_data_out           : out std_logic;
    mscb_oe                 : out std_logic;

    led_n                   : out   std_logic_vector(15 downto 0);
    FPGA_Test               : inout std_logic_vector(2 downto 0);
    PushButton              : in    std_logic_vector(1 downto 0);

    -- Si5345 out0 (125 MHz)
    si42_clk_125            : in  std_logic;
    -- Si5345 out1 (50 MHz)
    si42_clk_50             : in  std_logic;

    clk_aux                 : in  std_logic;
    reset_n                 : in  std_logic--;
);
end entity;

architecture arch of top is

    constant NPORTS         : integer := 4;
    constant N_LINKS        : integer := 1;

    signal led              : std_logic_vector(led_n'range) := (others => '0');

    signal fifo_write: std_logic_vector(N_LINKS-1 downto 0);
    signal fifo_wdata : std_logic_vector(36*(N_LINKS-1)+35 downto 0);

    signal malibu_reg, scifi_reg, mupix_reg : work.util.rw_t;

    signal nios_clk         : std_logic;

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO             : std_logic := '0';
    attribute keep          : boolean;
    attribute keep of ZERO  : signal is true;

    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n         : std_logic_vector(15 downto 0);

    signal run_state_125    : run_state_t;

    signal sync_reset_cnt   : std_logic;
    signal nios_clock       : std_logic;
    
    -- board dacs
    signal SPI_DIN          : std_logic;
    signal SPI_CLK          : std_logic;
    signal SPI_LD_ADC       : std_logic;
    signal SPI_LD_TEMP_DAC  : std_logic;
    signal SPI_LD_DAC       : std_logic;
    
    -- chip dacs
    signal CTRL_SDI         : std_logic_vector(NPORTS-1 downto 0);
    signal CTRL_SCK1        : std_logic_vector(NPORTS-1 downto 0);
    signal CTRL_SCK2        : std_logic_vector(NPORTS-1 downto 0);
    signal CTRL_LOAD        : std_logic_vector(NPORTS-1 downto 0);
    signal CTRL_RB          : std_logic_vector(NPORTS-1 downto 0);
    
    

begin

    ----------------------------------------------------------------------------
    -- MUPIX
    
    SPI_DIN0_A            <= SPI_DIN;
    SPI_CLK_A             <= SPI_CLK;
    SPI_LD_ADC_A          <= SPI_LD_ADC;
    SPI_LD_TEMP_DAC_A     <= SPI_LD_TEMP_DAC;
    SPI_LD_DAC_A          <= SPI_LD_DAC;
    
    SPI_DIN0_B            <= SPI_DIN;
    SPI_CLK_B             <= SPI_CLK;
    SPI_LD_ADC_B          <= SPI_LD_ADC;
    SPI_LD_TEMP_DAC_B     <= SPI_LD_TEMP_DAC;
    SPI_LD_DAC_B          <= SPI_LD_DAC;
    
    SPI_DIN0_C            <= SPI_DIN;
    SPI_CLK_C             <= SPI_CLK;
    SPI_LD_ADC_C          <= SPI_LD_ADC;
    SPI_LD_TEMP_DAC_C     <= SPI_LD_TEMP_DAC;
    SPI_LD_DAC_C          <= SPI_LD_DAC;
    
    SPI_DIN0_E            <= SPI_DIN;
    SPI_CLK_E             <= SPI_CLK;
    SPI_LD_ADC_E          <= SPI_LD_ADC;
    SPI_LD_TEMP_DAC_E     <= SPI_LD_TEMP_DAC;
    SPI_LD_DAC_E          <= SPI_LD_DAC;

    CTRL_SDI_A  <= CTRL_SDI(0);
    CTRL_SCK1_A <= CTRL_SCK1(0);
    CTRL_SCK2_A <= CTRL_SCK2(0);
    CTRL_LOAD_A <= CTRL_LOAD(0);
    CTRL_RB_A   <= CTRL_RB(0);
    
    gen_portB: if NPORTS > 1 generate
        CTRL_SDI_B  <= CTRL_SDI(1);
        CTRL_SCK1_B <= CTRL_SCK1(1);
        CTRL_SCK2_B <= CTRL_SCK2(1);
        CTRL_LOAD_B <= CTRL_LOAD(1);
        CTRL_RB_B   <= CTRL_RB(1);
    end generate;

    gen_portC: if NPORTS > 2 generate
        CTRL_SDI_C  <= CTRL_SDI(2);
        CTRL_SCK1_C <= CTRL_SCK1(2);
        CTRL_SCK2_C <= CTRL_SCK2(2);
        CTRL_LOAD_C <= CTRL_LOAD(2);
        CTRL_RB_C   <= CTRL_RB(2);
    end generate;

    gen_portE: if NPORTS > 3 generate
        CTRL_SDI_E  <= CTRL_SDI(3);
        CTRL_SCK1_E <= CTRL_SCK1(3);
        CTRL_SCK2_E <= CTRL_SCK2(3);
        CTRL_LOAD_E <= CTRL_LOAD(3);
        CTRL_RB_E   <= CTRL_RB(3);
    end generate;

    e_mupix_block : entity work.mupix_block
    generic map (
        NCHIPS          => 2,
        NCHIPS_SPI      => 4,
        NPORTS          => NPORTS,
        NLVDS           => 32,
        NINPUTS_BANK_A  => 16,
        NINPUTS_BANK_B  => 16--,
    )
    port map (

        -- chip dacs
        i_CTRL_SDO_A            => CTRL_SDO_A,
        o_CTRL_SDI              => CTRL_SDI,
        o_CTRL_SCK1             => CTRL_SCK1,
        o_CTRL_SCK2             => CTRL_SCK2,
        o_CTRL_Load             => CTRL_Load,
        o_CTRL_RB               => CTRL_RB,

        -- board dacs
        i_SPI_DOUT_ADC_0_A      => SPI_DOUT_ADC_0_A,
        o_SPI_DIN0_A            => SPI_DIN,
        o_SPI_CLK_A             => SPI_CLK,
        o_SPI_LD_ADC_A          => SPI_LD_ADC,
        o_SPI_LD_TEMP_DAC_A     => SPI_LD_TEMP_DAC,
        o_SPI_LD_DAC_A          => SPI_LD_DAC,

        -- mupix dac regs
        i_reg_add               => mupix_reg.addr(7 downto 0),
        i_reg_re                => mupix_reg.re,
        o_reg_rdata             => mupix_reg.rdata,
        i_reg_we                => mupix_reg.we,
        i_reg_wdata             => mupix_reg.wdata,

    -- data
        o_fifo_wdata            => fifo_wdata,
        o_fifo_write            => fifo_write(0),

        i_run_state_125         => run_state_125,

        i_data_in_A_0           => data_in_A_0,
        i_data_in_A_1           => data_in_A_1,
        i_data_in_B_0           => data_in_B_0,
        i_data_in_B_1           => data_in_B_1,
        i_data_in_C_0           => data_in_C_0,
        i_data_in_C_1           => data_in_C_1,
        i_data_in_E_0           => data_in_E_0,
        i_data_in_E_1           => data_in_E_1,

        i_reset                 => not reset_n,
        -- 156.25 MHz
        i_clk                   => qsfp_clk,
        i_clk125                => clk_125_bottom,
        i_sync_reset_cnt        => sync_reset_cnt--,
    );

    clock_A <= pod_clk_left;
    clock_B <= pod_clk_left;
    clock_C <= pod_clk_left;
    clock_E <= pod_clk_left;

    process(pod_clk_left)
    begin
    if falling_edge(pod_clk_left) then
        if(run_state_125 = RUN_STATE_SYNC)then
            fast_reset_A    <= '1';
            fast_reset_B    <= '1';
            fast_reset_C    <= '1';
            fast_reset_E    <= '1';
            sync_reset_cnt  <= '1';
        else
            fast_reset_A    <= '0';
            fast_reset_B    <= '0';
            fast_reset_C    <= '0';
            fast_reset_E    <= '0';
            sync_reset_cnt  <= '0';
        end if;
    end if;
    end process;

    ----------------------------------------------------------------------------
    led_n <= not led;

    -- enable Si5342
    si42_oe_n       <= '0';
    si42_rst_n      <= '1';

    -- enable Si5345
    si45_oe_n       <= '0';
    si45_rst_n      <= '1';

    -- enable QSFP
    QSFP_ModSel_n   <= '1';
    QSFP_Rst_n      <= '1';
    QSFP_LPM        <= '0';

    -- enable POD
    pod_tx_reset_n  <= '1';
    pod_rx_reset_n  <= '1';

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
    
    ----------------------------------------------------------------------------
    
    e_fe_block : entity work.fe_block
    generic map (
        NIOS_CLK_MHZ_g  => 50.0--,
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

        i_fifo_write    => fifo_write,
        i_fifo_wdata    => fifo_wdata,

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
