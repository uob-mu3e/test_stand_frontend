library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    --  PCIe connector    
    A10_CPCIE_SMB_SCL                   : INOUT STD_LOGIC;
    A10_CPCIE_SMB_SDA                   : INOUT STD_LOGIC;

    --  Reference clocks  
    A10_REFCLK_TFC_CMU_P                : IN    STD_LOGIC;
    A10_REFCLK_TFC_P                    : IN    STD_LOGIC;
    A10_REFCLK_2_TFC_P                  : IN    STD_LOGIC;
    A10_CLK_PCIE_P_0                    : IN    STD_LOGIC;
    A10_CLK_PCIE_P_1                    : IN    STD_LOGIC;
    A10_REFCLK_10G_P_0                  : IN    STD_LOGIC;
    A10_REFCLK_10G_P_1                  : IN    STD_LOGIC;
    A10_REFCLK_10G_P_2                  : IN    STD_LOGIC;
    A10_REFCLK_10G_P_3                  : IN    STD_LOGIC;
    A10_REFCLK_10G_P_4                  : IN    STD_LOGIC;
    A10_REFCLK_10G_P_5                  : IN    STD_LOGIC;
    A10_REFCLK_10G_P_6                  : IN    STD_LOGIC;
    A10_REFCLK_10G_P_7                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_0                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_1                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_2                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_3                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_4                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_5                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_6                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_7                  : IN    STD_LOGIC;

    --  XCVR serial lines
    rx_gbt                              : IN    STD_LOGIC_VECTOR(47 DOWNTO 0);
    tx_gbt                              : OUT   STD_LOGIC_VECTOR(47 DOWNTO 0);

    --  SI5344 PLL interface 
    A10_SI5344_240_P                    : IN    STD_LOGIC; -- 240 clock input from TFC connector or 40 MHz oscillator
    A10_SI5344_INTR_N                   : IN    STD_LOGIC;
    A10_SI5344_LOL_N                    : IN    STD_LOGIC; 
    A10_SI5344_LOS_XAXB_N               : IN    STD_LOGIC;
    A10_SI5344_SMB_SCL                  : INOUT STD_LOGIC;
    A10_SI5344_SMB_SDA                  : INOUT STD_LOGIC;

    --  SI5345 PLLs interface
    A10_SI5345_1_JITTER_CLOCK_P         : OUT   STD_LOGIC; -- TFC clock to jitter cleaner PLL
    A10_SI5345_1_JITTER_INTR_N          : IN    STD_LOGIC;
    A10_SI5345_1_JITTER_LOL_N           : IN    STD_LOGIC;
    A10_SI5345_1_SMB_SCL                : INOUT STD_LOGIC;
    A10_SI5345_1_SMB_SDA                : INOUT STD_LOGIC;
    A10_SI5345_2_CLK_240_P              : IN    STD_LOGIC; -- 240 MHz refclck image to matrix
    A10_SI5345_2_JITTER_CLOCK_P         : OUT   STD_LOGIC; -- TFC clock to jitter cleaner PLL
    A10_SI5345_2_JITTER_INTR_N          : IN    STD_LOGIC;
    A10_SI5345_2_JITTER_LOL_N           : IN    STD_LOGIC;
    A10_SI5345_2_SMB_SCL                : INOUT STD_LOGIC;
    A10_SI5345_2_SMB_SDA                : INOUT STD_LOGIC;

    --  SI53340 fanout
    A10_SI53340_2_CLK_40_P              : IN    STD_LOGIC; -- TFC connector or 40 MHz oscillator input

    --  SI53444 fanout
    A10_SI53344_FANOUT_CLK_P            : OUT   STD_LOGIC; -- FPGA clock to SI53344 fanout
    A10_CUSTOM_CLK_P                    : IN    STD_LOGIC; -- SI53344 fanout clock to FPGA

    --  Minipods
    A10_MP_SCL                          : INOUT STD_LOGIC;
    A10_MP_SDA                          : INOUT STD_LOGIC;
    A10_MP_INT                          : IN    STD_LOGIC;
    A10_U2_RESETL                       : OUT   STD_LOGIC; -- channels 0 to 11
    A10_U3_RESETL                       : OUT   STD_LOGIC;
    A10_U4_RESETL                       : OUT   STD_LOGIC; -- channels 12 to 23
    A10_U5_RESETL                       : OUT   STD_LOGIC;
    A10_U6_RESETL                       : OUT   STD_LOGIC; -- channels 24 to 35
    A10_U7_RESETL                       : OUT   STD_LOGIC;
    A10_U8_RESETL                       : OUT   STD_LOGIC; -- channels 36 to 47
    A10_U9_RESETL                       : OUT   STD_LOGIC;
    LT1_EN                              : OUT   STD_LOGIC; -- controls level translator for SMA/SCL/INT
    LT2_EN                              : OUT   STD_LOGIC; -- controls level translators Reset U2 to U5
    LT3_EN                              : OUT   STD_LOGIC; -- controls level translators Reset U6 to U9

    --  L2418 Current Sensors
    A10_M5FL_L2418_SPI_SDI              : IN    STD_LOGIC;
    A10_M5FL_L2418_SPI_SDO              : OUT   STD_LOGIC;
    A10_M5FL_L2418_SPI_SCK              : OUT   STD_LOGIC;
    A10_M5FL_L2418_SPI_CS0_N            : OUT   STD_LOGIC;

    --  L2498 Current Sensors
    A10_M5FL_L2498_SPI_SDI              : IN    STD_LOGIC;
    A10_M5FL_L2498_SPI_SDO              : OUT   STD_LOGIC;
    A10_M5FL_L2498_SPI_SCK              : OUT   STD_LOGIC;
    A10_M5FL_L2498_SPI_CS0_N            : OUT   STD_LOGIC;

    --  Temperature sensor
    MMC_A10_TSENSE_SMB_SCL              : INOUT STD_LOGIC;
    MMC_A10_TSENSE_SMB_SDA              : INOUT STD_LOGIC;
    A10_M5FL_M1619_ALERT_N              : IN    STD_LOGIC;
    A10_M5FL_M1619_OVTEMP_N             : IN    STD_LOGIC;

    --  SFP+
    A10_SFP1_SMB_SCL                    : INOUT STD_LOGIC;
    A10_SFP1_SMB_SDA                    : INOUT STD_LOGIC;
    A10_SFP1_RS0                        : OUT   STD_LOGIC;
    A10_SFP1_RS1                        : OUT   STD_LOGIC;
    A10_SFP1_RX_LOSS                    : IN    STD_LOGIC;
    A10_SFP1_TFC_RX_P                   : IN    STD_LOGIC;
    A10_SFP1_TFC_TX_P                   : OUT   STD_LOGIC;
    A10_SFP1_TX_DISABLE                 : OUT   STD_LOGIC;
    A10_SFP1_TX_FAULT                   : IN    STD_LOGIC;
    A10_SFP1_TX_MOD_ABS                 : OUT   STD_LOGIC;
    A10_SFP2_SMB_SCL                    : INOUT STD_LOGIC;
    A10_SFP2_SMB_SDA                    : INOUT STD_LOGIC;
    A10_SFP2_RS0                        : OUT   STD_LOGIC;
    A10_SFP2_RS1                        : OUT   STD_LOGIC;
    A10_SFP2_RX_LOSS                    : IN    STD_LOGIC;
    A10_SFP2_TFC_RX_P                   : IN    STD_LOGIC;
    A10_SFP2_TFC_TX_P                   : OUT   STD_LOGIC;
    A10_SFP2_TX_DISABLE                 : OUT   STD_LOGIC;
    A10_SFP2_TX_FAULT                   : IN    STD_LOGIC;
    A10_SFP2_TX_MOD_ABS                 : OUT   STD_LOGIC;

    --  PCIe clock fanout
    A10_SI53154_SMB_SCL                 : INOUT STD_LOGIC;
    A10_SI53154_SMB_SDA                 : INOUT STD_LOGIC;

    --  Power mezzanine interface
    A10_MMC_POWER_OFF_RQST_N            : OUT   STD_LOGIC;
    A10_MMC_SPARE                       : OUT   STD_LOGIC;
    A10_SI53340_2_CLK_SEL               : OUT   STD_LOGIC;
    A10_GP_SPARE                        : OUT   STD_LOGIC; 
    A10_MEZZ_I2C_SCL                    : INOUT STD_LOGIC;
    A10_MEZZ_I2C_SDA                    : INOUT STD_LOGIC;

    --  Reset from USB Blaster
    A10_PROC_RST_N                      : IN    STD_LOGIC; 

    --  Reset from push button through Max5
    A10_M5FL_CPU_RESET_N                : IN    STD_LOGIC; 

    --  I2C from PCIe connector 
    A10_CPCI_SMB_SCL                    : INOUT STD_LOGIC;
    A10_CPCI_SMB_SDA                    : INOUT STD_LOGIC;

    --  COLOR LEDS
    A10_LED_3C_1                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_2                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_3                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_4                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);

    --  LEDS
    A10_LED                             : OUT   STD_LOGIC_VECTOR(7 DOWNTO 0);

    --  SWITCHES
    A10_SW                              : IN    STD_LOGIC_VECTOR(7 DOWNTO 0);

    --  SMAs
    A10_SMA_CLK_IN_P                    : IN    STD_LOGIC;
    A10_SMA_CLK_OUT_P                   : OUT   STD_LOGIC;

    --  TFC connector interface
    TFC_CHANNEL_A_IN_P                  : IN    STD_LOGIC;
    TFC_CHANNEL_A_OUT_P                 : OUT   STD_LOGIC;
    TFC_CHANNEL_B_IN_P                  : IN    STD_LOGIC;
    TFC_CHANNEL_B_OUT_P                 : OUT   STD_LOGIC;
    TFC_CLK_OUT_P                       : OUT   STD_LOGIC;
    TFC_ORBIT_IN_P                      : IN    STD_LOGIC;
    TFC_ORBIT_OUT_P                     : OUT   STD_LOGIC;

    -- Transceiver calibration clock
    TRANSCEIVER_CALIBRATION_CLK         : IN    STD_LOGIC; -- reserved use

    --  USB debug interface
    A10_M10_USB_ADDR                    : IN    STD_LOGIC_VECTOR(1 DOWNTO 0);
    A10_M10_USB_DATA                    : INOUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    A10_M10_USB_CLK                     : IN    STD_LOGIC;
    A10_M10_USB_EMPTY                   : OUT   STD_LOGIC;
    A10_M10_USB_FULL                    : OUT   STD_LOGIC;
    A10_M10_USB_OE_N                    : IN    STD_LOGIC;
    A10_M10_USB_RD_N                    : IN    STD_LOGIC;
    A10_M10_USB_RESET_N                 : IN    STD_LOGIC;
    A10_M10_USB_SCL                     : INOUT STD_LOGIC;
    A10_M10_USB_SDA                     : INOUT STD_LOGIC;
    A10_M10_USB_WR_N                    : IN    STD_LOGIC;

    --  identification serial eeprom
    SERIAL_EEPROM_SCL                   : INOUT STD_LOGIC;
    SERIAL_EEPROM_SDA                   : INOUT STD_LOGIC;
    SERIAL_EEPROM_WRITE_PROTECT         : OUT   STD_LOGIC;

    --  spares
    SPARE                               : INOUT STD_LOGIC_VECTOR(7 DOWNTO 0); -- direct spares
    A10_M5FL_SPARE                      : INOUT STD_LOGIC_VECTOR(8 DOWNTO 0);

    --  general purpose internal clock
    CLK_A10_100MHZ_P                    : IN    STD_LOGIC--; -- from internal 100 MHz oscillator
);
end entity;

architecture rtl of top is

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal i2c_scl_in   : std_logic;
    signal i2c_scl_oe   : std_logic;
    signal i2c_sda_in   : std_logic;
    signal i2c_sda_oe   : std_logic;

    signal nios_clk      : std_logic;
    signal nios_rst_n    : std_logic;
    signal flash_rst_n  : std_logic;

    signal wd_rst_n     : std_logic;

    signal nios_pio_i : std_logic_vector(31 downto 0);

    signal av_pod0 : work.util.avalon_t;

begin

    e_nios_clk_hz : entity work.clkdiv
    generic map ( P => 100 * 10**6 )
    port map (
        clkout => A10_LED(0),
        rst_n => A10_SW(0),
        clk => CLK_A10_100MHZ_P--,
    );



    nios_clk <= CLK_A10_100MHZ_P;

    -- generate reset sequence for flash and nios
    i_reset_ctrl : entity work.reset_ctrl
    generic map (
        W => 2,
        N => 100 * 10**5 -- 100ms
    )
    port map (
        rstout_n(1) => flash_rst_n,
        rstout_n(0) => nios_rst_n,

        rst_n => A10_SW(0) and wd_rst_n,
        clk => nios_clk--,
    );
    A10_LED(1) <= not flash_rst_n;
    A10_LED(2) <= not nios_rst_n;

    watchdog_i : entity work.watchdog
    generic map (
        W => 4,
        N => 100 * 10**7 -- 10s
    )
    port map (
        d => nios_pio_i(3 downto 0),

        rstout_n => wd_rst_n,

        rst_n => A10_SW(0),
        clk => nios_clk--,
    );

    A10_LED(3) <= nios_pio_i(7);



    i_nios : component work.cmp.nios
    port map (
        avm_qsfp_address        => av_pod0.address(13 downto 0),
        avm_qsfp_read           => av_pod0.read,
        avm_qsfp_readdata       => av_pod0.readdata,
        avm_qsfp_write          => av_pod0.write,
        avm_qsfp_writedata      => av_pod0.writedata,
        avm_qsfp_waitrequest    => av_pod0.waitrequest,

        i2c_scl_in  => i2c_scl_in,
        i2c_scl_oe  => i2c_scl_oe,
        i2c_sda_in  => i2c_sda_in,
        i2c_sda_oe  => i2c_sda_oe,

        spi_MISO    => '-',
        spi_MOSI    => open,
        spi_SCLK    => open,
        spi_SS_n    => open,

        pio_export => nios_pio_i,

        rst_reset_n => nios_rst_n,
        clk_clk     => nios_clk--,
    );



    e_pod0 : entity work.xcvr_a10
    generic map (
        INPUT_CLOCK_FREQUENCY_g => 125000000,
        DATA_RATE_g => 5000,
        CLK_MHZ_g => 125--,
    )
    port map (
        i_tx_data   => X"03CAFEBC"
                     & X"02BABEBC"
                     & X"01DEADBC"
                     & X"00BEEFBC",
        i_tx_datak  => "0001"
                     & "0001"
                     & "0001"
                     & "0001",

        o_rx_data   => open,
        o_rx_datak  => open,

        o_tx_clkout => open,
        i_tx_clkin  => (others => A10_REFCLK_GBT_P_0),
        o_rx_clkout => open,
        i_rx_clkin  => (others => A10_REFCLK_GBT_P_0),

        o_tx_serial => tx_gbt(3 downto 0),
        i_rx_serial => rx_gbt(3 downto 0),

        i_pll_clk   => A10_REFCLK_GBT_P_0,
        i_cdr_clk   => A10_REFCLK_GBT_P_0,

        i_avs_address     => av_pod0.address(13 downto 0),
        i_avs_read        => av_pod0.read,
        o_avs_readdata    => av_pod0.readdata,
        i_avs_write       => av_pod0.write,
        i_avs_writedata   => av_pod0.writedata,
        o_avs_waitrequest => av_pod0.waitrequest,

        i_reset     => not nios_rst_n,
        i_clk       => nios_clk--,
    );

end architecture;
