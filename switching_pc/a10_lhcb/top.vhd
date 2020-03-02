library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    --  PODs
    rx_gbt                              : IN    STD_LOGIC_VECTOR(47 DOWNTO 0);
    tx_gbt                              : OUT   STD_LOGIC_VECTOR(47 DOWNTO 0);
    A10_REFCLK_GBT_P_0                  : IN    STD_LOGIC;

    --  LEDs
    A10_LED                             : OUT   STD_LOGIC_VECTOR(7 DOWNTO 0);

    --  Color LEDs
    A10_LED_3C_1                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_2                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_3                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_4                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);

    A10_SI53340_2_CLK_40_P              : in    std_logic;

    -- SI5345_1
    A10_SI5345_1_SMB_SCL                : inout std_logic;
    A10_SI5345_1_SMB_SDA                : inout std_logic;
    A10_SI5345_1_JITTER_CLOCK_P         : out   std_logic;

    -- SI5345_2
    A10_SI5345_2_SMB_SCL                : inout std_logic;
    A10_SI5345_2_SMB_SDA                : inout std_logic;
    A10_SI5345_2_JITTER_CLOCK_P         : out   std_logic;

    --  Reset from push button through Max5
    A10_M5FL_CPU_RESET_N                : IN    STD_LOGIC;

    --  general purpose internal clock
    CLK_A10_100MHZ_P                    : IN    STD_LOGIC--; -- from internal 100 MHz oscillator
);
end entity;

architecture rtl of top is

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal nios_clk, nios_reset_n : std_logic;
    signal i2c_scl_in, i2c_scl_oe, i2c_sda_in, i2c_sda_oe : std_logic;
    signal i2c_cs : std_logic_vector(31 downto 0);

    constant I2C_CS_SI5341_1_c : std_logic_vector(i2c_cs'range) := (1 => '1', others => '0');
    constant I2C_CS_SI5341_2_c : std_logic_vector(i2c_cs'range) := (2 => '1', others => '0');

    signal pod_clk, pod_reset_n : std_logic;
    signal pod_tx_clkout : std_logic_vector(47 downto 0);

    signal av_pod : work.util.avalon_t;

begin

    nios_clk <= CLK_A10_100MHZ_P;

    e_nios_reset_n : entity work.reset_sync
    port map ( o_reset_n => nios_reset_n, i_reset_n => A10_M5FL_CPU_RESET_N, i_clk => nios_clk );

    e_nios_clk_hz : entity work.clkdiv
    generic map ( P => 100 * 10**6 )
    port map (
        o_clk => A10_LED(0),
        i_reset_n => nios_reset_n,
        i_clk => nios_clk--,
    );

    i_nios : component work.cmp.nios
    port map (
        avm_pod_reset_reset_n   => pod_reset_n,
        avm_pod_clock_clk       => pod_clk,
        avm_pod_address         => av_pod.address(13 downto 0),
        avm_pod_read            => av_pod.read,
        avm_pod_readdata        => av_pod.readdata,
        avm_pod_write           => av_pod.write,
        avm_pod_writedata       => av_pod.writedata,
        avm_pod_waitrequest     => av_pod.waitrequest,

        i2c_scl_in      => i2c_scl_in,
        i2c_scl_oe      => i2c_scl_oe,
        i2c_sda_in      => i2c_sda_in,
        i2c_sda_oe      => i2c_sda_oe,
        i2c_cs_export   => i2c_cs,

        rst_reset_n => nios_reset_n,
        clk_clk     => nios_clk--,
    );

    -- i2c SCL
    i2c_scl_in <=
        '0' when ( i2c_cs = I2C_CS_SI5341_1_c and A10_SI5345_1_SMB_SCL = '0' ) else
        '0' when ( i2c_cs = I2C_CS_SI5341_2_c and A10_SI5345_2_SMB_SCL = '0' ) else
        '1';
    A10_SI5345_1_SMB_SCL <= ZERO when ( i2c_cs = I2C_CS_SI5341_1_c and i2c_scl_oe = '1' ) else 'Z';
    A10_SI5345_2_SMB_SCL <= ZERO when ( i2c_cs = I2C_CS_SI5341_2_c and i2c_scl_oe = '1' ) else 'Z';
    -- i2c SDA
    i2c_sda_in <=
        '0' when ( i2c_cs = I2C_CS_SI5341_1_c and A10_SI5345_1_SMB_SDA = '0' ) else
        '0' when ( i2c_cs = I2C_CS_SI5341_2_c and A10_SI5345_2_SMB_SDA = '0' ) else
        '1';
    A10_SI5345_1_SMB_SDA <= ZERO when ( i2c_cs = I2C_CS_SI5341_1_c and i2c_sda_oe = '1' ) else 'Z';
    A10_SI5345_2_SMB_SDA <= ZERO when ( i2c_cs = I2C_CS_SI5341_2_c and i2c_sda_oe = '1' ) else 'Z';



    -- TODO: use global 125/250 MHz clock
    pod_clk <= A10_REFCLK_GBT_P_0;

    e_pod_reset_n : entity work.reset_sync
    port map ( o_reset_n => pod_reset_n, i_reset_n => A10_M5FL_CPU_RESET_N, i_clk => pod_clk );

    e_pod_clk_hz : entity work.clkdiv
    generic map ( P => 125 * 10**6 )
    port map (
        o_clk => A10_LED(4),
        i_reset_n => pod_reset_n,
        i_clk => pod_clk--,
    );

    e_pods : entity work.xcvr_block
    generic map (
        N_XCVR_g => 1--,
    )
    port map (
        i_rx_serial => rx_gbt(5 downto 0),
        o_tx_serial => tx_gbt(5 downto 0),

        i_refclk(0) => A10_REFCLK_GBT_P_0,

        i_avs_address     => av_pod.address(13 downto 0),
        i_avs_read        => av_pod.read,
        o_avs_readdata    => av_pod.readdata,
        i_avs_write       => av_pod.write,
        i_avs_writedata   => av_pod.writedata,
        o_avs_waitrequest => av_pod.waitrequest,

        o_data   => open,
        o_datak  => open,

        i_data =>
            X"050000BC"
          & X"040000BC"
          & X"030000BC"
          & X"020000BC"
          & X"010000BC"
          & X"000000BC",
        i_datak =>
            "0001"
          & "0001"
          & "0001"
          & "0001"
          & "0001"
          & "0001",

        i_reset_n   => pod_reset_n,
        i_clk       => pod_clk--,
    );



    process(nios_clk)
    begin
    if rising_edge(nios_clk) then
    end if;
    end process;



    A10_SI5345_1_JITTER_CLOCK_P <= A10_SI53340_2_CLK_40_P;
    A10_SI5345_2_JITTER_CLOCK_P <= A10_SI53340_2_CLK_40_P;

end architecture;
