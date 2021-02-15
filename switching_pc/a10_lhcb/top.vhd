library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    -- LEDs
    A10_LED                             : OUT   STD_LOGIC_VECTOR(7 DOWNTO 0);

    -- Color LEDs
    A10_LED_3C_1                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_2                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_3                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_4                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);

    



    -- POD
    rx_gbt                              : IN    STD_LOGIC_VECTOR(47 DOWNTO 0);
    tx_gbt                              : OUT   STD_LOGIC_VECTOR(47 DOWNTO 0);
    A10_REFCLK_GBT_P_0                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_1                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_2                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_3                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_4                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_5                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_6                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_7                  : IN    STD_LOGIC;



    -- PCIe 0
    i_pcie0_rx                          : in    std_logic_vector(3 downto 0);
    o_pcie0_tx                          : out   std_logic_vector(3 downto 0);
    i_pcie0_perst_n                     : in    std_logic;
    i_pcie0_refclk                      : in    std_logic;

    -- PCIe 1
    i_pcie1_rx                          : in    std_logic_vector(3 downto 0);
    o_pcie1_tx                          : out   std_logic_vector(3 downto 0);
    i_pcie1_perst_n                     : in    std_logic;
    i_pcie1_refclk                      : in    std_logic;



    -- SI5345_1
    A10_SI5345_1_SMB_SCL                : inout std_logic;
    A10_SI5345_1_SMB_SDA                : inout std_logic;
    A10_SI5345_1_JITTER_CLOCK_P         : out   std_logic;

    -- SI5345_2
    A10_SI5345_2_SMB_SCL                : inout std_logic;
    A10_SI5345_2_SMB_SDA                : inout std_logic;
    A10_SI5345_2_JITTER_CLOCK_P         : out   std_logic;



    -- Reset from push button through Max5
    A10_M5FL_CPU_RESET_N                : IN    STD_LOGIC;

    -- general purpose internal clock
    CLK_A10_100MHZ_P                    : IN    STD_LOGIC--; -- from internal 100 MHz oscillator
);
end entity;

architecture arch of top is

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal led : std_logic_vector(7 downto 0) := (others => '0');



    signal nios_clk, nios_reset_n : std_logic;
    signal i2c_scl_in, i2c_scl_oe, i2c_sda_in, i2c_sda_oe : std_logic;
    signal i2c_cs : std_logic_vector(31 downto 0);

    constant I2C_CS_SI5341_1_c : std_logic_vector(i2c_cs'range) := (1 => '1', others => '0');
    constant I2C_CS_SI5341_2_c : std_logic_vector(i2c_cs'range) := (2 => '1', others => '0');



    signal pod_clk, pod_reset_n : std_logic;
    signal pod_tx_clkout : std_logic_vector(47 downto 0);

    signal av_pod : work.util.avalon_t;

    signal pod_rx_data : work.util.slv32_array_t(47 downto 0);
    signal pod_tx_data : work.util.slv32_array_t(47 downto 0) := (
        others => X"000000BC"--,
    );
    signal pod_rx_datak : work.util.slv4_array_t(47 downto 0);
    signal pod_tx_datak : work.util.slv4_array_t(47 downto 0) := (
        others => "0001"--,
    );



    signal pcie_rx, pcie_tx : std_logic_vector(7 downto 0);

    signal pcie_wregs, pcie_rregs : work.pcie_components.reg32array;

begin

    A10_LED <= not led;

    A10_SI5345_1_JITTER_CLOCK_P <= CLK_A10_100MHZ_P;
    A10_SI5345_2_JITTER_CLOCK_P <= CLK_A10_100MHZ_P;



    nios_clk <= CLK_A10_100MHZ_P;

    e_nios_reset_n : entity work.reset_sync
    port map ( o_reset_n => nios_reset_n, i_reset_n => A10_M5FL_CPU_RESET_N, i_clk => nios_clk );

    e_nios_clk_hz : entity work.clkdiv
    generic map ( P => 100 * 10**6 )
    port map (
        o_clk => led(0),
        i_reset_n => nios_reset_n,
        i_clk => nios_clk--,
    );

    i_nios : component work.cmp.nios
    port map (
        avm_pod_reset_reset_n   => nios_reset_n,
        avm_pod_clock_clk       => nios_clk,
        avm_pod_address         => av_pod.address(16 downto 0),
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
        o_clk => led(1),
        i_reset_n => pod_reset_n,
        i_clk => pod_clk--,
    );

    e_pods : entity work.xcvr_block
    generic map (
        N_XCVR_g => 8--,
    )
    port map (
        i_rx_serial => rx_gbt,
        o_tx_serial => tx_gbt,

        i_refclk    => A10_REFCLK_GBT_P_7 & A10_REFCLK_GBT_P_6 & A10_REFCLK_GBT_P_5 & A10_REFCLK_GBT_P_4 & A10_REFCLK_GBT_P_3 & A10_REFCLK_GBT_P_2 & A10_REFCLK_GBT_P_1 & A10_REFCLK_GBT_P_0,

        o_rx_data   => pod_rx_data,
        o_rx_datak  => pod_rx_datak,
        i_tx_data   => pod_tx_data,
        i_tx_datak  => pod_tx_datak,

        i_avs_address     => av_pod.address(16 downto 0),
        i_avs_read        => av_pod.read,
        o_avs_readdata    => av_pod.readdata,
        i_avs_write       => av_pod.write,
        i_avs_writedata   => av_pod.writedata,
        o_avs_waitrequest => av_pod.waitrequest,

        i_reset_n   => nios_reset_n,
        i_clk       => nios_clk--,
    );



    e_pcie_block_0 : entity work.pcie_block
    generic map (
        DMAMEMWRITEADDRSIZE     => 11,
        DMAMEMREADADDRSIZE      => 11,
        DMAMEMWRITEWIDTH        => 256,
        g_PCIE_X => 4--,
    )
    port map (
        local_rstn              => '1',
        appl_rstn               => '1',
        refclk                  => i_pcie0_refclk,
        pcie_fastclk_out        => open,

        pcie_rx_p               => i_pcie0_rx,
        pcie_tx_p               => o_pcie0_tx,
        pcie_refclk_p           => i_pcie0_refclk,
        pcie_perstn             => i_pcie0_perst_n,
        pcie_smbclk             => '0',
        pcie_smbdat             => '0',
        pcie_waken              => open,

        writeregs               => pcie_wregs,
        regwritten              => open,
        readregs                => pcie_rregs,

        writememclk             => CLK_A10_100MHZ_P,
        readmemclk              => CLK_A10_100MHZ_P,
        dmamemclk               => CLK_A10_100MHZ_P,
        dma2memclk              => CLK_A10_100MHZ_P
    );

    e_pcie_block_1 : entity work.pcie_block
    generic map (
        DMAMEMWRITEADDRSIZE     => 11,
        DMAMEMREADADDRSIZE      => 11,
        DMAMEMWRITEWIDTH        => 256,
        g_PCIE_X => 4--,
    )
    port map (
        local_rstn              => '1',
        appl_rstn               => '1',
        refclk                  => i_pcie1_refclk,
        pcie_fastclk_out        => open,

        pcie_rx_p               => i_pcie1_rx,
        pcie_tx_p               => o_pcie1_tx,
        pcie_refclk_p           => i_pcie1_refclk,
        pcie_perstn             => i_pcie1_perst_n,
        pcie_smbclk             => '0',
        pcie_smbdat             => '0',
        pcie_waken              => open,

        writeregs               => pcie_wregs,
        regwritten              => open,
        readregs                => pcie_rregs,

        writememclk             => CLK_A10_100MHZ_P,
        readmemclk              => CLK_A10_100MHZ_P,
        dmamemclk               => CLK_A10_100MHZ_P,
        dma2memclk              => CLK_A10_100MHZ_P
    );


    process(nios_clk)
    begin
    if rising_edge(nios_clk) then
    end if;
    end process;

end architecture;
