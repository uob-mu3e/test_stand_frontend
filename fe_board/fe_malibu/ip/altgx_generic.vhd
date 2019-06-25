library ieee;
use ieee.std_logic_1164.all;

entity altgx_generic is
generic (
    NUMBER_OF_CHANNELS_g : positive := 4;
    CHANNEL_WIDTH_g : positive := 32;
    INPUT_CLOCK_FREQUENCY_g : positive := 125000000;
    DATA_RATE_g : positive := 5000--;
);
port (
    cal_blk_clk                 : in    std_logic;
    pll_inclk                   : in    std_logic;
    pll_powerdown               : in    std_logic_vector(0 downto 0);
    reconfig_clk                : in    std_logic;
    reconfig_togxb              : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_analogreset              : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_coreclk                  : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_datain                   : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_digitalreset             : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_enapatternalign          : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_seriallpbken             : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    tx_coreclk                  : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    tx_ctrlenable               : in    std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    tx_datain                   : in    std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    tx_digitalreset             : in    std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    pll_locked                  : out   std_logic_vector(0 downto 0);
    reconfig_fromgxb            : out   std_logic_vector(16 downto 0);
    rx_clkout                   : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_ctrldetect               : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    rx_dataout                  : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g-1 downto 0);
    rx_disperr                  : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    rx_errdetect                : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    rx_freqlocked               : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_patterndetect            : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    rx_phase_comp_fifo_error    : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_pll_locked               : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    rx_syncstatus               : out   std_logic_vector(NUMBER_OF_CHANNELS_g*CHANNEL_WIDTH_g/8-1 downto 0);
    tx_clkout                   : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    tx_dataout                  : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0);
    tx_phase_comp_fifo_error    : out   std_logic_vector(NUMBER_OF_CHANNELS_g-1 downto 0)--;
);
end entity;

architecture arch of altgx_generic is

begin

g_ip_altgx_4ch_32bit_125MHz_5000Mbps : if (
    NUMBER_OF_CHANNELS_g = 4 and
    CHANNEL_WIDTH_g = 32 and
    INPUT_CLOCK_FREQUENCY_g = 125000000 and
    DATA_RATE_g = 5000
) generate
e_ip_altgx_4ch_32bit_125MHz_5000Mbps : entity work.ip_altgx_4ch_32bit_125MHz_5000Mbps
port map (
    cal_blk_clk                 =>  cal_blk_clk,
    pll_inclk                   =>  pll_inclk,
    pll_powerdown               =>  pll_powerdown,
    reconfig_clk                =>  reconfig_clk,
    reconfig_togxb              =>  reconfig_togxb,
    rx_analogreset              =>  rx_analogreset,
    rx_coreclk                  =>  rx_coreclk,
    rx_datain                   =>  rx_datain,
    rx_digitalreset             =>  rx_digitalreset,
    rx_enapatternalign          =>  rx_enapatternalign,
    rx_seriallpbken             =>  rx_seriallpbken,
    tx_coreclk                  =>  tx_coreclk,
    tx_ctrlenable               =>  tx_ctrlenable,
    tx_datain                   =>  tx_datain,
    tx_digitalreset             =>  tx_digitalreset,
    pll_locked                  =>  pll_locked,
    reconfig_fromgxb            =>  reconfig_fromgxb,
    rx_clkout                   =>  rx_clkout,
    rx_ctrldetect               =>  rx_ctrldetect,
    rx_dataout                  =>  rx_dataout,
    rx_disperr                  =>  rx_disperr,
    rx_errdetect                =>  rx_errdetect,
    rx_freqlocked               =>  rx_freqlocked,
    rx_patterndetect            =>  rx_patterndetect,
    rx_phase_comp_fifo_error    =>  rx_phase_comp_fifo_error,
    rx_pll_locked               =>  rx_pll_locked,
    rx_syncstatus               =>  rx_syncstatus,
    tx_clkout                   =>  tx_clkout,
    tx_dataout                  =>  tx_dataout,
    tx_phase_comp_fifo_error    =>  tx_phase_comp_fifo_error--,
);
end generate;

g_ip_altgx_4ch_32bit_125MHz_6250Mbps : if (
    NUMBER_OF_CHANNELS_g = 4 and
    CHANNEL_WIDTH_g = 32 and
    INPUT_CLOCK_FREQUENCY_g = 125000000 and
    DATA_RATE_g = 6250
) generate
e_ip_altgx_4ch_32bit_125MHz_6250Mbps : entity work.ip_altgx_4ch_32bit_125MHz_6250Mbps
port map (
    cal_blk_clk                 =>  cal_blk_clk,
    pll_inclk                   =>  pll_inclk,
    pll_powerdown               =>  pll_powerdown,
    reconfig_clk                =>  reconfig_clk,
    reconfig_togxb              =>  reconfig_togxb,
    rx_analogreset              =>  rx_analogreset,
    rx_coreclk                  =>  rx_coreclk,
    rx_datain                   =>  rx_datain,
    rx_digitalreset             =>  rx_digitalreset,
    rx_enapatternalign          =>  rx_enapatternalign,
    rx_seriallpbken             =>  rx_seriallpbken,
    tx_coreclk                  =>  tx_coreclk,
    tx_ctrlenable               =>  tx_ctrlenable,
    tx_datain                   =>  tx_datain,
    tx_digitalreset             =>  tx_digitalreset,
    pll_locked                  =>  pll_locked,
    reconfig_fromgxb            =>  reconfig_fromgxb,
    rx_clkout                   =>  rx_clkout,
    rx_ctrldetect               =>  rx_ctrldetect,
    rx_dataout                  =>  rx_dataout,
    rx_disperr                  =>  rx_disperr,
    rx_errdetect                =>  rx_errdetect,
    rx_freqlocked               =>  rx_freqlocked,
    rx_patterndetect            =>  rx_patterndetect,
    rx_phase_comp_fifo_error    =>  rx_phase_comp_fifo_error,
    rx_pll_locked               =>  rx_pll_locked,
    rx_syncstatus               =>  rx_syncstatus,
    tx_clkout                   =>  tx_clkout,
    tx_dataout                  =>  tx_dataout,
    tx_phase_comp_fifo_error    =>  tx_phase_comp_fifo_error--,
);
end generate;

g_ip_altgx_4ch_32bit_156250KHz_6250Mbps : if (
    NUMBER_OF_CHANNELS_g = 4 and
    CHANNEL_WIDTH_g = 32 and
    INPUT_CLOCK_FREQUENCY_g = 156250000 and
    DATA_RATE_g = 6250
) generate
g_ip_altgx_4ch_32bit_156250KHz_6250Mbps : entity work.ip_altgx_4ch_32bit_156250KHz_6250Mbps
port map (
    cal_blk_clk                 =>  cal_blk_clk,
    pll_inclk                   =>  pll_inclk,
    pll_powerdown               =>  pll_powerdown,
    reconfig_clk                =>  reconfig_clk,
    reconfig_togxb              =>  reconfig_togxb,
    rx_analogreset              =>  rx_analogreset,
    rx_coreclk                  =>  rx_coreclk,
    rx_datain                   =>  rx_datain,
    rx_digitalreset             =>  rx_digitalreset,
    rx_enapatternalign          =>  rx_enapatternalign,
    rx_seriallpbken             =>  rx_seriallpbken,
    tx_coreclk                  =>  tx_coreclk,
    tx_ctrlenable               =>  tx_ctrlenable,
    tx_datain                   =>  tx_datain,
    tx_digitalreset             =>  tx_digitalreset,
    pll_locked                  =>  pll_locked,
    reconfig_fromgxb            =>  reconfig_fromgxb,
    rx_clkout                   =>  rx_clkout,
    rx_ctrldetect               =>  rx_ctrldetect,
    rx_dataout                  =>  rx_dataout,
    rx_disperr                  =>  rx_disperr,
    rx_errdetect                =>  rx_errdetect,
    rx_freqlocked               =>  rx_freqlocked,
    rx_patterndetect            =>  rx_patterndetect,
    rx_phase_comp_fifo_error    =>  rx_phase_comp_fifo_error,
    rx_pll_locked               =>  rx_pll_locked,
    rx_syncstatus               =>  rx_syncstatus,
    tx_clkout                   =>  tx_clkout,
    tx_dataout                  =>  tx_dataout,
    tx_phase_comp_fifo_error    =>  tx_phase_comp_fifo_error--,
);
end generate;

g_ip_altgx_4ch_8bit_125MHz_1250Mbps : if (
    NUMBER_OF_CHANNELS_g = 4 and
    CHANNEL_WIDTH_g = 8 and
    INPUT_CLOCK_FREQUENCY_g = 125000000 and
    DATA_RATE_g = 5000
) generate
e_ip_altgx_4ch_8bit_125MHz_1250Mbps : entity work.ip_altgx_4ch_8bit_125MHz_1250Mbps
port map (
    cal_blk_clk                 =>  cal_blk_clk,
    pll_inclk                   =>  pll_inclk,
    pll_powerdown               =>  pll_powerdown,
    reconfig_clk                =>  reconfig_clk,
    reconfig_togxb              =>  reconfig_togxb,
    rx_analogreset              =>  rx_analogreset,
    rx_coreclk                  =>  rx_coreclk,
    rx_datain                   =>  rx_datain,
    rx_digitalreset             =>  rx_digitalreset,
    rx_enapatternalign          =>  rx_enapatternalign,
    rx_seriallpbken             =>  rx_seriallpbken,
    tx_coreclk                  =>  tx_coreclk,
    tx_ctrlenable               =>  tx_ctrlenable,
    tx_datain                   =>  tx_datain,
    tx_digitalreset             =>  tx_digitalreset,
    pll_locked                  =>  pll_locked,
    reconfig_fromgxb            =>  reconfig_fromgxb,
    rx_clkout                   =>  rx_clkout,
    rx_ctrldetect               =>  rx_ctrldetect,
    rx_dataout                  =>  rx_dataout,
    rx_disperr                  =>  rx_disperr,
    rx_errdetect                =>  rx_errdetect,
    rx_freqlocked               =>  rx_freqlocked,
    rx_patterndetect            =>  rx_patterndetect,
    rx_phase_comp_fifo_error    =>  rx_phase_comp_fifo_error,
    rx_pll_locked               =>  rx_pll_locked,
    rx_syncstatus               =>  rx_syncstatus,
    tx_clkout                   =>  tx_clkout,
    tx_dataout                  =>  tx_dataout,
    tx_phase_comp_fifo_error    =>  tx_phase_comp_fifo_error--,
);
end generate;

end architecture;
