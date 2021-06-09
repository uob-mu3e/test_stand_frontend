--
-- author : Alexandr Kozlinskiy
-- date : 2021-06-08
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity xcvr_fifo is
generic (
    g_DATA : std_logic_vector := X"000000BC";
    g_DATAK : std_logic_vector := "0001";
    g_W : positive := 32;
    g_FIFO_ADDR_WIDTH : positive := 4--;
);
port (
    i_xcvr_rx_data      : in    std_logic_vector(g_W-1 downto 0);
    i_xcvr_rx_datak     : in    std_logic_vector(g_W/8-1 downto 0);
    o_xcvr_tx_data      : out   std_logic_vector(g_W-1 downto 0);
    o_xcvr_tx_datak     : out   std_logic_vector(g_W/8-1 downto 0);
    i_xcvr_clk          : in    std_logic;

    o_rx_data           : out   std_logic_vector(g_W-1 downto 0);
    o_rx_datak          : out   std_logic_vector(g_W/8-1 downto 0);
    i_tx_data           : in    std_logic_vector(g_W-1 downto 0);
    i_tx_datak          : in    std_logic_vector(g_W/8-1 downto 0);
    i_clk               : in    std_logic;

    i_reset_n           : in    std_logic--;
);
end entity;

architecture arch of xcvr_fifo is

    signal rx_data, tx_data : std_logic_vector(g_W-1 downto 0);
    signal rx_datak, tx_datak : std_logic_vector(g_W/8+g_W-1 downto g_W);
    signal rx_empty, tx_empty : std_logic;

begin

    assert ( g_W mod 8 = 0 ) report "" severity failure;
    assert ( g_DATA'length = g_W ) report "" severity failure;
    assert ( g_DATAK'length = g_W/8 ) report "" severity failure;

    e_rx_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => g_FIFO_ADDR_WIDTH,
        g_DATA_WIDTH => rx_data'length + rx_datak'length,
        DEVICE_FAMILY => "Arria 10"--,
    )
    port map (
        i_wdata(rx_data'range) => i_xcvr_rx_data,
        i_wdata(rx_datak'range) => i_xcvr_rx_datak,
        i_we => '1',
        o_wfull => open,
        i_wclk => i_xcvr_clk,

        o_rdata(rx_data'range) => rx_data,
        o_rdata(rx_datak'range) => rx_datak,
        i_re => '1',
        o_rempty => rx_empty,
        i_rclk => i_clk,

        i_reset_n => i_reset_n--,
    );

    o_rx_data <= rx_data when ( rx_empty = '0' ) else g_DATA;
    o_rx_datak <= rx_datak when ( rx_empty = '0' ) else g_DATAK;

    e_tx_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => g_FIFO_ADDR_WIDTH,
        g_DATA_WIDTH => tx_data'length + tx_datak'length,
        DEVICE_FAMILY => "Arria 10"--,
    )
    port map (
        i_wdata(tx_data'range) => i_tx_data,
        i_wdata(tx_datak'range) => i_tx_datak,
        i_we => '1',
        o_wfull => open,
        i_wclk => i_clk,

        o_rdata(tx_data'range) => tx_data,
        o_rdata(tx_datak'range) => tx_datak,
        i_re => '1',
        o_rempty => tx_empty,
        i_rclk => i_xcvr_clk,

        i_reset_n => i_reset_n--,
    );

    o_xcvr_tx_data <= tx_data when ( tx_empty = '0' ) else g_DATA;
    o_xcvr_tx_datak <= tx_datak when ( tx_empty = '0' ) else g_DATAK;

end architecture;
