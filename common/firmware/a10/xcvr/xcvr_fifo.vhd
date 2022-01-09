--
-- author : Alexandr Kozlinskiy
-- date : 2021-06-08
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity xcvr_fifo is
generic (
    g_W : positive := 32;
    g_IDLE_DATA : std_logic_vector := X"000000BC";
    g_IDLE_DATAK : std_logic_vector := "0001";
    g_FIFO_ADDR_WIDTH : positive := 4--;
);
port (
    i_xcvr_rx_data      : in    std_logic_vector(g_W-1 downto 0);
    i_xcvr_rx_datak     : in    std_logic_vector(g_W/8-1 downto 0);
    o_xcvr_tx_data      : out   std_logic_vector(g_W-1 downto 0);
    o_xcvr_tx_datak     : out   std_logic_vector(g_W/8-1 downto 0);
    o_xcvr_rx_wfull     : out   std_logic;
    i_xcvr_clk          : in    std_logic;

    o_rx_data           : out   std_logic_vector(g_W-1 downto 0);
    o_rx_datak          : out   std_logic_vector(g_W/8-1 downto 0);
    i_tx_data           : in    std_logic_vector(g_W-1 downto 0);
    i_tx_datak          : in    std_logic_vector(g_W/8-1 downto 0);
    o_tx_wfull          : out   std_logic;
    i_clk               : in    std_logic;

    i_reset_n           : in    std_logic--;
);
end entity;

architecture arch of xcvr_fifo is

    signal rx_data, tx_data, xcvr_tx_data : std_logic_vector(g_W-1 downto 0);
    signal rx_datak, tx_datak, xcvr_tx_datak : std_logic_vector(g_W/8+g_W-1 downto g_W);
    signal rx_rempty, xcvr_tx_rempty : std_logic;

begin

    assert ( true
        and ( g_W mod 8 = 0 )
        and ( g_IDLE_DATA'length = g_W )
        and ( g_IDLE_DATAK'length = g_W/8 )
    ) report "ERROR: xcvr_fifo"
        & ", g_W = " & integer'image(g_W)
        & ", g_IDLE_DATA'length = " & integer'image(g_IDLE_DATA'length)
        & ", g_IDLE_DATAK'length = " & integer'image(g_IDLE_DATAK'length)
    severity failure;

    e_rx_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => g_FIFO_ADDR_WIDTH,
        g_DATA_WIDTH => i_xcvr_rx_data'length + i_xcvr_rx_datak'length--,
    )
    port map (
        i_wdata(rx_data'range) => i_xcvr_rx_data,
        i_wdata(rx_datak'range) => i_xcvr_rx_datak,
        -- write non IDLE data/k
        i_we => not work.util.to_std_logic(i_xcvr_rx_data = g_IDLE_DATA and i_xcvr_rx_datak = g_IDLE_DATAK),
        o_wfull => o_xcvr_rx_wfull,
        i_wclk => i_xcvr_clk,

        o_rdata(rx_data'range) => rx_data,
        o_rdata(rx_datak'range) => rx_datak,
        -- always read
        i_rack => not rx_rempty,
        o_rempty => rx_rempty,
        i_rclk => i_clk,

        i_reset_n => i_reset_n--,
    );

    -- insert IDLE when rempty
    o_rx_data <= rx_data when ( rx_rempty = '0' ) else g_IDLE_DATA;
    o_rx_datak <= rx_datak when ( rx_rempty = '0' ) else g_IDLE_DATAk;

    process(i_clk)
    begin
    if rising_edge(i_clk) then
        tx_data <= i_tx_data;
        tx_datak <= i_tx_datak;
    end if;
    end process;

    e_tx_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => g_FIFO_ADDR_WIDTH,
        g_DATA_WIDTH => tx_data'length + tx_datak'length--,
    )
    port map (
        i_wdata(tx_data'range) => tx_data,
        i_wdata(tx_datak'range) => tx_datak,
        -- write non IDLE data/k
        i_we => not work.util.to_std_logic(tx_data = g_IDLE_DATA and tx_datak = g_IDLE_DATAK),
        o_wfull => o_tx_wfull,
        i_wclk => i_clk,

        o_rdata(xcvr_tx_data'range) => xcvr_tx_data,
        o_rdata(xcvr_tx_datak'range) => xcvr_tx_datak,
        -- always read
        i_rack => not xcvr_tx_rempty,
        o_rempty => xcvr_tx_rempty,
        i_rclk => i_xcvr_clk,

        i_reset_n => i_reset_n--,
    );

    -- insert IDLE when rempty
    o_xcvr_tx_data <= xcvr_tx_data when ( xcvr_tx_rempty = '0' ) else g_IDLE_DATA;
    o_xcvr_tx_datak <= xcvr_tx_datak when ( xcvr_tx_rempty = '0' ) else g_IDLE_DATAk;

end architecture;
