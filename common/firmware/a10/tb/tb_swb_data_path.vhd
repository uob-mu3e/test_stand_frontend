library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.dataflow_components.all;
use work.pcie_components.all;
use work.mudaq_registers.all;

entity tb_swb_data_path is
end entity;

architecture arch of tb_swb_data_path is

    constant CLK_MHZ : real := 1000.0; -- MHz
    constant g_NLINKS_TOTL : integer := 64;
    constant g_NLINKS_FARM : integer := 8;
    constant g_NLINKS_DATA : integer := 12;

    signal clk, clk_fast, reset_n : std_logic := '0';
    --! data link signals
    signal rx : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_k : work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);

    signal i_writeregs_156  : reg32array;
    signal i_writeregs_250  : reg32array;


begin

    clk     <= not clk after (0.5 us / CLK_MHZ);
    clk_fast<= not clk_fast after (0.1 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);

    --! SWB Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_data_path : entity work.swb_data_path
    generic map (
        g_NLINKS_TOTL           => g_NLINKS_TOTL,
        g_NLINKS_FARM           => g_NLINKS_FARM,
        g_NLINKS_DATA           => g_NLINKS_DATA,
        LINK_FIFO_ADDR_WIDTH    => 10,
        TREE_w                  => 10,
        TREE_r                  => 10,
        SWB_ID                  => x"01",
        -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
        DATA_TYPE               => x"01"--;
    )
    port map(
        i_clk_156        => clk,
        i_clk_250        => clk_fast,
        
        i_reset_n_156    => reset_n,
        i_reset_n_250    => reset_n,

        i_resets_n_156   => i_resets_n_156,
        i_resets_n_250   => i_resets_n_250,
        
        i_rx             => i_rx,
        i_rx_k           => i_rx_k,
        i_rmask_n        => x"0000000000" & i_writeregs_250(SWB_LINK_MASK_PIXEL_REGISTER_W),

        i_writeregs_156  => i_writeregs_156,
        i_writeregs_250  => i_writeregs_250,

        o_counter        => open,

        i_dmamemhalffull => '0',
        
        o_farm_data      => open,
        o_farm_datak     => open,
        o_fram_wen       => open,

        o_dma_wren       => open,
        o_dma_done       => open,
        o_endofevent     => open,
        o_dma_data       => open--;
    );

end architecture;
