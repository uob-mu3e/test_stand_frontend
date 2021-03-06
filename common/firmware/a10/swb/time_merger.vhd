library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


-- merge packets delimited by SOP and EOP from N input streams
entity time_merger is
generic (
    g_ADDR_WIDTH    : positive := 11;
    g_NLINKS_DATA   : positive := 8;
    -- Data type: x"00" = pixel, x"01" = scifi, "10" = tiles
    DATA_TYPE : std_logic_vector(1 downto 0) := "00"--;
);
port (
    -- input streams
    i_data          : in    work.mu3e.link_array_t(g_NLINKS_DATA-1 downto 0);
    i_empty         : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_mask_n        : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_rack          : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- read ACK

    -- output stream
    o_rdata         : out   work.mu3e.link_t; -- output is hit
    i_rack          : in    std_logic;
    o_empty         : out   std_logic;

    -- counters
    o_counters      : out work.util.slv32_array_t(3 * (N_LINKS_TREE(3) + N_LINKS_TREE(2) + N_LINKS_TREE(1)) - 1 downto 0);

    i_en            : in    std_logic;
    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of time_merger is

    -- input signals
    signal data : work.mu3e.link_array_t(N_LINKS_TREE(0) - 1 downto 0);
    signal countersL0 : work.util.slv32_array_t(3 * N_LINKS_TREE(1) - 1 downto 0);
    signal empty, mask_n, rack : std_logic_vector(N_LINKS_TREE(0) - 1 downto 0) := (others => '0');

    -- layer0
    signal data0 : work.mu3e.link_array_t(N_LINKS_TREE(1) - 1 downto 0);
    signal countersL1 : work.util.slv32_array_t(3 * N_LINKS_TREE(2) - 1 downto 0);
    signal empty0, mask_n0, rack0 : std_logic_vector(N_LINKS_TREE(1) - 1 downto 0);

    -- layer1
    signal data1 : work.mu3e.link_array_t(N_LINKS_TREE(2) - 1 downto 0);
    signal countersL2 : work.util.slv32_array_t(3 * N_LINKS_TREE(3) - 1 downto 0);
    signal empty1, mask_n1, rack1 : std_logic_vector(N_LINKS_TREE(2) - 1 downto 0);

begin

    --! map input signals to always have vectors of size 8 for the first layer
    data(g_NLINKS_DATA - 1 downto 0)    <= i_data(g_NLINKS_DATA - 1 downto 0);
    empty(g_NLINKS_DATA - 1 downto 0)   <= i_empty(g_NLINKS_DATA - 1 downto 0);
    mask_n(g_NLINKS_DATA - 1 downto 0)  <= i_mask_n(g_NLINKS_DATA - 1 downto 0);
    o_rack(g_NLINKS_DATA - 1 downto 0)  <= rack(g_NLINKS_DATA - 1 downto 0);

    --! map counters
    o_counters(3 * (N_LINKS_TREE(1)) - 1 downto 0) <= countersL0;
    o_counters(3 * (N_LINKS_TREE(2)+N_LINKS_TREE(1)) - 1 downto 3*N_LINKS_TREE(1)) <= countersL1;
    o_counters(3 * (N_LINKS_TREE(3)+N_LINKS_TREE(2)+N_LINKS_TREE(1)) - 1 downto 3*(N_LINKS_TREE(2)+N_LINKS_TREE(1))) <= countersL2;

    --! setup tree from layer0 8-4, layer1 4-2, layer2 2-1
    layer0 : entity work.time_merger_tree
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH, N_LINKS_IN => N_LINKS_TREE(0), N_LINKS_OUT => N_LINKS_TREE(1), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input data stream
        i_data          => data,
        i_empty         => empty,
        i_mask_n        => mask_n,
        o_rack          => rack,

        -- output data stream
        o_data          => data0,
        o_empty         => empty0,
        o_mask_n        => mask_n0,
        i_rack          => rack0,

        -- counters
        o_counters      => countersL0,

        i_en            => i_en,
        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer1 : entity work.time_merger_tree
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH, N_LINKS_IN => N_LINKS_TREE(1), N_LINKS_OUT => N_LINKS_TREE(2), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input data stream
        i_data          => data0,
        i_empty         => empty0,
        i_mask_n        => mask_n0,
        o_rack          => rack0,

        -- output data stream
        o_data          => data1,
        o_empty         => empty1,
        o_mask_n        => mask_n1,
        i_rack          => rack1,

        -- counters
        o_counters      => countersL1,

        i_en            => i_en,
        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer2 : entity work.time_merger_tree
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH, N_LINKS_IN => N_LINKS_TREE(2), N_LINKS_OUT => N_LINKS_TREE(3), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input data stream
        i_data          => data1,
        i_empty         => empty1,
        i_mask_n        => mask_n1,
        o_rack          => rack1,

        -- output data stream
        o_data(0)       => o_rdata,
        o_empty(0)      => o_empty,
        o_mask_n        => open,
        i_rack(0)       => i_rack,

        -- counters
        o_counters      => countersL2,

        i_en            => i_en,
        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

end architecture;
