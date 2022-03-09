library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


-- merge packets delimited by SOP and EOP from N input streams
entity time_merger_v4 is
generic (
    g_ADDR_WIDTH    : positive := 11;
    g_NLINKS_DATA   : positive := 8;
    -- Data type: x"00" = pixel, x"01" = scifi, "10" = tiles
    DATA_TYPE : std_logic_vector(1 downto 0) := "00"--;
);
port (
    -- input streams
    i_data          : in    work.util.slv32_array_t(g_NLINKS_DATA - 1 downto 0);
    i_sop           : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- start of packet (SOP)
    i_eop           : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- end of packet (EOP)
    i_shop          : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- sub header of packet (SHOP)
    i_hit           : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- data is hit
    i_t0            : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- data is t0 
    i_t1            : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- data is t1
    i_empty         : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_mask_n        : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_rack          : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- read ACK

    -- output stream
    o_wdata         : out   std_logic_vector(31 downto 0); -- output is hit
    o_wsop          : out   std_logic; -- SOP
    o_weop          : out   std_logic; -- EOP
    o_t0            : out   std_logic;
    o_t1            : out   std_logic;
    i_rack          : in    std_logic;
    o_empty         : out   std_logic;

    -- error outputs
    o_error         : out   std_logic;

    i_en            : in    std_logic;
    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of time_merger_v4 is

    -- input signals
    signal data : work.util.slv32_array_t(N_LINKS_TREE(0) - 1 downto 0) := (others => (others => '0'));
    signal shop, sop, eop, hit, t0, t1, empty, mask_n, rack, error : std_logic_vector(N_LINKS_TREE(0) - 1 downto 0) := (others => '0');

    -- layer0
    signal data0 : work.util.slv32_array_t(N_LINKS_TREE(1) - 1 downto 0);
    signal shop0, sop0, eop0, hit0, t00, t10, empty0, mask_n0, rack0, error0 : std_logic_vector(N_LINKS_TREE(1) - 1 downto 0);

    -- layer1
    signal data1 : work.util.slv32_array_t(N_LINKS_TREE(2) - 1 downto 0);
    signal shop1, sop1, eop1, hit1, t01, t11, empty1, mask_n1, rack1, error1 : std_logic_vector(N_LINKS_TREE(2) - 1 downto 0);

begin

    --! map input signals to always have vectors of size 8 for the first layer
    data(g_NLINKS_DATA - 1 downto 0)    <= i_data(g_NLINKS_DATA - 1 downto 0);
    shop(g_NLINKS_DATA - 1 downto 0)    <= i_shop(g_NLINKS_DATA - 1 downto 0);
    sop(g_NLINKS_DATA - 1 downto 0)     <= i_sop(g_NLINKS_DATA - 1 downto 0);
    eop(g_NLINKS_DATA - 1 downto 0)     <= i_eop(g_NLINKS_DATA - 1 downto 0);
    hit(g_NLINKS_DATA - 1 downto 0)     <= i_hit(g_NLINKS_DATA - 1 downto 0);
    t0(g_NLINKS_DATA - 1 downto 0)      <= i_t0(g_NLINKS_DATA - 1 downto 0);
    t1(g_NLINKS_DATA - 1 downto 0)      <= i_t1(g_NLINKS_DATA - 1 downto 0);
    empty(g_NLINKS_DATA - 1 downto 0)   <= i_empty(g_NLINKS_DATA - 1 downto 0);
    mask_n(g_NLINKS_DATA - 1 downto 0)  <= i_mask_n(g_NLINKS_DATA - 1 downto 0);
    o_rack(g_NLINKS_DATA - 1 downto 0)  <= rack(g_NLINKS_DATA - 1 downto 0);
        

    --! setup tree from layer0 8-4, layer1 4-2, layer2 2-1
    layer0 : entity work.time_merger_tree_fifo_32_v3
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH, N_LINKS_IN => N_LINKS_TREE(0), N_LINKS_OUT => N_LINKS_TREE(1), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input data stream
        i_data          => data,
        i_shop          => shop,
        i_sop           => sop,
        i_eop           => eop,
        i_hit           => hit,
        i_t0            => t0,
        i_t1            => t1,
        i_empty         => empty,
        i_mask_n        => mask_n,
        o_rack          => rack,
        i_error         => (others => '0'),

        -- output data stream
        o_data          => data0,
        o_shop          => shop0,
        o_sop           => sop0,
        o_eop           => eop0,
        o_hit           => hit0,
        o_t0            => t00,
        o_t1            => t10,
        o_empty         => empty0,
        o_mask_n        => mask_n0,
        i_rack          => rack0,
        o_error         => error0,

        i_en            => i_en,
        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer1 : entity work.time_merger_tree_fifo_32_v3
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH, N_LINKS_IN => N_LINKS_TREE(1), N_LINKS_OUT => N_LINKS_TREE(2), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input data stream
        i_data          => data0,
        i_shop          => shop0,
        i_sop           => sop0,
        i_eop           => eop0,
        i_hit           => hit0,
        i_t0            => t00,
        i_t1            => t10,
        i_empty         => empty0,
        i_mask_n        => mask_n0,
        o_rack          => rack0,
        i_error         => error0,

        -- output data stream
        o_data          => data1,
        o_shop          => shop1,
        o_sop           => sop1,
        o_eop           => eop1,
        o_hit           => hit1,
        o_t0            => t01,
        o_t1            => t11,
        o_empty         => empty1,
        o_mask_n        => mask_n1,
        i_rack          => rack1,
        o_error         => error1,

        i_en            => i_en,
        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer2 : entity work.time_merger_tree_fifo_32_v3
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH, N_LINKS_IN => N_LINKS_TREE(2), N_LINKS_OUT => N_LINKS_TREE(3), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input data stream
        i_data          => data1,
        i_shop          => shop1,
        i_sop           => sop1,
        i_eop           => eop1,
        i_hit           => hit1,
        i_t0            => t01,
        i_t1            => t11,
        i_empty         => empty1,
        i_mask_n        => mask_n1,
        o_rack          => rack1,
        i_error         => error1,

        -- output data stream
        o_data(0)       => o_wdata,
        o_shop          => open,
        o_sop(0)        => o_wsop,
        o_eop(0)        => o_weop,
        o_hit           => open,
        o_t0(0)         => o_t0,
        o_t1(0)         => o_t1,
        o_empty(0)      => o_empty,
        o_mask_n        => open,
        i_rack(0)       => i_rack,
        o_error(0)      => o_error,

        i_en            => i_en,
        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

end architecture;
