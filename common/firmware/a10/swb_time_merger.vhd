library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;


entity swb_time_merger is
generic (
    W : positive := 8*32+8*6;
    TREE_w : integer := 10;
    TREE_r : integer := 10;
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE: std_logic_vector(7 downto 0) := x"01";
    g_NLINKS : positive := 16--;
);
port (
    -- input streams
    i_rx        : in    work.util.slv38_array_t(g_NLINKS - 1 downto 0);
    i_rsop      : in    std_logic_vector(g_NLINKS-1 downto 0);
    i_reop      : in    std_logic_vector(g_NLINKS-1 downto 0);
    i_rshop     : in    std_logic_vector(g_NLINKS-1 downto 0);
    i_rempty    : in    std_logic_vector(g_NLINKS-1 downto 0);
    i_rmask_n   : in    std_logic_vector(g_NLINKS-1 downto 0);
    o_rack      : out   std_logic_vector(g_NLINKS-1 downto 0);

    -- output strem
    o_q         : out   std_logic_vector(W-1 downto 0);
    o_q_debug   : out   std_logic_vector(31 downto 0);
    o_rempty    : out   std_logic;
    i_ren       : in    std_logic;
    o_header    : out   std_logic;
    o_trailer   : out   std_logic;
    o_error     : out   std_logic;

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of swb_time_merger is

    signal rdata : std_logic_vector(W-1 downto 0);
    signal rempty, wfull : std_logic;
    signal link_number : std_logic_vector(5 downto 0);

begin

    e_time_merger : entity work.time_merger
        generic map (
        W => W,
        TREE_DEPTH_w => TREE_w,
        TREE_DEPTH_r => TREE_r,
        N => g_NLINKS--,
    )
    port map (
        -- input streams
        i_rdata                 => i_rx,
        i_rsop                  => i_rsop,
        i_reop                  => i_reop,
        i_rshop                 => i_rshop,
        i_rempty                => i_rempty,
        i_link                  => 1, -- which link should be taken to check ts etc.
        i_mask_n                => i_rmask_n,
        o_rack                  => o_rack,
        
        -- output stream
        o_rdata                 => rdata,
        i_ren                   => not rempty and not wfull,
        o_empty                 => rempty,
        
        -- error outputs
        o_error_pre             => open,
        o_error_sh              => open,
        o_error_gtime           => open,
        o_error_shtime          => open,
        
        i_reset_n               => i_reset_n,
        i_clk                   => i_clk--,
    );


    e_merger_fifo_farm : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => W,
        DEVICE => "Arria 10"--,
    )
    port map (
        q               => o_q,
        empty           => o_rempty,
        rdreq           => i_ren,
        data            => rdata,
        full            => wfull,
        wrreq           => not rempty and not wfull,
        sclr            => not i_reset_n,
        clock           => i_clk--,
    );
   
    -- link number
    link_number <= o_q(37 downto 32);
    -- hit
    o_q_debug <= o_q(31 downto 0);
    -- header info
    o_header    <= '1' when o_q(37 downto 32) = pre_marker else '0';
    o_trailer   <= '1' when o_q(37 downto 32) = tr_marker else '0';
    -- TODO: handle errors, at the moment they are sent out at the end of normal events
    o_error     <= '1' when o_q(37 downto 32) = err_marker and o_q(7 downto 0) = x"DC" else '0';

end architecture;
