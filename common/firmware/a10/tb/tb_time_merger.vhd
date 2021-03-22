library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


entity tb_time_merger is 
end entity tb_time_merger;


architecture TB of tb_time_merger is

    -- Input from merging (first board) or links (subsequent boards)
    signal clk, clk_fast, reset_n : std_logic;
    
    -- links and datageneration
    constant CLK_MHZ : real := 10000.0; -- MHz
    constant g_NLINKS_TOTL : integer := 64;
    constant g_NLINKS_DATA : integer := 12;
    signal slow_down : std_logic_vector(31 downto 0);
    signal gen_link : std_logic_vector(31 downto 0);
    signal gen_link_k : std_logic_vector(3 downto 0);
    
    -- signals
    signal rx_q : work.util.slv38_array_t(g_NLINKS_TOTL-1 downto 0) := (others => (others => '0'));
    signal rx_ren, rx_rdempty, sop, shop, eop, link_mask_n : std_logic_vector(g_NLINKS_TOTL-1 downto 0);
    signal rempty : std_logic;


begin

    clk     <= not clk after (0.5 us / CLK_MHZ);
    clk_fast<= not clk_fast after (0.1 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);

    -- data generation and ts counter_ddr3
    slow_down <= x"00000002";

        --! data_generator_a10
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_gen_link : entity work.data_generator_a10
    generic map (
            go_to_sh => 3,
            go_to_trailer => 4--,
        )
    port map (
        i_reset_n           => reset_n,
        enable_pix          => '1',
        i_dma_half_full     => '0',
        random_seed         => (others => '1'),
        data_pix_generated  => gen_link,
        datak_pix_generated => gen_link_k,
        data_pix_ready      => open,
        start_global_time   => (others => '0'),
        slow_down           => slow_down,
        state_out           => open,
        clk                 => clk--,
    );

    --! generate link_to_fifo_32
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    gen_link_fifos : FOR i in 0 to g_NLINKS_DATA - 1 GENERATE
        
        e_link_to_fifo_32 : entity work.link_to_fifo_32
        generic map (
            LINK_FIFO_ADDR_WIDTH => 8--;
        )
        port map (
            i_rx            => gen_link,
            i_rx_k          => gen_link_k,
            
            o_q             => rx_q(i),
            i_ren           => rx_ren(i),
            o_rdempty       => rx_rdempty(i),

            o_counter       => open,
            
            i_reset_n_156   => reset_n,
            i_clk_156       => clk,

            i_reset_n_250   => reset_n,
            i_clk_250       => clk_fast--;
        );
  
        sop(i) <= rx_q(i)(36);
        shop(i) <= '1' when rx_q(i)(37 downto 36) = "00" and rx_q(i)(31 downto 26) = "111111" else '0';
        eop(i) <= rx_q(i)(37);

    END GENERATE gen_link_fifos;
    
--     link_mask_n <= "1101110111011101110111011101110111";
    --link_mask_n <= "11111111111111111111111111111111";
        --link_mask_n <= "1111111111111111111111111111111111";
--     link_mask_n <= "1111111111111111";
--     link_mask_n <= "11011101";
--     link_mask_n <= "11111111";
    link_mask_n <= x"000000000000000F";

    e_time_merger : entity work.time_merger_v2
        generic map (
        W => 8*32 + 8*6,
        TREE_DEPTH_w => 10,
        TREE_DEPTH_r => 10,
        N => 64--,
    )
    port map (
        -- input streams
        i_rdata                 => rx_q,
        i_rsop                  => sop,
        i_reop                  => eop,
        i_rshop                 => shop,
        i_rempty                => rx_rdempty,
        i_link                  => 1, -- which link should be taken to check ts etc.
        i_mask_n                => link_mask_n,
        o_rack                  => rx_ren,
        
        -- output stream
        o_rdata                 => open,
        i_ren                   => not rempty,
        o_empty                 => rempty,
        
        -- error outputs
        o_error_pre             => open,
        o_error_sh              => open,
        o_error_gtime           => open,
        o_error_shtime          => open,
        
        i_reset_n               => reset_n,
        i_clk                   => clk_fast--,
    );
    
end TB;


