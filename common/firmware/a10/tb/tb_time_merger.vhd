library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;



entity tb_time_merger is 
end entity tb_time_merger;


architecture TB of tb_time_merger is

    -- Input from merging (first board) or links (subsequent boards)
    signal clk, clk_fast, reset_n : std_logic;
    
    -- links and datageneration
    constant ckTime: 		time	:= 10 ns;
    constant ckTime_fast: 		time	:= 8 ns;
    constant g_NLINKS_TOTL : integer := 64;
    constant g_NLINKS_DATA : integer := 12;
    constant W : integer := 8*32 + 8*6;
    signal slow_down_0, slow_down_1 : std_logic_vector(31 downto 0);
    signal gen_link_reg : work.util.slv32_array_t(1 downto 0);
    signal gen_link : std_logic_vector(35 downto 0);
    signal gen_link_valid : std_logic_vector(1 downto 0);
    signal gen_link_k, gen_link_k_reg : work.util.slv4_array_t(1 downto 0);
    
    -- signals
    signal rx_q : work.util.slv34_array_t(g_NLINKS_TOTL-1 downto 0) := (others => (others => '0'));
    signal rx_ren, rx_rdempty, sop, shop, eop, link_mask_n : std_logic_vector(g_NLINKS_TOTL-1 downto 0);
    signal rempty : std_logic;
    signal counter_int: unsigned(31 downto 0);

begin

    -- generate the clock
    ckProc: process
    begin
        clk <= '0';
        wait for ckTime/2;
        clk <= '1';
        wait for ckTime/2;
    end process;
    
    ckProcfast: process
    begin
        clk_fast <= '0';
        wait for ckTime_fast/2;
        clk_fast <= '1';
        wait for ckTime_fast/2;
    end process;

    inita : process
    begin
        reset_n	 <= '0';
        wait for 8 ns;
        reset_n	 <= '1';
        wait;
    end process inita;

    -- data generation and ts counter_ddr3
    slow_down_0 <= x"00000002";
    slow_down_1 <= x"00000003";

    --! we generate different sequences for the hit time:
    --! gen0: 3, 44, 55, 6, 77, 8, 9, AA, B, CC, DD, E, F
    --! gen1-63: 3, 4, 55, 66, 77, 88, 9, A, BB, CC, D, E, F
        --! data_generator_a10
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --e_data_gen_0 : entity work.data_generator_a10
    --generic map (
            --go_to_sh => 3,
            --go_to_trailer => 4--,
        --)
    --port map (
        --i_reset_n           => reset_n,
        --enable_pix          => '1',
        --i_dma_half_full     => '0',
        --random_seed         => (others => '1'),
        --data_pix_generated  => gen_link(0),
        --datak_pix_generated => gen_link_k(0),
        --data_pix_ready      => open,
        --start_global_time   => (others => '0'),
        --delay               => (others => '0'),
        --slow_down           => slow_down_0,
        --state_out           => open,
        --clk                 => clk--,
    --);

    --e_data_gen_1 : entity work.data_generator_a10
    --generic map (
            --go_to_sh => 3,
            --go_to_trailer => 4--,
        --)
    --port map (
        --i_reset_n           => reset_n,
        --enable_pix          => '1',
        --i_dma_half_full     => '0',
        --random_seed         => (others => '1'),
        --data_pix_generated  => gen_link(1),
        --datak_pix_generated => gen_link_k(1),
        --data_pix_ready      => open,
        --start_global_time   => (others => '0'),
        --delay               => x"0002",
        --slow_down           => slow_down_1,
        --state_out           => open,
        --clk                 => clk--,
    --);
    
    process
    begin
        counter_int <= (others => '0');
        wait until ( reset_n = '1' );
        
        for i in 0 to 80000 loop
            wait until rising_edge(clk);
            counter_int <= counter_int + 1;
        end loop;
        wait;
    end process;
    
    e_data_gen : entity work.mp_sorter_datagen
    generic map (
        send_header_trailer => '1'--,
    )
    port map (
        i_reset_n                 => reset_n,
        i_clk                     => clk,
        i_running                 => '1',
        i_global_ts(31 downto 0)  => std_logic_vector(counter_int),
        i_global_ts(63 downto 32) => (others => '0'),
        i_control_reg             => (31 => '1', others => '0'),
        i_seed                    => "11001111101100010101110100100010011010110001101011110100101000000",
        o_fifo_wdata              => gen_link,
        o_fifo_write              => gen_link_valid(0),
        o_datak                   => gen_link_k(0)--,
        --,
    );
    
    e_feb0 : entity work.f0_sim
    port map (
        clk         => clk,
        data_feb0   => gen_link_reg(0),
        datak_feb0  => gen_link_k_reg(0),
        reset_n     => reset_n--,
    );
    
    e_feb1 : entity work.f1_sim
    port map (
        clk         => clk,
        data_feb2   => gen_link_reg(1),
        datak_feb2  => gen_link_k_reg(1),
        reset_n     => reset_n--,
    );
    
    --gen_link_reg(0) <=  x"000000BC" when gen_link_valid(0) = '0' else gen_link(31 downto 0);
    --gen_link_k_reg(0) <= "0001" when gen_link_valid(0) = '0' else gen_link_k(0);

    --gen_link_fifos : FOR i in 0 to g_NLINKS_DATA - 1 GENERATE
        
        e_link_to_fifo_32 : entity work.link_to_fifo_32
        generic map (
            LINK_FIFO_ADDR_WIDTH => 8--;
        )
        port map (
            i_rx            => gen_link_reg(0),
            i_rx_k          => gen_link_k_reg(0),
            
            o_q             => rx_q(0),
            i_ren           => rx_ren(0),
            o_rdempty       => rx_rdempty(0),

            o_counter       => open,
            
            i_reset_n_156   => reset_n,
            i_clk_156       => clk,

            i_reset_n_250   => reset_n,
            i_clk_250       => clk_fast--;
        );
  
        sop(0)  <= '1' when rx_q(0)(33 downto 32) = "10" else '0';
        shop(0) <= '1' when rx_q(0)(33 downto 32) = "11" else '0'; 
        eop(0)  <= '1' when rx_q(0)(33 downto 32) = "10" else '0';
        
        e_link_to_fifo_32_2 : entity work.link_to_fifo_32
        generic map (
            LINK_FIFO_ADDR_WIDTH => 8--;
        )
        port map (
            i_rx            => gen_link_reg(1),
            i_rx_k          => gen_link_k_reg(1),
            
            o_q             => rx_q(1),
            i_ren           => rx_ren(1),
            o_rdempty       => rx_rdempty(1),

            o_counter       => open,
            
            i_reset_n_156   => reset_n,
            i_clk_156       => clk,

            i_reset_n_250   => reset_n,
            i_clk_250       => clk_fast--;
        );
  
        sop(1)  <= '1' when rx_q(1)(33 downto 32) = "10" else '0';
        shop(1) <= '1' when rx_q(1)(33 downto 32) = "11" else '0'; 
        eop(1)  <= '1' when rx_q(1)(33 downto 32) = "10" else '0';

    --END GENERATE gen_link_fifos;
    
    link_mask_n <= x"0000000000000003";

    --e_time_merger : entity work.time_merger_v2
        --generic map (
        --W => W,
        --TREE_DEPTH_w => 10,
        --TREE_DEPTH_r => 10,
        --g_NLINKS_DATA => 12,
        --N => 64--,
    --)
    --port map (
        --input streams
        --i_rdata                 => rx_q,
        --i_rsop                  => sop,
        --i_reop                  => eop,
        --i_rshop                 => shop,
        --i_rempty                => rx_rdempty,
        --i_link                  => 1, -- which link should be taken to check ts etc.
        --i_mask_n                => link_mask_n,
        --o_rack                  => rx_ren,
        
        --output stream
        --o_rdata                 => open,
        --i_ren                   => not rempty,
        --o_empty                 => rempty,
        
        --error outputs
        --o_error_pre             => open,
        --o_error_sh              => open,
        --o_error_gtime           => open,
        --o_error_shtime          => open,
        
        --i_reset_n               => reset_n,
        --i_clk                   => clk_fast--,
    --);


    e_time_merger : entity work.swb_time_merger
    generic map (
        W               => W,
        TREE_w          => 10,
        TREE_r          => 10,
        g_NLINKS_DATA   => g_NLINKS_DATA--,
    )
    port map (
        i_rx        => rx_q,
        i_rsop      => sop,
        i_reop      => eop,
        i_rshop     => shop,
        i_rempty    => rx_rdempty,
        i_rmask_n   => link_mask_n,
        o_rack      => rx_ren,

        -- output strem
        o_q             => open,
        o_q_debug       => open,
        o_rempty        => open,
        o_rempty_debug  => rempty,
        i_ren           => not rempty,
        o_header_debug  => open,
        o_trailer_debug => open,
        o_error         => open,

        i_reset_n       => reset_n,
        i_clk           => clk_fast--,
    );
    
end TB;


