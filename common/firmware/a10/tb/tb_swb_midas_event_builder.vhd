library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use std.textio.all;
use ieee.std_logic_textio.all;

entity tb_swb_midas_event_builder is 
end entity tb_swb_midas_event_builder;


architecture TB of tb_swb_midas_event_builder is

    -- Input from merging (first board) or links (subsequent boards)
    signal clk, clk_fast, reset_n : std_logic;
    
    -- links and datageneration
    constant ckTime: 		time	:= 10 ns;
    constant ckTime_fast: 		time	:= 8 ns;
    file file_RESULTS : text;
    constant g_NLINKS_TOTL : integer := 1;
    constant g_NLINKS_DATA : integer := 12;
    constant W : integer := 8*32 + 8*6;
    signal slow_down_0, slow_down_1 : std_logic_vector(31 downto 0);
    signal gen_link, gen_link_reg : work.util.slv32_array_t(1 downto 0);
    signal gen_link_k : work.util.slv4_array_t(1 downto 0);
    
    -- signals
    signal rx_q : work.util.slv38_array_t(g_NLINKS_TOTL-1 downto 0) := (others => (others => '0'));
    signal rx_ren, rx_rdempty, sop, shop, eop, link_mask_n : std_logic_vector(g_NLINKS_TOTL-1 downto 0);
    signal rempty, dma_wen : std_logic;
    signal dma_data : std_logic_vector (255 downto 0);
    signal dma_data_array : work.util.slv32_array_t(7 downto 0);

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
        file_open(file_RESULTS, "memory_content.txt", write_mode);
        file_close(file_RESULTS);
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
    e_data_gen_0 : entity work.data_generator_a10
    generic map (
            go_to_sh => 3,
            go_to_trailer => 4--,
        )
    port map (
        i_reset_n           => reset_n,
        enable_pix          => '1',
        i_dma_half_full     => '0',
        random_seed         => (others => '1'),
        data_pix_generated  => gen_link(0),
        datak_pix_generated => gen_link_k(0),
        data_pix_ready      => open,
        start_global_time   => (others => '0'),
        delay               => (others => '0'),
        slow_down           => slow_down_0,
        state_out           => open,
        clk                 => clk--,
    );
    
    gen_link_reg(0) <=  gen_link(0) when gen_link_k(0) = "0001" or gen_link(0)(28 downto 23) = "111111" else
                        gen_link(0)(31 downto 4) & gen_link(0)(31 downto 28); -- set hit time to zero for simulation checks

    --! generate link_to_fifo_32
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------

    e_link_to_fifo_32 : entity work.link_to_fifo_32
    generic map (
        LINK_FIFO_ADDR_WIDTH => 8--;
    )
    port map (
        i_rx            => gen_link_reg(0),
        i_rx_k          => gen_link_k(0),
        
        o_q             => rx_q(0),
        i_ren           => rx_ren(0),
        o_rdempty       => rx_rdempty(0),

        o_counter       => open,
        
        i_reset_n_156   => reset_n,
        i_clk_156       => clk,

        i_reset_n_250   => reset_n,
        i_clk_250       => clk_fast--;
    );

    sop(0) <= rx_q(0)(36);
    shop(0) <= '1' when rx_q(0)(37 downto 36) = "00" and rx_q(0)(31 downto 26) = "111111" else '0';
    eop(0) <= rx_q(0)(37);
    
    e_swb_midas_event_builder : entity work.swb_midas_event_builder
    generic map (
        DATA_TYPE => x"01"--,
    )
    port map (
        i_rx            => rx_q(0)(35 downto 4),
        i_rempty        => rx_rdempty(0),
        i_header        => sop(0),
        i_trailer       => eop(0),

        i_get_n_words   => (others => '1'),
        i_dmamemhalffull=> '0',
        i_wen           => '1',
        o_data          => dma_data,
        o_wen           => dma_wen,
        o_ren           => rx_ren(0),
        o_endofevent    => open,
        o_done          => open,
        o_state_out     => open,
    
        --! status counters 
        --! 0: bank_builder_idle_not_header
        --! 1: bank_builder_skip_event_dma
        --! 2: bank_builder_ram_full
        --! 3: bank_builder_tag_fifo_full
        o_counters      => open,

        i_reset_n_250   => reset_n,
        i_clk_250       => clk_fast--,
    );

    dma_data_array(0) <= dma_data(0*32 + 31 downto 0*32);
    dma_data_array(1) <= dma_data(1*32 + 31 downto 1*32);
    dma_data_array(2) <= dma_data(2*32 + 31 downto 2*32);
    dma_data_array(3) <= dma_data(3*32 + 31 downto 3*32);
    dma_data_array(4) <= dma_data(4*32 + 31 downto 4*32);
    dma_data_array(5) <= dma_data(5*32 + 31 downto 5*32);
    dma_data_array(6) <= dma_data(6*32 + 31 downto 6*32);
    dma_data_array(7) <= dma_data(7*32 + 31 downto 7*32);

    process
        variable v_OLINE : line;
    begin
        wait until rising_edge(clk_fast);
            if ( dma_wen = '1' ) then
                file_open(file_RESULTS, "memory_content.txt", append_mode); 
                for i in 0 to 7 loop
                    write(v_OLINE, work.util.to_hstring(dma_data_array(i)));
                    writeline(file_RESULTS, v_OLINE);
                end loop;
                file_close(file_RESULTS);
            end if;
    end process;
    
end TB;


