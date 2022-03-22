library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity tb_data_path_farm is
end entity;


architecture TB of tb_data_path_farm is

    signal reset_n		: std_logic;
    signal reset		: std_logic;

    -- Input from merging (first board) or links (subsequent boards)
    signal clk_156		: 		 std_logic := '0';
    signal dataclk		: 		 std_logic := '0';
    signal data_en		:		 std_logic;
    signal data_in		:		 std_logic_vector(511 downto 0);
    signal ts_in		:		 std_logic_vector(31 downto 0);

    -- Input from PCIe demanding events
    signal pcieclk		:		std_logic := '0';
    signal ts_req_A		:		std_logic_vector(31 downto 0);
    signal req_en_A		:		std_logic;
    signal ts_req_B		:		std_logic_vector(31 downto 0);
    signal req_en_B		:		std_logic;
    signal tsblock_done :		std_logic_vector(15 downto 0);

    -- Output to DMA
    signal dma_data_out	    :	std_logic_vector(255 downto 0);
    signal dma_data_en		:	std_logic;
    signal dma_eoe			:   std_logic;

    -- Interface to memory bank A
    signal A_mem_clk		: std_logic := '0';
    signal A_mem_ready		: std_logic;
    signal A_mem_calibrated	: std_logic;
    signal A_mem_addr		: std_logic_vector(25 downto 0);
    signal A_mem_data		: std_logic_vector(511 downto 0);
    signal A_mem_write		: std_logic;
    signal A_mem_read		: std_logic;
    signal A_mem_q			: std_logic_vector(511 downto 0);
    signal A_mem_q_valid	: std_logic;

    -- Interface to memory bank B
    signal B_mem_clk		: std_logic := '0';
    signal B_mem_ready		: std_logic;
    signal B_mem_calibrated	: std_logic;
    signal B_mem_addr		: std_logic_vector(25 downto 0);
    signal B_mem_data		: std_logic_vector(511 downto 0);
    signal B_mem_write		: std_logic;
    signal B_mem_read		: std_logic;
    signal B_mem_q			: std_logic_vector(511 downto 0);
    signal B_mem_q_valid	: std_logic;

    -- links and datageneration
    constant NLINKS     : positive := 8;
    constant NLINKS_TOTL: positive := 16;
    constant LINK_FIFO_ADDR_WIDTH : integer := 10;
    constant g_NLINKS_FARM_TOTL : positive := 3;

    signal link_data        : std_logic_vector(NLINKS * 32 - 1 downto 0);
    signal link_datak       : std_logic_vector(NLINKS * 4 - 1 downto 0);
    signal counter_ddr3     : std_logic_vector(31 downto 0);

    signal w_pixel, r_pixel, w_scifi, r_scifi : std_logic_vector(NLINKS * 38 -1 downto 0);
    signal w_pixel_en, r_pixel_en, full_pixel, empty_pixel : std_logic;
    signal w_scifi_en, r_scifi_en, full_scifi, empty_scifi : std_logic;

    signal farm_data, farm_datak : work.util.slv32_array_t(g_NLINKS_FARM_TOTL-1 downto 0);

    signal rx : work.util.slv32_array_t(NLINKS_TOTL-1 downto 0);
    signal rx_k : work.util.slv4_array_t(NLINKS_TOTL-1 downto 0);

    signal link_data_pixel, link_data_scifi : std_logic_vector(NLINKS * 32 - 1  downto 0);
    signal link_datak_pixel, link_datak_scifi : std_logic_vector(NLINKS * 4 - 1  downto 0);

    signal pixel_data, scifi_data : std_logic_vector(257 downto 0);
    signal pixel_empty, pixel_ren, scifi_empty, scifi_ren : std_logic;
    signal data_wen, ddr_ready : std_logic;
    signal event_ts : std_logic_vector(47 downto 0);
    signal ts_req_num : std_logic_vector(31 downto 0);

    signal writeregs_156 : work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));
    signal writeregs_250 : work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));

    signal resets_n_156, resets_n_250 : std_logic_vector(31 downto 0) := (others => '0');

    -- clk period
    constant dataclk_period : time := 4 ns;
    constant pcieclk_period : time := 4 ns;
    constant A_mem_clk_period : time := 3.76 ns;
    constant B_mem_clk_period : time := 3.76 ns;

    constant CLK_MHZ : real := 10000.0; -- MHz

    signal toggle : std_logic_vector(1 downto 0);
    signal startinput : std_logic;
    signal ts_in_next			:		 std_logic_vector(31 downto 0);

    signal A_mem_read_del1: std_logic;
    signal A_mem_read_del2: std_logic;
    signal A_mem_read_del3: std_logic;
    signal A_mem_read_del4: std_logic;

    signal A_mem_addr_del1		: std_logic_vector(25 downto 0);
    signal A_mem_addr_del2		: std_logic_vector(25 downto 0);
    signal A_mem_addr_del3		: std_logic_vector(25 downto 0);
    signal A_mem_addr_del4		: std_logic_vector(25 downto 0);

    signal B_mem_read_del1: std_logic;
    signal B_mem_read_del2: std_logic;
    signal B_mem_read_del3: std_logic;
    signal B_mem_read_del4: std_logic;

    signal B_mem_addr_del1		: std_logic_vector(25 downto 0);
    signal B_mem_addr_del2		: std_logic_vector(25 downto 0);
    signal B_mem_addr_del3		: std_logic_vector(25 downto 0);
    signal B_mem_addr_del4		: std_logic_vector(25 downto 0);

    signal midas_data_511 : work.util.slv32_array_t(15 downto 0);


begin

    clk_156 <= not clk_156 after (0.5 us / CLK_MHZ);
    dataclk <= not dataclk after (0.1 us / CLK_MHZ);
    pcieclk <= not dataclk after (0.1 us / CLK_MHZ);
    A_mem_clk <= not A_mem_clk after (0.1 us / CLK_MHZ);
    B_mem_clk <= not B_mem_clk after (0.1 us / CLK_MHZ);

    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);

    --! Setup
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! USE_GEN_LINK | USE_STREAM | USE_MERGER | USE_LINK | USE_GEN_MERGER | USE_FARM | SWB_READOUT_LINK_REGISTER_W | EFFECT                                                                         | WORKS
    --! ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --! 1            | 0          | 0          | 1        | 0              | 0        | n                           | Generate data for all 64 links, readout link n via DAM                         | x
    --! 1            | 1          | 0          | 0        | 0              | 0        | -                           | Generate data for all 64 links, simple merging of links, readout via DAM       | x
    --! 1            | 0          | 1          | 0        | 0              | 0        | -                           | Generate data for all 64 links, time merging of links, readout via DAM         | x
    --! 0            | 0          | 0          | 0        | 1              | 1        | -                           | Generate time merged data, send to farm                                        | x

    resets_n_156(RESET_BIT_DATAGEN)                             <= '0', '1' after (1.0 us / CLK_MHZ);
    resets_n_250(RESET_BIT_SWB_STREAM_MERGER)                   <= '0', '1' after (1.0 us / CLK_MHZ);
    resets_n_250(RESET_BIT_SWB_TIME_MERGER)                     <= '0', '1' after (1.0 us / CLK_MHZ);
    writeregs_156(DATAGENERATOR_DIVIDER_REGISTER_W)             <= x"00000002";
    writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK)   <= '1';
    writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM)     <= '0';
    writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER)     <= '1';

    writeregs_250(SWB_LINK_MASK_PIXEL_REGISTER_W)               <= x"00000003";--x"00000048";
    writeregs_250(SWB_READOUT_LINK_REGISTER_W)                  <= x"00000001";
    writeregs_250(GET_N_DMA_WORDS_REGISTER_W)                   <= (others => '1');
    writeregs_250(DMA_REGISTER_W)(DMA_BIT_ENABLE)               <= '1';
    mask_n <= x"00000000" & writeregs_250(SWB_LINK_MASK_PIXEL_REGISTER_W);


    --! SWB data path
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    g_demerge: FOR i in g_NLINKS_FARM_TOTL-1 downto 0 GENERATE
        e_swb_data_path : entity work.swb_data_path
        generic map (
            g_LOOPUP_NAME           => "intRun2021",
            g_ADDR_WIDTH            => 8,
            g_NLINKS_DATA           => g_NLINKS_DATA,
            LINK_FIFO_ADDR_WIDTH    => 8,
            SWB_ID                  => x"01",
            -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
            DATA_TYPE               => x"01"--;
        )
        port map(
            i_clk_156        => clk,
            i_clk_250        => clk_fast,
            
            i_reset_n_156    => reset_n,
            i_reset_n_250    => reset_n,

            i_resets_n_156   => resets_n_156,
            i_resets_n_250   => resets_n_250,
            
            i_rx             => rx,
            i_rx_k           => rx_k,
                
            i_rmask_n        => mask_n(g_NLINKS_DATA - 1 downto 0),

            i_writeregs_156  => writeregs_156,
            i_writeregs_250  => writeregs_250,

            o_counter_156    => open,
            o_counter_250    => open,

            i_dmamemhalffull => '0',
            
            o_farm_data      => farm_data(i),
            o_farm_datak     => farm_datak(i),

            o_dma_wren       => open,
            o_dma_done       => open,
            o_endofevent     => open,
            o_dma_data       => open--;
        );
    END GENERATE;

    e_farm_link_to_fifo : entity work.farm_link_to_fifo
    generic map (
        g_LOOPUP_NAME       => g_LOOPUP_NAME,
        g_NLINKS_SWB_TOTL   => g_NLINKS_TOTL,
        N_PIXEL             => g_NLINKS_PIXEL,
        N_SCIFI             => g_NLINKS_SCIFI,
        LINK_FIFO_ADDR_WIDTH=> LINK_FIFO_ADDR_WIDTH--,
    )
    port map (
        --! link data in
        i_rx                => farm_data,
        i_rx_k              => farm_datak,
        
        --! link data out
        o_tx                => open,
        o_tx_k              => open,

        --! data out
        o_data              => aligned_data,
        o_empty             => aligned_empty,
        i_ren               => aligned_ren,
        o_shop              => aligned_shop,
        o_sop               => aligned_sop,
        o_eop               => aligned_eop,
        o_hit               => aligned_hit,
        o_t0                => aligned_t0,
        o_t1                => aligned_t1,
        o_error             => aligned_error,
    
        --! status counters
        --! (g_NLINKS_DATA*5)-1 downto 0 -> link to fifo counters
        --! (g_NLINKS_DATA*4)+(g_NLINKS_DATA*5)-1 downto (g_NLINKS_DATA*5) -> link align counters
        o_counter           => counter_link_to_fifo,
        
        i_clk_250_link      => i_clk_250_link,
        i_reset_n_250_link  => i_reset_n_250_link,
        
        i_clk_250           => i_clk_250_pcie,
        i_reset_n_250       => i_resets_n_pcie(RESET_BIT_FARM_DATA_PATH)--,
    );
    
    e_farm_midas_event_builder : entity work.farm_midas_event_builder
    generic map (
        g_NLINKS_SWB_TOTL => 16,
        N_PIXEL           => 8,
        N_SCIFI           => 8,
        RAM_ADDR          => 12--,
    )
    port map (
        i_pixel         => pixel_data,
        i_empty_pixel   => pixel_empty,
        o_ren_pixel     => pixel_ren,

        i_scifi         => scifi_data,
        i_empty_scifi   => scifi_empty,
        o_ren_scifi     => scifi_ren,

        i_farm_id       => x"AFFEAFFE",

        -- DDR
        o_data          => data_in,
        o_wen           => data_wen,
        o_event_ts      => event_ts,
        i_ddr_ready     => ddr_ready,

        -- Link data
        o_pixel         => open,
        o_wen_pixel     => open,

        o_scifi         => open,
        o_wen_scifi     => open,

        o_counters      => open,

        i_reset_n_250   => reset_n,
        i_clk_250       => dataclk--,
    );

    process(dataclk, reset_n)
    begin
        if( reset_n <= '0' ) then
            counter_ddr3    <= (others => '0');
        elsif ( dataclk'event and dataclk = '1' ) then
            counter_ddr3 <= counter_ddr3 + '1';
        end if;
    end process;

    gen_midas_data : FOR I in 0 to 15 GENERATE
        midas_data_511(I) <= data_in(I*32 + 31 downto I*32);
    END GENERATE;

    e_farm_data_path : entity work.farm_data_path
    port map(
        reset_n         => reset_n,
        reset_n_ddr3    => reset_n,

        -- Input from merging (first board) or links (subsequent boards)
        dataclk         => dataclk,
        data_in         => data_in,
        data_en         => data_wen,
        ts_in           => event_ts(35 downto 4), -- 3:0 -> hit, 9:0 -> sub header
        o_ddr_ready     => ddr_ready,

        -- Input from PCIe demanding events
        pcieclk         => pcieclk,
        ts_req_A        => ts_req_A,
        req_en_A        => req_en_A,
        ts_req_B        => ts_req_B,
        req_en_B        => req_en_B,
        tsblock_done    => tsblock_done,

        -- Output to DMA
        dma_data_out    => dma_data_out,
        dma_data_en     => dma_data_en,
        dma_eoe         => dma_eoe,
        i_dmamemhalffull=> '0',
        i_num_req_events=> ts_req_num,
        o_dma_done      => open,
        i_dma_wen       => '1',

        -- Interface to memory bank A
        A_mem_clk       => A_mem_clk,
        A_mem_ready     => A_mem_ready,
        A_mem_calibrated=> A_mem_calibrated,
        A_mem_addr      => A_mem_addr,
        A_mem_data      => A_mem_data,
        A_mem_write     => A_mem_write,
        A_mem_read      => A_mem_read,
        A_mem_q         => A_mem_q,
        A_mem_q_valid   => A_mem_q_valid,

        -- Interface to memory bank B
        B_mem_clk       => B_mem_clk,
        B_mem_ready     => B_mem_ready,
        B_mem_calibrated=> B_mem_calibrated,
        B_mem_addr		=> B_mem_addr,
        B_mem_data		=> B_mem_data,
        B_mem_write		=> B_mem_write,
        B_mem_read		=> B_mem_read,
        B_mem_q			=> B_mem_q,
        B_mem_q_valid	=> B_mem_q_valid
    );

    e_ddr3_a : entity work.ip_ram
    generic map (
        ADDR_WIDTH_A    => 9,
        ADDR_WIDTH_B    => 9,
        DATA_WIDTH_A    => 512,
        DATA_WIDTH_B    => 512--,
    )
    port map (
        address_a       => A_mem_addr(8 downto 0),
        address_b       => A_mem_addr(8 downto 0),
        clock_a         => A_mem_clk,
        clock_b         => A_mem_clk,
        data_a          => A_mem_data,
        data_b          => (others => '0'),
        wren_a          => A_mem_write,
        wren_b          => '0',
        q_a             => open,
        q_b             => A_mem_q--,
    );

    e_ddr3_b : entity work.ip_ram
    generic map (
        ADDR_WIDTH_A    => 9,
        ADDR_WIDTH_B    => 9,
        DATA_WIDTH_A    => 512,
        DATA_WIDTH_B    => 512--,
    )
    port map (
        address_a       => B_mem_addr(8 downto 0),
        address_b       => B_mem_addr(8 downto 0),
        clock_a         => B_mem_clk,
        clock_b         => B_mem_clk,
        data_a          => B_mem_data,
        data_b          => (others => '0'),
        wren_a          => B_mem_write,
        wren_b          => '0',
        q_a             => open,
        q_b             => B_mem_q--,
    );

    -- Memready
    process begin
        A_mem_ready <= '0';
        B_mem_ready <= '0';
        wait for A_mem_clk_period * 25;
        A_mem_ready <= '1';
        B_mem_ready <= '1';
        wait for A_mem_clk_period * 300;
        A_mem_ready <= '0';
        B_mem_ready <= '0';
        wait for A_mem_clk_period;
        A_mem_ready <= '1';
        B_mem_ready <= '1';
        wait for A_mem_clk_period * 250;
        A_mem_ready <= '0';
        B_mem_ready <= '0';
        wait for A_mem_clk_period;
        A_mem_ready <= '1';
        B_mem_ready <= '1';
        wait for A_mem_clk_period * 600;
    end process;

    A_mem_calibrated <= '1';
    B_mem_calibrated <= '1';

    -- Request generation
    process begin
    req_en_A <= '0';
    wait for pcieclk_period;-- * 26500;
    req_en_A <= '1';
    ts_req_num <= x"00000008";
    ts_req_A <= x"04030201";--"00010000";
    wait for pcieclk_period;
    req_en_A <= '1';
    ts_req_A <= x"0B0A0906";--x"00030002";
    wait for pcieclk_period;
    req_en_A <= '0';
    wait for pcieclk_period;
    req_en_A <= '0';
    tsblock_done	<= (others => '0');
    end process;

    -- Memory A simulation
    process(A_mem_clk, reset_n)
    begin
    if(reset_n <= '0') then
        A_mem_q_valid 	<= '0';
        A_mem_read_del1 <= '0';
        A_mem_read_del2 <= '0';
        A_mem_read_del3 <= '0';
        A_mem_read_del4 <= '0';
    elsif(A_mem_clk'event and A_mem_clk = '1') then
        A_mem_read_del1 <= A_mem_read;
        A_mem_read_del2 <= A_mem_read_del1;
        A_mem_read_del3	<= A_mem_read_del2;
        A_mem_read_del4	<= A_mem_read_del3;
        A_mem_q_valid   <= A_mem_read_del4;

        A_mem_addr_del1 <= A_mem_addr;
        A_mem_addr_del2 <= A_mem_addr_del1;
        A_mem_addr_del3	<= A_mem_addr_del2;
        A_mem_addr_del4	<= A_mem_addr_del3;
    --  A_mem_q		<= (others => '0');
    --  A_mem_q(25 downto 0)  <= A_mem_addr_del4;
    end if;
    end process;

    -- Memory B simulation
    process(B_mem_clk, reset_n)
    begin
    if(reset_n <= '0') then
        B_mem_q_valid 	<= '0';
        B_mem_read_del1 <= '0';
        B_mem_read_del2 <= '0';
        B_mem_read_del3 <= '0';
        B_mem_read_del4 <= '0';
    elsif(B_mem_clk'event and B_mem_clk = '1') then
        B_mem_read_del1 <= B_mem_read;
        B_mem_read_del2 <= B_mem_read_del1;
        B_mem_read_del3	<= B_mem_read_del2;
        B_mem_read_del4	<= B_mem_read_del3;
        B_mem_q_valid   <= B_mem_read_del4;

        B_mem_addr_del1 <= B_mem_addr;
        B_mem_addr_del2 <= B_mem_addr_del1;
        B_mem_addr_del3	<= B_mem_addr_del2;
        B_mem_addr_del4	<= B_mem_addr_del3;
    --  B_mem_q		<= (others => '0');
    --  B_mem_q(25 downto 0)  <= B_mem_addr_del4;
    end if;
    end process;

end architecture;
