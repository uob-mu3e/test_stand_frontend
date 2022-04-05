-------------------------------------------------------
--! @swb_block.vhd
--! @brief the swb_block can be used
--! for the LCHb Board and the development board
--! mainly it includes the datapath which includes
--! merging hits from multiple FEBs. There will be
--! four types of SWB which differe accordingly to
--! the detector data they receive (inner pixel,
--! scifi, down and up stream pixel/tiles)
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity swb_block is
generic (
    g_NLINKS_FEB_TOTL       : positive := 12;
    g_NLINKS_FARM_TOTL      : positive := 3;
    g_NLINKS_FARM_PIXEL     : positive := 2;
    g_NLINKS_DATA_PIXEL     : positive := 10;
    g_NLINKS_DATA_PIXEL_US  : positive := 5;
    g_NLINKS_DATA_PIXEL_DS  : positive := 5;
    g_NLINKS_FARM_SCIFI     : positive := 1;
    g_NLINKS_DATA_SCIFI     : positive := 2;
    -- needed for simulation
    g_SC_SEC_SKIP_INIT      : std_logic := '0';
    SWB_ID                  : std_logic_vector(7 downto 0) := x"01"--;
);
port (

    --! links to/from FEBs
    i_feb_rx            : in  work.mu3e.link_array_t(g_NLINKS_FEB_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    o_feb_tx            : out work.mu3e.link_array_t(g_NLINKS_FEB_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    --! PCIe registers / memory
    i_writeregs         : in  work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));
    i_regwritten        : in  std_logic_vector(63 downto 0) := (others => '0');
    o_readregs          : out work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));
    i_resets_n          : in  std_logic_vector(31 downto 0) := (others => '0');

    i_wmem_rdata        : in  std_logic_vector(31 downto 0) := (others => '0');
    o_wmem_addr         : out std_logic_vector(15 downto 0) := (others => '0');

    o_rmem_wdata        : out std_logic_vector(31 downto 0) := (others => '0');
    o_rmem_addr         : out std_logic_vector(15 downto 0) := (others => '0');
    o_rmem_we           : out std_logic := '0';

    i_dmamemhalffull    : in  std_logic := '0';
    o_dma_wren          : out std_logic := '0';
    o_endofevent        : out std_logic := '0';
    o_dma_data          : out std_logic_vector(255 downto 0) := (others => '0');

    --! links to farm
    o_farm_tx           : out work.mu3e.link_array_t(g_NLINKS_FARM_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    --! clock / reset_n
    i_reset_n           : in  std_logic;
    i_clk               : in  std_logic--;

);
end entity;

--! @brief arch definition of the swb_block
--! @details The arch of the swb_block can be used
--! for the LCHb Board and the development board
--! mainly it includes the datapath which includes
--! merging hits from multiple FEBs. There will be
--! four types of SWB which differe accordingly to
--! the detector data they receive (inner pixel,
--! scifi, down and up stream pixel/tiles)
architecture arch of swb_block is

    --! masking signals
    signal pixel_mask_n, scifi_mask_n : std_logic_vector(63 downto 0);

    --! feb links
    signal feb_rx : work.mu3e.link_array_t(g_NLINKS_FEB_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    --! farm links
    signal farm_data       : work.mu3e.link_array_t(g_NLINKS_FARM_TOTL-1 downto 0)  := (others => work.mu3e.LINK_IDLE);
    signal pixel_farm_data : work.mu3e.link_array_t(g_NLINKS_FARM_PIXEL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    signal scifi_farm_data : work.mu3e.link_array_t(g_NLINKS_FARM_SCIFI-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    
    --! DMA debug path
    signal builder_data : work.mu3e.link_t;
    signal builder_rempty, builder_rack : std_logic;
        -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
    signal builder_data_type : std_logic_vector(1 downto 0);

    signal data_debug_pixel : work.mu3e.link_array_t(g_NLINKS_FARM_PIXEL-1 downto 0);
    signal rempty_debug_pixel, rack_debug_pixel : std_logic_vector(g_NLINKS_FARM_PIXEL-1 downto 0);
    
    signal data_debug_scifi : work.mu3e.link_array_t(g_NLINKS_FARM_SCIFI-1 downto 0);
    signal rempty_debug_scifi, rack_debug_scifi : std_logic_vector(g_NLINKS_FARM_SCIFI-1 downto 0);
    
    --! demerged FEB links
    signal rx_data         : work.mu3e.link_array_t(g_NLINKS_FEB_TOTL-1 downto 0)   := (others => work.mu3e.LINK_IDLE);
    signal rx_sc           : work.mu3e.link_array_t(g_NLINKS_FEB_TOTL-1 downto 0)   := (others => work.mu3e.LINK_IDLE);
    signal rx_rc           : work.mu3e.link_array_t(g_NLINKS_FEB_TOTL-1 downto 0)   := (others => work.mu3e.LINK_IDLE);
    signal rx_data_pixel   : work.mu3e.link_array_t(g_NLINKS_DATA_PIXEL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    signal rx_data_scifi   : work.mu3e.link_array_t(g_NLINKS_DATA_SCIFI-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    --! counters
    signal counter_swb_data_pixel_us    : work.util.slv32_array_t(g_NLINKS_DATA_PIXEL_US * 5 + 3 * (N_LINKS_TREE(3) + N_LINKS_TREE(2) + N_LINKS_TREE(1)) + 7 downto 0);
    signal counter_swb_data_pixel_ds    : work.util.slv32_array_t(g_NLINKS_DATA_PIXEL_DS * 5 + 3 * (N_LINKS_TREE(3) + N_LINKS_TREE(2) + N_LINKS_TREE(1)) + 7 downto 0);
    signal counter_swb_data_scifi       : work.util.slv32_array_t(g_NLINKS_DATA_SCIFI * 5 + 3 * (N_LINKS_TREE(3) + N_LINKS_TREE(2) + N_LINKS_TREE(1)) + 7 downto 0);
    signal builder_counters             : work.util.slv32_array_t(3 downto 0);
    signal counter_swb                  : work.util.slv32_array_t(builder_counters'left+1 + counter_swb_data_pixel_us'left+1 + counter_swb_data_pixel_ds'left+1 + counter_swb_data_scifi'left+1 - 1 downto 0);

begin

    --! @brief data path of the SWB board
    --! @details the data path of the SWB board is first splitting the
    --! data from the FEBs into data, slow control and run control packages.
    --! The different paths are than assigned to the corresponding entities.
    --! The data is merged in time over all incoming FEBs. After this packages
    --! are build and the data is send of to the farm boars. The slow control
    --! data is saved in the PCIe memory and can be further used in the MIDAS
    --! system. The run control packages are used to control the run and give
    --! feedback to MIDAS if all FEBs started the run.

    --! counter readout
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    o_readregs(SWB_COUNTER_REGISTER_R) <= counter_swb(to_integer(unsigned(i_writeregs(SWB_COUNTER_REGISTER_W))));


    --! demerge data
    --! three types of data will be extracted from the links
    --! data => detector data
    --! sc => slow control packages
    --! rc => runcontrol packages
    g_demerge: FOR i in g_NLINKS_FEB_TOTL-1 downto 0 GENERATE
        feb_rx(i) <= work.mu3e.to_link(i_feb_rx(i).data, i_feb_rx(i).datak);
        e_data_demerge : entity work.swb_data_demerger
        port map(
            i_clk               => i_clk,
            i_reset_n           => i_resets_n(RESET_BIT_EVENT_COUNTER),
            i_aligned           => '1',
            i_data              => feb_rx(i),

            o_data              => rx_data(i),
            o_sc                => rx_sc(i),
            o_rc                => rx_rc(i),
            o_fpga_id           => open--,
        );
    end generate;


    --! run control used by MIDAS
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_run_control : entity work.run_control
    generic map (
        N_LINKS_g              => g_NLINKS_FEB_TOTL--,
    )
    port map (
        i_reset_ack_seen_n     => i_resets_n(RESET_BIT_RUN_START_ACK),
        i_reset_run_end_n      => i_resets_n(RESET_BIT_RUN_END_ACK),
        -- TODO: Write out padding 4kB at MIDAS Bank Builder if run end is done
        -- TODO: connect buffers emtpy from dma here
        -- o_all_run_end_seen => MIDAS Builder => i_buffer_empty
        i_buffers_empty        => (others => '1'),
        o_feb_merger_timeout   => o_readregs(CNT_FEB_MERGE_TIMEOUT_R),
        i_aligned              => (others => '1'),
        i_data                 => rx_rc,
        i_link_enable          => i_writeregs(FEB_ENABLE_REGISTER_W),
        i_addr                 => i_writeregs(RUN_NR_ADDR_REGISTER_W), -- ask for run number of FEB with this addr.
        i_run_number           => i_writeregs(RUN_NR_REGISTER_W)(23 downto 0),
        o_run_number           => o_readregs(RUN_NR_REGISTER_R), -- run number of i_addr
        o_runNr_ack            => o_readregs(RUN_NR_ACK_REGISTER_R), -- which FEBs have responded with run number in i_run_number
        o_run_stop_ack         => o_readregs(RUN_STOP_ACK_REGISTER_R),
        o_time_counter(31 downto 0)  => o_readregs(GLOBAL_TS_LOW_REGISTER_R),
        o_time_counter(63 downto 32) => o_readregs(GLOBAL_TS_HIGH_REGISTER_R),

        i_reset_n              => i_resets_n(RESET_BIT_GLOBAL_TS),
        i_clk                  => i_clk--,
    );


    --! SWB slow control
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_sc_main : entity work.swb_sc_main
    generic map (
        NLINKS => g_NLINKS_FEB_TOTL--,
    )
    port map (
        i_clk           => i_clk,
        i_reset_n       => i_resets_n(RESET_BIT_SC_MAIN),
        i_length_we     => i_writeregs(SC_MAIN_ENABLE_REGISTER_W)(0),
        i_length        => i_writeregs(SC_MAIN_LENGTH_REGISTER_W)(15 downto 0),
        i_mem_data      => i_wmem_rdata,
        o_mem_addr      => o_wmem_addr,
        o_mem_data      => o_feb_tx,
        o_done          => o_readregs(SC_MAIN_STATUS_REGISTER_R)(SC_MAIN_DONE),
        o_state         => o_readregs(SC_STATE_REGISTER_R)(27 downto 0)--,
    );

    e_sc_secondary : entity work.swb_sc_secondary
    generic map (
        NLINKS      => g_NLINKS_FEB_TOTL,
        skip_init   => g_SC_SEC_SKIP_INIT--,
    )
    port map (
        i_reset_n               => i_resets_n(RESET_BIT_SC_SECONDARY),
        i_link_enable           => i_writeregs(FEB_ENABLE_REGISTER_W)(g_NLINKS_FEB_TOTL-1 downto 0),
        i_link_data             => rx_sc,
        mem_addr_out            => o_rmem_addr,
        mem_addr_finished_out   => o_readregs(MEM_WRITEADDR_LOW_REGISTER_R)(15 downto 0),
        mem_data_out            => o_rmem_wdata,
        mem_wren                => o_rmem_we,
        stateout                => o_readregs(SC_STATE_REGISTER_R)(31 downto 28),
        i_clk                   => i_clk--,
    );


    --! Mapping Signals
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    -- mask_n
    pixel_mask_n <= x"00000000" & i_writeregs(SWB_LINK_MASK_PIXEL_REGISTER_W);
    scifi_mask_n <= x"00000000" & i_writeregs(SWB_LINK_MASK_SCIFI_REGISTER_W);

    -- farm data
    o_farm_tx(g_NLINKS_FARM_PIXEL - 1 downto 0)                                          <= pixel_farm_data;
    o_farm_tx(g_NLINKS_FARM_PIXEL + g_NLINKS_FARM_SCIFI - 1 downto g_NLINKS_FARM_PIXEL)  <= scifi_farm_data;

    -- link mapping
    gen_pixel_data_mapping : FOR i in 0 to g_NLINKS_DATA_PIXEL - 1 GENERATE
        rx_data_pixel(i)   <= rx_data(i);
    END GENERATE gen_pixel_data_mapping;
    gen_scifi_data_mapping : FOR i in g_NLINKS_DATA_PIXEL to g_NLINKS_DATA_PIXEL + g_NLINKS_DATA_SCIFI - 1 GENERATE
        rx_data_scifi(i-g_NLINKS_DATA_PIXEL)   <= rx_data(i);
    END GENERATE gen_scifi_data_mapping;

    -- counter mapping
    counter_swb(builder_counters'left downto 0) <= builder_counters;
    counter_swb(builder_counters'left+1 + counter_swb_data_pixel_ds'left+1 - 1 downto builder_counters'left+1) <= counter_swb_data_pixel_ds;
    counter_swb(builder_counters'left+1 + counter_swb_data_pixel_us'left+1 + counter_swb_data_pixel_ds'left+1 - 1 downto counter_swb_data_pixel_ds'left+1 + builder_counters'left+1) <= counter_swb_data_pixel_us;
    counter_swb(builder_counters'left+1 + counter_swb_data_scifi'left+1 + counter_swb_data_pixel_ds'left+1 + counter_swb_data_pixel_us'left+1 - 1 downto counter_swb_data_pixel_ds'left+1 + counter_swb_data_pixel_us'left+1 + builder_counters'left+1) <= counter_swb_data_scifi;

    -- DAM mapping
    builder_data    <=  data_debug_pixel(0) when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else
                        data_debug_pixel(1) when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else
                        data_debug_scifi(0) when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                        work.mu3e.LINK_IDLE;
    builder_rempty  <=  rempty_debug_pixel(0) when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else
                        rempty_debug_pixel(1) when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else
                        rempty_debug_scifi(0) when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                        '0';
    builder_data_type <= "00" when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else
                         "00" when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else
                         "01" when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                         "00";
    rack_debug_pixel(0) <=  builder_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else '0';
    rack_debug_pixel(1) <=  builder_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else '0';
    rack_debug_scifi(0) <=  builder_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else '0';


    --! SWB data path Pixel
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_data_path_pixel_ds : entity work.swb_data_path
    generic map (
        g_LOOPUP_NAME           => "intRun2021",
        g_ADDR_WIDTH            => 11,
        g_NLINKS_DATA           => g_NLINKS_DATA_PIXEL_DS,
        LINK_FIFO_ADDR_WIDTH    => 13,
        SWB_ID                  => SWB_ID,
        -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
        DATA_TYPE               => "00"--,
    )
    port map (
        -- clk and reset signals
        i_clk               => i_clk,
        i_reset_n           => i_resets_n(RESET_BIT_DATA_PATH),
        i_resets_n          => i_resets_n,

        -- link inputs
        i_rx                => rx_data_pixel(g_NLINKS_DATA_PIXEL_DS-1 downto 0),
        i_rmask_n           => pixel_mask_n(g_NLINKS_DATA_PIXEL_DS-1 downto 0),

        i_writeregs         => i_writeregs,

        o_counter           => counter_swb_data_pixel_ds,

        o_farm_data         => pixel_farm_data(0),

        i_rack_debug        => rack_debug_pixel(0),
        o_data_debug        => data_debug_pixel(0),
        o_rempty_debug      => rempty_debug_pixel(0)--;
    );

    e_swb_data_path_pixel_us : entity work.swb_data_path
    generic map (
        g_LOOPUP_NAME           => "intRun2021",
        g_ADDR_WIDTH            => 11,
        g_NLINKS_DATA           => g_NLINKS_DATA_PIXEL_US,
        LINK_FIFO_ADDR_WIDTH    => 13,
        SWB_ID                  => SWB_ID,
        -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
        DATA_TYPE               => "00"--,
    )
    port map (
        -- clk and reset signals
        i_clk               => i_clk,
        i_reset_n           => i_resets_n(RESET_BIT_DATA_PATH),
        i_resets_n          => i_resets_n,

        -- link inputs
        i_rx                => rx_data_pixel(g_NLINKS_DATA_PIXEL_US+g_NLINKS_DATA_PIXEL_DS-1 downto g_NLINKS_DATA_PIXEL_DS),
        i_rmask_n           => pixel_mask_n(g_NLINKS_DATA_PIXEL_US+g_NLINKS_DATA_PIXEL_DS-1 downto g_NLINKS_DATA_PIXEL_DS),

        i_writeregs         => i_writeregs,

        o_counter           => counter_swb_data_pixel_us,

        o_farm_data         => pixel_farm_data(1),

        i_rack_debug        => rack_debug_pixel(1),
        o_data_debug        => data_debug_pixel(1),
        o_rempty_debug      => rempty_debug_pixel(1)--;
    );


    --! SWB data path Scifi
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_data_path_pixel_scifi : entity work.swb_data_path
    generic map (
        g_LOOPUP_NAME           => "intRun2021",
        g_ADDR_WIDTH            => 11,
        g_NLINKS_DATA           => g_NLINKS_DATA_SCIFI,
        LINK_FIFO_ADDR_WIDTH    => 13,
        SWB_ID                  => SWB_ID,
        g_gen_time_merger       => false,
        -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
        DATA_TYPE               => "01"--,
    )
    port map (
        -- clk and reset signals
        i_clk               => i_clk,
        i_reset_n           => i_resets_n(RESET_BIT_DATA_PATH),
        i_resets_n          => i_resets_n,

        -- link inputs
        i_rx                => rx_data_scifi(g_NLINKS_DATA_SCIFI-1 downto 0),
        i_rmask_n           => scifi_mask_n(g_NLINKS_DATA_SCIFI-1 downto 0),

        i_writeregs         => i_writeregs,

        o_counter           => counter_swb_data_scifi,

        o_farm_data         => scifi_farm_data(0),

        i_rack_debug        => rack_debug_scifi(0),
        o_data_debug        => data_debug_scifi(0),
        o_rempty_debug      => rempty_debug_scifi(0)--;
    );


    --! event builder used for the debug readout on the SWB
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_event_builder : entity work.swb_midas_event_builder
    port map (
        i_rx                => builder_data,
        i_rempty            => builder_rempty,

        i_get_n_words       => i_writeregs(GET_N_DMA_WORDS_REGISTER_W),
        i_dmamemhalffull    => i_dmamemhalffull,
        i_wen               => i_writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE),
        i_data_type         => builder_data_type,

        o_data              => o_dma_data,
        o_wen               => o_dma_wren,
        o_ren               => builder_rack,
        o_endofevent        => o_endofevent,
        o_dma_cnt_words     => o_readregs(DMA_CNT_WORDS_REGISTER_R),
        o_done              => o_readregs(EVENT_BUILD_STATUS_REGISTER_R)(EVENT_BUILD_DONE),

        --! bank_builder_idle_not_header
        --! bank_builder_skip_event_dma
        --! bank_builder_event_dma
        --! bank_builder_tag_fifo_full
        o_counters          => builder_counters,

        i_reset_n           => i_reset_n,
        i_clk               => i_clk--,
    );

end architecture;
