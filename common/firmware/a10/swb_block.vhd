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

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.pcie_components.all;
use work.mudaq_registers.all;

entity swb_block is
generic (
    g_NLINKS_TOTL    : integer := 16;
    g_NLINKS_DATA    : integer := 16--;
);
port (

    --! links from FEBs
    i_rx                 : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    i_rx_k               : work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    o_tx                 : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    o_tx_k               : work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);

    --! PCIe registers / memory
    i_writeregs_250      : in reg32array;
    i_writeregs_156      : in reg32array;
    
    o_readregs_250       : out reg32array;
    o_readregs_156       : out reg32array;

    i_resets_n_156       : in std_logic_vector(31 downto 0);
    i_resets_n_250       : in std_logic_vector(31 downto 0);

    i_wmem_rdata         : in std_logic_vector(31 downto 0);
    o_wmem_addr          : out std_logic_vector(15 downto 0);

    o_rmem_wdata         : out std_logic_vector(31 downto 0);
    o_rmem_addr          : out std_logic_vector(15 downto 0);
    o_rmem_we            : out std_logic;

    --! pcie clk / reset_n
    i_refclk_pcie0       : in std_logic; --PCIE_REFCLK_p

    --! 250 MHz clock / reset_n
    i_reset_n_250        : in std_logic;
    i_clk_250            : in std_logic;    

    --! 156 MHz clock / reset_n
    i_reset_n_156        : in std_logic;
    i_clk_156            : in std_logic--;
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

    --! mapping signals
    --! fiber link_mapping(0)=1 
    --! Fiber QSFPA.1 is mapped to first(0) link
    type mapping_t is array(natural range <>) of integer;
    constant link_mapping : mapping_t(NLINKS_DATA-1 downto 0) := (1,2,4);

    --! demerged FEB links
    signal rx_data      : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_data_k    : work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_sc        : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_sc_k      : work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_rc        : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_rc_k      : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);


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

    --! demerge data
    --! three types of data will be extracted from the links
    --! data => detector data
    --! sc => slow control packages
    --! rc => runcontrol packages
    g_demerge: for i in g_NLINKS_TOTL-1 downto 0 generate
        e_data_demerge : entity work.data_demerge
        port map(
            i_clk               => i_clk_156,
            i_reset             => not i_resets_n_156(RESET_BIT_EVENT_COUNTER),
            i_aligned           => '1',
            i_data              => i_rx(i),
            i_datak             => i_rx_k(i),
            i_fifo_almost_full  => '0',--link_fifo_almost_full(i),
            o_data              => rx_data(i),
            o_datak             => rx_data_k(i),
            o_sc                => rx_sc(i),
            o_sck               => rx_sc_k(i),
            o_rc                => rx_rc(i),
            o_rck               => rx_rc_k(i),
            o_fpga_id           => open--,
        );
    end generate;

    --! run control used by MIDAS
    e_run_control : entity work.run_control
    generic map (
        N_LINKS_g              => g_NLINKS_TOTL--,
    )
    port map (
        i_reset_ack_seen_n     => i_resets_n_156(RESET_BIT_RUN_START_ACK),
        i_reset_run_end_n      => i_resets_n_156(RESET_BIT_RUN_END_ACK),
        i_buffers_empty        => (others => '1'), -- TODO: connect buffers emtpy from dma here
        o_feb_merger_timeout   => o_readregs_156(CNT_FEB_MERGE_TIMEOUT_R),
        i_aligned              => (others => '1'),
        i_data                 => rx_rc,
        i_datak                => rx_rc_k,
        i_link_enable          => i_writeregs_156(FEB_ENABLE_REGISTER_W),
        i_addr                 => i_writeregs_156(RUN_NR_ADDR_REGISTER_W), -- ask for run number of FEB with this addr.
        i_run_number           => i_writeregs_156(RUN_NR_REGISTER_W)(23 downto 0),
        o_run_number           => o_readregs_156(RUN_NR_REGISTER_R), -- run number of i_addr
        o_runNr_ack            => o_readregs_156(RUN_NR_ACK_REGISTER_R), -- which FEBs have responded with run number in i_run_number
        o_run_stop_ack         => o_readregs_156(RUN_STOP_ACK_REGISTER_R),
        i_clk                  => i_clk_156--,
    );

    --! SWB slow control
    e_sc_main : entity work.sc_main
    generic map (
        NLINKS => g_NLINKS_TOTL
    )
    port map (
        i_clk           => i_clk_156,
        i_reset_n       => i_resets_n_156(RESET_BIT_SC_MAIN),
        i_length_we     => i_writeregs_156(SC_MAIN_ENABLE_REGISTER_W)(0),
        i_length        => i_writeregs_156(SC_MAIN_LENGTH_REGISTER_W)(15 downto 0),
        i_mem_data      => i_wmem_rdata,
        o_mem_addr      => o_wmem_addr,
        o_mem_data      => o_tx,
        o_mem_datak     => o_tx_k,
        o_done          => o_readregs_156(SC_MAIN_STATUS_REGISTER_R)(SC_MAIN_DONE),
        o_state         => o_readregs_156(SC_STATE_REGISTER_R)(27 downto 0)--,
    );
    
    e_sc_secondary : entity work.sc_secondary
    generic map (
        NLINKS => g_NLINKS_TOTL
    )
    port map (
        reset_n                 => i_resets_n_156(RESET_BIT_SC_SECONDARY),
        i_link_enable           => i_writeregs_156(FEB_ENABLE_REGISTER_W)(NLINKS_TOTL-1 downto 0),
        link_data_in            => rx_sc,
        link_data_in_k          => rx_sc_k,
        mem_addr_out            => o_rmem_addr,
        mem_addr_finished_out   => o_readregs_156(MEM_WRITEADDR_LOW_REGISTER_R)(15 downto 0),
        mem_data_out            => o_rmem_wdata,
        mem_wren                => o_rmem_we,
        stateout                => o_readregs_156(SC_STATE_REGISTER_R)(31 downto 28),
        clk                     => i_clk_156--,
    );

    --! assign long vectors for used fibers. 
    --! wired to run_control, sc, data receivers
    g_assign_usedlinks: for i in g_NLINKS_DATA-1 downto 0 generate
       rx_mapped_data_v(32*(i+1)-1 downto 32*i) <= rx_data(link_mapping(i));
       rx_mapped_datak_v(4*(i+1)-1 downto  4*i) <= rx_datak(link_mapping(i));
       rx_mapped_linkmask(i) <= i_writeregs_156(FEB_ENABLE_REGISTER_W)(link_mapping(i));
    end generate;

    

    -------- Event Builder --------

    e_data_gen : entity work.data_generator_a10
    generic map(
        go_to_trailer => 4,
        go_to_sh => 3--,
    )
    port map (
        reset               => resets(RESET_BIT_DATAGEN),
        enable_pix          => writeregs_slow(DATAGENERATOR_REGISTER_W)(DATAGENERATOR_BIT_ENABLE),
        i_dma_half_full     => dmamemhalffull_tx,
        random_seed         => (others => '1'),
        data_pix_generated  => data_pix_generated,
        datak_pix_generated => datak_pix_generated,
        data_pix_ready      => data_pix_ready,
        start_global_time   => (others => '0'),
        slow_down           => writeregs_slow(DATAGENERATOR_DIVIDER_REGISTER_W),
        state_out           => state_out_datagen,
        clk                 => clk_156--,
    );




end architecture;
