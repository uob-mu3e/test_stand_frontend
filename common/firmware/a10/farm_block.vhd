-------------------------------------------------------
--! @farm_block.vhd
--! @brief the farm_block can be used
--! for the development board mainly it includes 
--! the datapath which includes merging detector data
--! from multiple SWBs.
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity farm_block is
generic (
    g_NLINKS_TOTL  : positive := 16;
    g_NLINKS_PIXEL : positive := 8;
    g_NLINKS_SCIFI : positive := 8--;
);
port (

    --! links to/from FEBs
    i_rx                : in  work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    i_rx_k              : in  work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    o_tx                : out work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    o_tx_k              : out work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);

    --! PCIe registers / memory
    i_writeregs_pcie    : in  work.util.slv32_array_t(63 downto 0);
    i_writeregs_link    : in  work.util.slv32_array_t(63 downto 0);
    i_writeregs_ddr     : in  work.util.slv32_array_t(63 downto 0);   
    
    o_readregs_pcie     : out work.util.slv32_array_t(63 downto 0);
    o_readregs_link     : out work.util.slv32_array_t(63 downto 0);
    o_readregs_ddr      : out work.util.slv32_array_t(63 downto 0);   

    i_resets_n_pcie     : in  std_logic_vector(31 downto 0);
    i_resets_n_link     : in  std_logic_vector(31 downto 0);
    i_resets_n_ddr      : in  std_logic_vector(31 downto 0);
    
    -- TODO: write status readout entity with ADDR to PCIe REGS and mapping to one counter REG
    o_counter           : out work.util.slv32_array_t(19 downto 0);
    o_status            : out std_logic_vector(31 downto 0);

    i_dmamemhalffull    : in  std_logic;
    o_dma_wren          : out std_logic;
    o_endofevent        : out std_logic;
    o_dma_data          : out std_logic_vector(255 downto 0);
    
    --! 250 MHz clock pice / reset_n
    i_reset_n_250_pcie  : in  std_logic;
    i_clk_250_pcie      : in  std_logic;   

    --! 250 MHz clock link / reset_n
    i_reset_n_250_link  : in  std_logic;
    i_clk_250_link      : in  std_logic;
    
    -- Interface to memory bank A
    o_A_mem_clk           : out   std_logic;
    A_mem_ck            : out   std_logic_vector(0 downto 0);                      -- mem_ck
    A_mem_ck_n          : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
    A_mem_a             : out   std_logic_vector(15 downto 0);                     -- mem_a
    A_mem_ba            : out   std_logic_vector(2 downto 0);                      -- mem_ba
    A_mem_cke           : out   std_logic_vector(0 downto 0);                      -- mem_cke
    A_mem_cs_n          : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
    A_mem_odt           : out   std_logic_vector(0 downto 0);                      -- mem_odt
    A_mem_reset_n       : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
    A_mem_we_n          : out   std_logic_vector(0 downto 0);                      -- mem_we_n
    A_mem_ras_n         : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
    A_mem_cas_n         : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
    A_mem_dqs           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
    A_mem_dqs_n         : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
    A_mem_dq            : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
    A_mem_dm            : out   std_logic_vector(7 downto 0);                      -- mem_dm
    A_oct_rzqin         : in    std_logic                      := 'X';             -- oct_rzqin
    A_pll_ref_clk       : in    std_logic                      := 'X';             -- clk

    -- Interface to memory bank B
    o_B_mem_clk           : out   std_logic;
    B_mem_ck            : out   std_logic_vector(0 downto 0);                      -- mem_ck
    B_mem_ck_n          : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
    B_mem_a             : out   std_logic_vector(15 downto 0);                     -- mem_a
    B_mem_ba            : out   std_logic_vector(2 downto 0);                      -- mem_ba
    B_mem_cke           : out   std_logic_vector(0 downto 0);                      -- mem_cke
    B_mem_cs_n          : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
    B_mem_odt           : out   std_logic_vector(0 downto 0);                      -- mem_odt
    B_mem_reset_n       : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
    B_mem_we_n          : out   std_logic_vector(0 downto 0);                      -- mem_we_n
    B_mem_ras_n         : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
    B_mem_cas_n         : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
    B_mem_dqs           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
    B_mem_dqs_n         : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
    B_mem_dq            : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
    B_mem_dm            : out   std_logic_vector(7 downto 0);                      -- mem_dm
    B_oct_rzqin         : in    std_logic                      := 'X';             -- oct_rzqin
    B_pll_ref_clk       : in    std_logic                      := 'X'              -- clk
    
);
end entity;

--! @brief arch definition of the farm_block
--! @details the farm_block can be used
--! for the development board mainly it includes 
--! the datapath which includes merging detector data
--! from multiple SWBs.
--! scifi, down and up stream pixel/tiles)
architecture arch of farm_block is

    --! mapping signals
    signal rx : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_k : work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    
    --! data gen pixel
    signal full_pixel, w_pixel_en, r_pixel_en, empty_pixel : std_logic;
    signal w_pixel, r_pixel : std_logic_vector(g_NLINKS_PIXEL * 38 - 1 downto 0);
    signal link_data_pixel : std_logic_vector(g_NLINKS_PIXEL * 32 - 1 downto 0);
    signal link_datak_pixel : std_logic_vector(g_NLINKS_PIXEL * 4 - 1 downto 0);
    
    --! data gen scifi
    signal full_scifi, w_scifi_en, r_scifi_en, empty_scifi : std_logic;
    signal w_scifi, r_scifi : std_logic_vector(g_NLINKS_SCIFI * 38 - 1 downto 0);
    signal link_data_scifi : std_logic_vector(g_NLINKS_SCIFI * 32 - 1 downto 0);
    signal link_datak_scifi : std_logic_vector(g_NLINKS_SCIFI * 4 - 1 downto 0);
    
    --! link to fifo
    signal pixel_data, scifi_data : std_logic_vector(257 downto 0);
    signal pixel_empty, pixel_ren, scifi_empty, scifi_ren, pixel_error_link_to_fifo, scifi_error_link_to_fifo : std_logic;
    
    --! midas event builder
    signal data_in : std_logic_vector(511 downto 0);
    signal data_wen : std_logic;
    signal event_ts : std_logic_vector(47 downto 0);
    signal pixel_next_farm : std_logic_vector(g_NLINKS_PIXEL * 32 + 1 downto 0);
    signal scifi_next_farm : std_logic_vector(g_NLINKS_SCIFI * 32 + 1 downto 0);
    signal pixel_next_farm_wen, scifi_next_farm_wen : std_logic;
    
    --! link to next farm
    signal tx_wen, tx_empty_pixel, tx_empty_scifi : std_logic;
    signal tx_data_pixel, tx_data_scifi, tx_q_pixel, tx_q_scifi : std_logic_vector(257 downto 0);
        
    --! farm data path
    signal ddr_ready        : std_logic;
    signal A_mem_clk        : std_logic;
    signal A_mem_ready      : std_logic;
    signal A_mem_calibrated : std_logic;
    signal A_mem_addr       : std_logic_vector(25 downto 0);
    signal A_mem_data       : std_logic_vector(511 downto 0);
    signal A_mem_write      : std_logic;
    signal A_mem_read       : std_logic;
    signal A_mem_q          : std_logic_vector(511 downto 0);
    signal A_mem_q_valid    : std_logic;
    signal B_mem_clk        : std_logic;
    signal B_mem_ready      : std_logic;
    signal B_mem_calibrated : std_logic;
    signal B_mem_addr       : std_logic_vector(25 downto 0);
    signal B_mem_data       : std_logic_vector(511 downto 0);
    signal B_mem_write      : std_logic;
    signal B_mem_read       : std_logic;
    signal B_mem_q          : std_logic_vector(511 downto 0);
    signal B_mem_q_valid    : std_logic;
    
    --! counters
    --! 0: fifo sync_almost_full (pixel)
    --! 1: fifo sync_wrfull (pixel)
    --! 2: # of overflow event (pixel)
    --! 3: cnt events (pixel)
    --! 4: fifo sync_almost_full (scifi)
    --! 5: fifo sync_wrfull (scifi)
    --! 6: # of overflow event (scifi)
    --! 7: cnt events (scifi)
    signal counter_link_to_fifo : work.util.slv32_array_t(7 downto 0);
    --! 0: cnt_idle_not_header_pixel
    --! 1: cnt_idle_not_header_scifi
    --! 2: bank_builder_ram_full
    --! 3: bank_builder_tag_fifo_full
    --! 4: # events written to RAM (one subheader of Pixel and Scifi)
    --! 5: # 256b pixel x"FFF"
    --! 6: # 256b scifi x"FFF"
    --! 7: # 256b pixel written to link
    --! 8: # 256b scifi written to link
    signal counter_midas_event_builder : work.util.slv32_array_t(8 downto 0);
    --! 0: cnt_skip_event_dma
    --! 1: A_almost_full
    --! 2: B_almost_full
    --! 3: i_dmamemhalffull
    signal counter_ddr : work.util.slv32_array_t(3 downto 0);


begin

    --! @brief data path of the Farm board
    --! @details the data path of the farm board is first aligning the 
    --! data from the SWB and is than grouping them into Pixel, Scifi and Tiles.
    --! The data is saved according to the sub-header time in the DDR3 memory.
    --! Via MIDAS one can select how much data one wants to readout from the DDR3 memory
    --! the stored data is marked and than forworded to the next farm pc
    
    
    --! status counter / outputs
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    gen_link_to_fifo_cnt : FOR I in 0 to 7 GENERATE
        o_counter(I) <= counter_link_to_fifo(I);
    END GENERATE;
    gen_midas_event_cnt : FOR I in 8 to 16 GENERATE
        o_counter(I) <= counter_midas_event_builder(I-8);
    END GENERATE;
    gen_ddr_cnt : FOR I in 16 to 19 GENERATE
        o_counter <= counter_ddr(I-16);
    END GENeRATE;
    
    o_status(0) <= pixel_error_link_to_fifo;
    o_status(1) <= scifi_error_link_to_fifo;
    
    o_A_mem_clk <= A_mem_clk;
    o_B_mem_clk <= B_mem_clk;
    
    --! SWB Data Generation
    --! generate data in the format from the SWB
    --! PIXEL, SCIFI --> Int Run 2021
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    -- gen pixel data
    e_data_gen_pixel : entity work.data_generator_merged_data
    generic map(
        NLINKS => g_NLINKS_PIXEL,
        go_to_sh => 2,
        go_to_trailer => 3--,
    )
    port map(
        i_clk       => i_clk_250_link,
        i_reset_n   => i_reset_n_250_link,
        i_en        => not full_pixel,
        i_sd        => x"00000002",
        o_data      => w_pixel,
        o_data_we   => w_pixel_en,
        o_state     => open--,
    );
    
    e_merger_fifo_pixel : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => g_NLINKS_PIXEL * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_pixel,
        wrreq           => w_pixel_en,
        rdreq           => r_pixel_en,
        clock           => i_clk_250_link,
        q               => r_pixel,
        full            => full_pixel,
        empty           => empty_pixel,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => not i_reset_n_250_link--,
    );
    
    e_swb_data_merger_pixel : entity work.swb_data_merger
    generic map (
        NLINKS      => g_NLINKS_PIXEL,
        DATA_TYPE   => x"01"--,
    )
    port map (
        i_reset_n   => i_reset_n_250_link,
        i_clk       => i_clk_250_link,
        
        i_data      => r_pixel,
        i_empty     => empty_pixel,
        
        o_ren       => r_pixel_en,
        o_wen       => open,
        o_data      => link_data_pixel,
        o_datak     => link_datak_pixel--,
    );
    
    -- gen scifi data
    e_data_gen_scifi : entity work.data_generator_merged_data
    generic map(
        go_to_sh => 2,
        go_to_trailer => 3--,
    )
    port map(
        i_clk       => i_clk_250_link,
        i_reset_n   => i_reset_n_250_link,
        i_en        => not full_scifi,
        i_sd        => x"00000002",
        o_data      => w_scifi,
        o_data_we   => w_scifi_en,
        o_state     => open--,
    );
    
    e_merger_fifo_scifi : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => NLINKS * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_scifi,
        wrreq           => w_scifi_en,
        rdreq           => r_scifi_en,
        clock           => i_clk_250_link,
        q               => r_scifi,
        full            => full_scifi,
        empty           => empty_scifi,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => not i_reset_n_250_link--,
    );
    
    e_swb_data_merger_scifi : entity work.swb_data_merger
    generic map (
        NLINKS      => NLINKS,
        DATA_TYPE   => x"02"--,
    )
    port map (
        i_reset_n   => i_reset_n_250_link,
        i_clk       => i_clk_250_link,
        
        i_data      => r_scifi,
        i_empty     => empty_scifi,
        
        o_ren       => r_scifi_en,
        o_wen       => open,
        o_data      => link_data_scifi,
        o_datak     => link_datak_scifi--,
    );
    
    --! map links pixel / scifi
    --! NOTE: we say that g_NLINKS_PIXEL = g_NLINKS_SCIFI at the moment
    gen_link_data : FOR I in 0 to g_NLINKS_PIXEL - 1 GENERATE
    
        process(i_clk_250_link, i_reset_n_250_link)
        begin
        if ( i_reset_n_250_link = '0' ) then
            rx(I)   <= (others => '0');
            rx_k(I) <= (others => '0');
            rx(I+g_NLINKS_SCIFI)   <= (others => '0');
            rx_k(I+g_NLINKS_SCIFI) <= (others => '0');
        elsif ( rising_edge( i_clk_250_link ) ) then
            if ( i_writeregs_link(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK) = '1' ) then
                rx(I)   <= link_data_pixel(I*32 + 31 downto I*32);
                rx_k(I) <= link_datak_pixel(I*4 + 3 downto I*4);
                rx(I+g_NLINKS_SCIFI) <= link_data_scifi(I*32 + 31 downto I*32);
                rx_k(I+g_NLINKS_SCIFI) <= link_datak_scifi(I*4 + 3 downto I*4);
            else
                rx(I)   <= i_rx(I);
                rx_k(I) <= i_rx_k(I);
                rx(I+g_NLINKS_SCIFI)   <= i_rx(I+g_NLINKS_SCIFI);
                rx_k(I+g_NLINKS_SCIFI) <= i_rx_k(I+g_NLINKS_SCIFI);
            end if;
        end if;
        end process;
        
    END GENERATE gen_link_data;
    

    --! Link Alignment
    --! align data according to detector data
    --! two types of data will be extracted from the links
    --! PIXEL, SCIFI --> Int Run 2021
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_link_to_fifo : entity work.farm_link_to_fifo
    generic map (
        g_NLINKS_SWB_TOTL   => g_NLINKS_SWB_TOTL,
        N_PIXEL             => g_NLINKS_PIXEL,
        N_SCIFI             => g_NLINKS_SCIFI--,
    )
    port map (
        i_rx                => rx,
        i_rx_k              => rx_k,

        -- pixel data
        o_pixel             => pixel_data,
        o_empty_pixel       => pixel_empty, 
        i_ren_pixel         => pixel_ren,
        o_error_pixel       => pixel_error_link_to_fifo,
        
        -- scifi data
        o_scifi             => scifi_data,
        o_empty_scifi       => scifi_empty,
        i_ren_scifi         => scifi_ren,
        o_error_scifi       => scifi_error_link_to_fifo,
    
        --! status counters 
        --! 0: fifo sync_almost_full (pixel)
        --! 1: fifo sync_wrfull (pixel)
        --! 2: # of overflow event (pixel)
        --! 3: cnt events (pixel)
        --! 4: fifo sync_almost_full (scifi)
        --! 5: fifo sync_wrfull (scifi)
        --! 6: # of overflow event (scifi)
        --! 7: cnt events (scifi)
        o_counter           => counter_link_to_fifo,
        
        i_clk_250_link      => i_clk_250_link,
        i_reset_n_250_link  => i_reset_n_250_link,
        
        i_clk_250           => i_clk_250_pcie,
        i_reset_n_250       => i_resets_n_pcie(RESET_BIT_FARM_DATA_PATH)--,
    );
    

    --! Farm MIDAS Event Builder
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_midas_event_builder : entity work.farm_midas_event_builder
    generic map (
        g_NLINKS_SWB_TOTL => g_NLINKS_SWB_TOTL,
        N_PIXEL           => g_NLINKS_PIXEL,
        N_SCIFI           => g_NLINKS_SCIFI,
        RAM_ADDR          => 12--,
    )
    port map (
        i_pixel         => pixel_data,
        i_empty_pixel   => pixel_empty,
        o_ren_pixel     => pixel_ren,
    
        i_scifi         => scifi_data,
        i_empty_scifi   => scifi_empty,
        o_ren_scifi     => scifi_ren,
        
        i_farm_id       => i_writeregs_pcie(FARM_ID_REGISTER_W)

        -- DDR
        o_data          => data_in,
        o_wen           => data_wen,
        o_event_ts      => event_ts,
        i_ddr_ready     => ddr_ready,
    
        -- Link data
        o_pixel         => pixel_next_farm,
        o_wen_pixel     => pixel_next_farm_wen,
    
        o_scifi         => scifi_next_farm,
        o_wen_scifi     => scifi_next_farm_wen,

        --! 0: cnt_idle_not_header_pixel
        --! 1: cnt_idle_not_header_scifi
        --! 2: bank_builder_ram_full
        --! 3: bank_builder_tag_fifo_full
        --! 4: # events written to RAM (one subheader of Pixel and Scifi)
        --! 5: # 256b pixel x"FFF"
        --! 6: # 256b scifi x"FFF"
        --! 7: # 256b pixel written to link
        --! 8: # 256b scifi written to link
        o_counters      => counter_midas_event_builder,

        i_reset_n_250   => i_reset_n_250_pcie,
        i_clk_250       => i_clk_250_pcie--,
    );
    
    --! map links pixel / scifi
    --! NOTE: we say that g_NLINKS_PIXEL = g_NLINKS_SCIFI at the moment
    process(i_clk_250_pcie, i_reset_n_250_pcie)
    begin
    if ( i_reset_n_250_pcie = '0' ) then
        tx_data_pixel   <= (others => '0');
        tx_data_scifi   <= (others => '0');
        tx_wen          <= '0';
        --
    elsif ( rising_edge( i_clk_250_pcie ) ) then
        tx_wen <= '1';
        if ( pixel_next_farm_wen = '1' ) then
            tx_data_pixel <= pixel_next_farm;
        else
            tx_data_pixel(g_NLINKS_PIXEL * 32 + 1 downto g_NLINKS_PIXEL * 32) <= "01";
            for i in 0 to g_NLINKS_PIXEL - 1 loop
                tx_data_pixel(i * 32 + 31 downto i * 32) <= x"000000" & work.util.D28_5;
            end loop;
        end if;
        
        if ( scifi_next_farm_wen = '1' ) then
            tx_data_scifi <= scifi_next_farm;
        else
            tx_data_scifi(g_NLINKS_SCIFI * 32 + 1 downto g_NLINKS_SCIFI * 32) <= "01";
            for i in 0 to g_NLINKS_SCIFI - 1 loop
                tx_data_scifi(i * 32 + 31 downto i * 32) <= x"000000" & work.util.D28_5;
            end loop;
        end if;
    end if;
    end process;
    
    e_sync_fifo_tx_pixel : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 6,
        DATA_WIDTH  => 258,
        DEVICE      => "Arria 10"--,
    )
    port map (
        data        => tx_data_pixel,
        wrreq       => tx_wen,
        rdreq       => not tx_empty_pixel,
        wrclk       => i_clk_250_pcie,
        rdclk       => i_clk_250_link,
        q           => tx_q_pixel,
        rdempty     => tx_empty_pixel,
        aclr        => not i_reset_n_250_link--,
    );
    
    e_sync_fifo_tx_scifi : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 6,
        DATA_WIDTH  => 258,
        DEVICE      => "Arria 10"--,
    )
    port map (
        data        => tx_data_scifi,
        wrreq       => tx_wen,
        rdreq       => not tx_empty_scifi,
        wrclk       => i_clk_250_pcie,
        rdclk       => i_clk_250_link,
        q           => tx_q_scifi,
        rdempty     => tx_empty_scifi,
        aclr        => not i_reset_n_250_link--,
    );
    
    gen_tx_data : FOR I in 0 to g_NLINKS_PIXEL - 1 GENERATE
    
        process(i_clk_250_link, i_reset_n_250_link)
        begin
        if ( i_reset_n_250_link = '0' ) then
            o_tx(I)                 <= (others => '0');
            o_tx_k(I)               <= (others => '0');
            o_tx(I+g_NLINKS_SCIFI)  <= (others => '0');
            o_tx_k(I+g_NLINKS_SCIFI)<= (others => '0');
            --
        elsif ( rising_edge( i_clk_250_link ) ) then
            
            if ( tx_empty_pixel = '1' ) then
                o_tx(I)   <= x"000000" & work.util.D28_5;
                o_tx_k(I) <= "0001";
            elsif ( tx_q_pixel(g_NLINKS_PIXEL * 32 + 1 downto g_NLINKS_PIXEL * 32) == "00" ) then
                o_tx(I)   <= tx_q_pixel(I * 32 + 31 downto I * 32)
                o_tx_k(I) <= "0000";
            else
                o_tx(I)   <= tx_q_pixel(I * 32 + 31 downto I * 32)
                o_tx_k(I) <= "0001";
            end if;
            
            if ( tx_empty_scifi = '1' ) then
                o_tx(I+g_NLINKS_SCIFI)   <= x"000000" & work.util.D28_5;
                o_tx_k(I+g_NLINKS_SCIFI) <= "0001";
            elsif ( tx_q_scifi(g_NLINKS_PIXEL * 32 + 1 downto g_NLINKS_PIXEL * 32) == "00" ) then
                o_tx(I+g_NLINKS_SCIFI)   <= tx_q_scifi(I * 32 + 31 downto I * 32)
                o_tx_k(I+g_NLINKS_SCIFI) <= "0000";
            else
                o_tx(I+g_NLINKS_SCIFI)   <= tx_q_scifi(I * 32 + 31 downto I * 32)
                o_tx_k(I+g_NLINKS_SCIFI) <= "0001";
            end if;
        end if;
        end process;
        
    END GENERATE gen_tx_data;
    
    --! Farm Data Path
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_data_path : entity work.farm_data_path 
    port map(
        reset_n         => i_resets_n_pcie(RESET_BIT_DDR3),
        reset_n_ddr3    => i_resets_n_ddr(RESET_BIT_DDR3),
        dataclk         => i_clk_250_pcie,

        -- Input from merging (first board) or links (subsequent boards)
        data_in         => data_in,
        data_en         => data_wen, 
        ts_in           => event_ts(35 downto 4), -- 3:0 -> hit, 9:0 -> sub header
        o_ddr_ready     => ddr_ready,

        -- Input from PCIe demanding events
        pcieclk        => i_clk_250_pcie,
        ts_req_A       => writeregs_ddr3(DATA_REQ_A_W),
        req_en_A       => regwritten_C(DATA_REQ_A_W),
        ts_req_B       => writeregs_ddr3(DATA_REQ_B_W),
        req_en_B       => regwritten_C(DATA_REQ_B_W),
        tsblock_done   => writeregs_ddr3(DATA_TSBLOCK_DONE_W)(15 downto 0),
        tsblocks       => readregs_ddr3(DATA_TSBLOCKS_R),

        -- Output to DMA
        dma_data_out    => o_dma_data,
        dma_data_en     => o_dma_wren,
        dma_eoe         => o_endofevent,
        i_dmamemhalffull=> i_dmamemhalffull,
        i_num_req_events=> i_writeregs_pcie(FARM_REQ_EVENTS_W),
        o_dma_done      => o_readregs_pcie(EVENT_BUILD_STATUS_REGISTER_R)(EVENT_BUILD_DONE),
        i_dma_wen       => i_writeregs_pcie(DMA_REGISTER_W)(DMA_BIT_ENABLE),
        
        --! status counters 
        --! 0: cnt_skip_event_dma
        --! 1: A_almost_full
        --! 2: B_almost_full
        --! 3: i_dmamemhalffull
        o_counters      => counter_ddr;

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
    
    
    --! Farm DDR Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_ddr3_block : entity work.ddr3_block 
    port map(
        reset_n             => resets_n_ddr3(RESET_BIT_DDR3),
        
        -- Control and status registers
        ddr3control         => i_writeregs_ddr(DDR3_CONTROL_W),
        ddr3status          => o_readregs_ddr(DDR3_STATUS_R),

        -- A interface
        A_ddr3clk           => A_mem_clk,
        A_ddr3calibrated    => A_mem_calibrated,
        A_ddr3ready         => A_mem_ready,
        A_ddr3addr          => A_mem_addr,
        A_ddr3datain        => A_mem_data,
        A_ddr3dataout       => A_mem_q,
        A_ddr3_write        => A_mem_write,
        A_ddr3_read         => A_mem_read,
        A_ddr3_read_valid   => A_mem_q_valid,
        
        -- B interface
        B_ddr3clk           => B_mem_clk,
        B_ddr3calibrated    => B_mem_calibrated,
        B_ddr3ready         => B_mem_ready,
        B_ddr3addr          => B_mem_addr,
        B_ddr3datain        => B_mem_data,
        B_ddr3dataout       => B_mem_q,
        B_ddr3_write        => B_mem_write,
        B_ddr3_read         => B_mem_read,
        B_ddr3_read_valid   => B_mem_q_valid,
        
        -- Error counters
        errout              => readregs_ddr3(DDR3_ERR_R),

        -- Interface to memory bank A
        A_mem_ck            => A_mem_ck,
        A_mem_ck_n          => A_mem_ck_n,
        A_mem_a             => A_mem_a,
        A_mem_ba            => A_mem_ba,
        A_mem_cke           => A_mem_cke,
        A_mem_cs_n          => A_mem_cs_n,
        A_mem_odt           => A_mem_odt,
        A_mem_reset_n(0)    => A_mem_reset_n(0),      
        A_mem_we_n(0)       => A_mem_we_n(0),
        A_mem_ras_n(0)      => A_mem_ras_n(0),
        A_mem_cas_n(0)      => A_mem_cas_n(0),
        A_mem_dqs           => A_mem_dqs,
        A_mem_dqs_n         => A_mem_dqs_n,
        A_mem_dq            => A_mem_dq,
        A_mem_dm            => A_mem_dm,
        A_oct_rzqin         => A_oct_rzqin,
        A_pll_ref_clk       => A_pll_ref_clk,
        
        -- Interface to memory bank B
        B_mem_ck            => B_mem_ck,
        B_mem_ck_n          => B_mem_ck_n,
        B_mem_a             => B_mem_a,
        B_mem_ba            => B_mem_ba,
        B_mem_cke           => B_mem_cke,
        B_mem_cs_n          => B_mem_cs_n,
        B_mem_odt           => B_mem_odt,
        B_mem_reset_n(0)    => B_mem_reset_n(0),      
        B_mem_we_n(0)       => B_mem_we_n(0),
        B_mem_ras_n(0)      => B_mem_ras_n(0),
        B_mem_cas_n(0)      => B_mem_cas_n(0),
        B_mem_dqs           => B_mem_dqs,
        B_mem_dqs_n         => B_mem_dqs_n,
        B_mem_dq            => B_mem_dq,
        B_mem_dm            => B_mem_dm,
        B_oct_rzqin         => B_oct_rzqin,
        B_pll_ref_clk       => B_pll_ref_clk--,
     );

end architecture;
