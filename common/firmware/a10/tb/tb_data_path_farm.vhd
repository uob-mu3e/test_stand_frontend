library ieee;
use ieee.std_logic_1164.all;
package cmp is

component ddr3_if is
    port (
        amm_ready_0         : out   std_logic;                                         -- waitrequest_n
        amm_read_0          : in    std_logic                      := 'X';             -- read
        amm_write_0         : in    std_logic                      := 'X';             -- write
        amm_address_0       : in    std_logic_vector(25 downto 0)  := (others => 'X'); -- address
        amm_readdata_0      : out   std_logic_vector(511 downto 0);                    -- readdata
        amm_writedata_0     : in    std_logic_vector(511 downto 0) := (others => 'X'); -- writedata
        amm_burstcount_0    : in    std_logic_vector(6 downto 0)   := (others => 'X'); -- burstcount
        amm_byteenable_0    : in    std_logic_vector(63 downto 0)  := (others => 'X'); -- byteenable
        amm_readdatavalid_0 : out   std_logic;                                         -- readdatavalid
        emif_usr_clk        : out   std_logic;                                         -- clk
        emif_usr_reset_n    : out   std_logic;                                         -- reset_n
        global_reset_n      : in    std_logic                      := 'X';             -- reset_n
        mem_ck              : out   std_logic_vector(0 downto 0);                      -- mem_ck
        mem_ck_n            : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
        mem_a               : out   std_logic_vector(15 downto 0);                     -- mem_a
        mem_ba              : out   std_logic_vector(2 downto 0);                      -- mem_ba
        mem_cke             : out   std_logic_vector(0 downto 0);                      -- mem_cke
        mem_cs_n            : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
        mem_odt             : out   std_logic_vector(0 downto 0);                      -- mem_odt
        mem_reset_n         : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
        mem_we_n            : out   std_logic_vector(0 downto 0);                      -- mem_we_n
        mem_ras_n           : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
        mem_cas_n           : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
        mem_dqs             : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
        mem_dqs_n           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
        mem_dq              : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
        mem_dm              : out   std_logic_vector(7 downto 0);                      -- mem_dm
        oct_rzqin           : in    std_logic                      := 'X';             -- oct_rzqin
        pll_ref_clk         : in    std_logic                      := 'X';             -- clk
        local_cal_success   : out   std_logic;                                         -- local_cal_success
        local_cal_fail      : out   std_logic                                          -- local_cal_fail
    );
end component ddr3_if;

component ddr4_if is
    port (
        amm_ready_0         : out   std_logic;                                         -- waitrequest_n
        amm_read_0          : in    std_logic                      := 'X';             -- read
        amm_write_0         : in    std_logic                      := 'X';             -- write
        amm_address_0       : in    std_logic_vector(25 downto 0)  := (others => 'X'); -- address
        amm_readdata_0      : out   std_logic_vector(511 downto 0);                    -- readdata
        amm_writedata_0     : in    std_logic_vector(511 downto 0) := (others => 'X'); -- writedata
        amm_burstcount_0    : in    std_logic_vector(6 downto 0)   := (others => 'X'); -- burstcount
        amm_byteenable_0    : in    std_logic_vector(63 downto 0)  := (others => 'X'); -- byteenable
        amm_readdatavalid_0 : out   std_logic;                                         -- readdatavalid
        emif_usr_clk        : out   std_logic;                                         -- clk
        emif_usr_reset_n    : out   std_logic;                                         -- reset_n
        global_reset_n      : in    std_logic                      := 'X';             -- reset_n
        mem_ck              : out   std_logic_vector(0 downto 0);                      -- mem_ck
        mem_ck_n            : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
        mem_a               : out   std_logic_vector(16 downto 0);                     -- mem_a
        mem_act_n           : out   std_logic_vector(0 downto 0);                      -- mem_act_n
        mem_ba              : out   std_logic_vector(1 downto 0);                      -- mem_ba
        mem_bg              : out   std_logic_vector(1 downto 0);                      -- mem_bg
        mem_cke             : out   std_logic_vector(0 downto 0);                      -- mem_cke
        mem_cs_n            : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
        mem_odt             : out   std_logic_vector(0 downto 0);                      -- mem_odt
        mem_reset_n         : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
        mem_par             : out   std_logic_vector(0 downto 0);                      -- mem_par
        mem_alert_n         : in    std_logic_vector(0 downto 0)   := (others => 'X'); -- mem_alert_n
        mem_dqs             : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
        mem_dqs_n           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
        mem_dq              : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
        mem_dbi_n           : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dbi_n
        oct_rzqin           : in    std_logic                      := 'X';             -- oct_rzqin
        pll_ref_clk         : in    std_logic                      := 'X';             -- clk
        local_cal_success   : out   std_logic;                                         -- local_cal_success
        local_cal_fail      : out   std_logic                                          -- local_cal_fail
    );
end component ddr4_if;

end package;

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity tb_data_path_farm is
end entity;


architecture TB of tb_data_path_farm is

    signal reset_n  : std_logic;
    signal reset    : std_logic;

    -- Input from merging (first board) or links (subsequent boards)
    signal clk  :   std_logic := '0';
    signal data_en  :   std_logic;
    signal data_in  :   std_logic_vector(511 downto 0);
    signal ts_in    :   std_logic_vector(31 downto 0);

    -- Input from PCIe demanding events
    signal pcieclk      :   std_logic := '0';
    signal ts_req_A     :   std_logic_vector(31 downto 0);
    signal req_en_A     :   std_logic;
    signal ts_req_B     :   std_logic_vector(31 downto 0);
    signal req_en_B     :   std_logic;
    signal tsblock_done :   std_logic_vector(15 downto 0);

    -- Output to DMA
    signal dma_data_out     :   std_logic_vector(255 downto 0);
    signal dma_data_en      :   std_logic;
    signal dma_eoe          :   std_logic;

    -- Interface to memory bank A
    signal A_mem_clk        : std_logic := '0';
    signal A_mem_ready      : std_logic;
    signal A_mem_calibrated : std_logic;
    signal A_mem_addr       : std_logic_vector(25 downto 0);
    signal A_mem_data       : std_logic_vector(511 downto 0);
    signal A_mem_write      : std_logic;
    signal A_mem_read       : std_logic;
    signal A_mem_q          : std_logic_vector(511 downto 0);
    signal A_mem_q_valid    : std_logic;

    -- Interface to memory bank B
    signal B_mem_clk        : std_logic := '0';
    signal B_mem_ready      : std_logic;
    signal B_mem_calibrated : std_logic;
    signal B_mem_addr       : std_logic_vector(25 downto 0);
    signal B_mem_data       : std_logic_vector(511 downto 0);
    signal B_mem_write      : std_logic;
    signal B_mem_read       : std_logic;
    signal B_mem_q          : std_logic_vector(511 downto 0);
    signal B_mem_q_valid    : std_logic;

    -- links and datageneration
    constant NLINKS                 : positive := 8;
    constant NLINKS_TOTL            : positive := 16;
    constant LINK_FIFO_ADDR_WIDTH   : integer := 10;
    constant g_NLINKS_FARM_TOTL     : positive := 3;

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

    signal writeregs : work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));

    signal resets_n : std_logic_vector(31 downto 0) := (others => '0');

    -- clk period
    constant dataclk_period : time := 4 ns;
    constant pcieclk_period : time := 4 ns;
    constant A_mem_clk_period : time := 3.76 ns;
    constant B_mem_clk_period : time := 3.76 ns;

    constant CLK_MHZ : real := 10000.0; -- MHz

    signal toggle : std_logic_vector(1 downto 0);
    signal startinput : std_logic;
    signal ts_in_next   :   std_logic_vector(31 downto 0);

    signal A_mem_read_del1: std_logic;
    signal A_mem_read_del2: std_logic;
    signal A_mem_read_del3: std_logic;
    signal A_mem_read_del4: std_logic;

    signal A_mem_addr_del1  : std_logic_vector(25 downto 0);
    signal A_mem_addr_del2  : std_logic_vector(25 downto 0);
    signal A_mem_addr_del3  : std_logic_vector(25 downto 0);
    signal A_mem_addr_del4  : std_logic_vector(25 downto 0);

    signal B_mem_read_del1: std_logic;
    signal B_mem_read_del2: std_logic;
    signal B_mem_read_del3: std_logic;
    signal B_mem_read_del4: std_logic;

    signal B_mem_addr_del1  : std_logic_vector(25 downto 0);
    signal B_mem_addr_del2  : std_logic_vector(25 downto 0);
    signal B_mem_addr_del3  : std_logic_vector(25 downto 0);
    signal B_mem_addr_del4  : std_logic_vector(25 downto 0);

    signal midas_data_511 : work.util.slv32_array_t(15 downto 0);
    
    signal test : std_logic := '0';
    signal dma_data_array : work.util.slv32_array_t(7 downto 0);
    signal dma_data : std_logic_vector(255 downto 0);


begin

-- synthesis read_comments_as_HDL on
    --test <= '1';
-- synthesis read_comments_as_HDL off

    clk         <= not clk after (0.5 us / CLK_MHZ);
    A_mem_clk   <= not A_mem_clk after (0.1 us / CLK_MHZ);
    B_mem_clk   <= not B_mem_clk after (0.1 us / CLK_MHZ);

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

    resets_n(RESET_BIT_DATAGEN)                                 <= '0', '1' after (1.0 us / CLK_MHZ);
    resets_n(RESET_BIT_FARM_STREAM_MERGER)                      <= '0', '1' after (1.0 us / CLK_MHZ);
    resets_n(RESET_BIT_FARM_TIME_MERGER)                        <= '0', '1' after (1.0 us / CLK_MHZ);
    writeregs(DATAGENERATOR_DIVIDER_REGISTER_W)                 <= x"00000002";
    writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK)  <= '1';
    writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM)    <= '0';
    writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER)    <= '1';
    writeregs(GET_N_DMA_WORDS_REGISTER_W)                       <= (others => '1');
    writeregs(FARM_LINK_MASK_REGISTER_W)                        <= x"00000003";--x"00000048";
    writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE)                   <= '1';
    writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR)       <= '1';

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
        tsblock_done    <= (others => '0');
    end process;

    
    --! Farm Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    farm_block : entity work.farm_block
    generic map (
        g_DDR4         => true,
        g_NLINKS_TOTL  => 8--,
    )
    port map (

        --! links to/from FEBs
        i_rx            => (others => work.mu3e.LINK_IDLE),
        o_tx            => open,

        --! PCIe registers / memory
        i_writeregs     => writeregs,
        i_regwritten    => (others => '0'),
        o_readregs      => open,

        i_resets_n      => resets_n,

        i_dmamemhalffull=> '0',
        o_dma_wren      => open,
        o_endofevent    => open,
        o_dma_data      => dma_data,

        --! 250 MHz clock pice / reset_n
        i_reset_n       => reset_n,
        i_clk           => clk--,
    );

    dma_data_array(0) <= dma_data(0*32 + 31 downto 0*32);
    dma_data_array(1) <= dma_data(1*32 + 31 downto 1*32);
    dma_data_array(2) <= dma_data(2*32 + 31 downto 2*32);
    dma_data_array(3) <= dma_data(3*32 + 31 downto 3*32);
    dma_data_array(4) <= dma_data(4*32 + 31 downto 4*32);
    dma_data_array(5) <= dma_data(5*32 + 31 downto 5*32);
    dma_data_array(6) <= dma_data(6*32 + 31 downto 6*32);
    dma_data_array(7) <= dma_data(7*32 + 31 downto 7*32);

end architecture;
