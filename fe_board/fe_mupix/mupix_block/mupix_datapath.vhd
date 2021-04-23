-- Mupix 10 data path (online repo)
-- M.Mueller, Oktober 2020

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;


entity mupix_datapath is
port (
    i_reset_n           : in  std_logic;
    i_reset_n_regs      : in  std_logic;
    i_reset_n_lvds      : in  std_logic;

    i_clk156            : in  std_logic;
    i_clk125            : in  std_logic;

    i_lvds_rx_inclock_A : in  std_logic;
    i_lvds_rx_inclock_B : in  std_logic;
    lvds_data_in        : in  std_logic_vector(35 downto 0);

    i_reg_add           : in  std_logic_vector(7 downto 0);
    i_reg_re            : in  std_logic;
    o_reg_rdata         : out std_logic_vector(31 downto 0);
    i_reg_we            : in  std_logic;
    i_reg_wdata         : in  std_logic_vector(31 downto 0);

    o_fifo_wdata        : out std_logic_vector(35 downto 0);
    o_fifo_write        : out std_logic;

    i_sync_reset_cnt    : in  std_logic;
    i_fpga_id           : in  std_logic_vector(7 downto 0);
    i_run_state_125     : in  run_state_t;
    i_run_state_156     : in  run_state_t;
    o_hotfix_reroute    : out work.util.slv32_array_t(35 downto 0); -- TODO: fix problem and remove
    i_hotfix_backroute  : in  std_logic--;
);
end mupix_datapath;

architecture rtl of mupix_datapath is

    signal reset_156_n              : std_logic;
    signal reset_125_n              : std_logic;
    signal sorter_reset_n           : std_logic;

    -- signals after mux
    signal rx_data                  : work.util.slv8_array_t(35 downto 0);
    signal rx_k                     : std_logic_vector(35 downto 0);
    signal lvds_status              : work.util.slv32_array_t(35 downto 0);
    signal data_valid               : std_logic_vector(35 downto 0);

    -- hits + flag to indicate a word as a hit, after unpacker
    signal hits_ena                 : std_logic_vector(35 downto 0);
    signal ts                       : ts_array_t(35 downto 0);
    signal row                      : row_array_t(35 downto 0);
    signal col                      : col_array_t(35 downto 0);
    signal tot                      : tot_array_t(35 downto 0);
    signal chip_ID                  : ch_ID_array_t(35 downto 0);

    signal hits_ena_unpacker        : std_logic_vector(35 downto 0);
    signal ts_unpacker              : ts_array_t(35 downto 0);
    signal row_unpacker             : row_array_t(35 downto 0);
    signal col_unpacker             : col_array_t(35 downto 0);
    signal tot_unpacker             : tot_array_t(35 downto 0);
    signal chip_ID_unpacker         : ch_ID_array_t(35 downto 0);

    signal hits_ena_gen             : std_logic_vector(35 downto 0);
    signal ts_gen                   : ts_array_t(35 downto 0);
    signal row_gen                  : row_array_t(35 downto 0);
    signal col_gen                  : col_array_t(35 downto 0);
    signal tot_gen                  : tot_array_t(35 downto 0);
    signal chip_ID_gen              : ch_ID_array_t(35 downto 0);

    -- hits afer 3-1 multiplexing
    signal hits_ena_hs              : std_logic_vector(35 downto 0);
    signal ts_hs                    : ts_array_t(35 downto 0);
    signal row_hs                   : row_array_t(35 downto 0);
    signal col_hs                   : col_array_t(35 downto 0);
    signal tot_hs                   : tot_array_t(35 downto 0);
    signal chip_ID_hs               : ch_ID_array_t(35 downto 0);
    signal hits_sorter_in           : hit_array;
    signal hits_sorter_in_ena       : std_logic_vector(11 downto 0);

    signal running                  : std_logic;

    -- error signal output from unpacker
    signal unpack_errorcounter      : work.util.slv32_array_t(35 downto 0);

    --signal regwritten_reg         : std_logic_vector(NREGISTERS-1 downto 0); 

    signal counter125               : std_logic_vector(63 downto 0);

    signal gen_seed                 : std_logic_vector(64 downto 0);
    signal mp_datagen_control_reg   : std_logic_vector(31 downto 0);

    signal rx_state                 : std_logic_vector(36*4-1 downto 0);

    signal multichip_ro_overflow    : std_logic_vector(31 downto 0);

    signal link_enable              : std_logic_vector(35 downto 0);
    signal link_enable_125          : std_logic_vector(35 downto 0);
    signal lvds_link_mask           : std_logic_vector(35 downto 0);

    -- count hits ena
    signal hits_ena_count           : std_logic_vector(31 downto 0);
    signal time_counter             : unsigned(31 downto 0);
    signal rate_counter             : unsigned(31 downto 0);

    --hit_ts conversion settings
    signal mp_readout_mode          : std_logic_vector(31 downto 0);

    signal fifo_wdata               : std_logic_vector(35 downto 0);
    signal fifo_write               : std_logic;
    signal sync_fifo_empty          : std_logic;
    signal sync_fifo_wdata_out      : std_logic_vector(35 downto 0);
    signal sync_fifo_write_out      : std_logic;

    signal fifo_wdata_hs            : std_logic_vector(35 downto 0);
    signal fifo_write_hs            : std_logic;

    signal fifo_wdata_gen           : std_logic_vector(35 downto 0);
    signal fifo_write_gen           : std_logic;

    -- Sorter config and diagnositc
    signal sorter_counters          : sorter_reg_array;
    signal sorter_delay             : ts_t;

begin

    reset_156_n <= '0' when (i_run_state_156=RUN_STATE_SYNC) else '1';
    reset_125_n <= '0' when (i_run_state_125=RUN_STATE_SYNC) else '1';
------------------------------------------------------------------------------------
---------------------- registers ---------------------------------------------------
    e_mupix_reg_mapping : work.mupix_reg_mapping
    port map (
        i_clk156                    => i_clk156,
        i_reset_n                   => i_reset_n_regs,

        i_reg_add                   => i_reg_add,
        i_reg_re                    => i_reg_re,
        o_reg_rdata                 => o_reg_rdata,
        i_reg_we                    => i_reg_we,
        i_reg_wdata                 => i_reg_wdata,

        -- inputs  156--------------------------------------------
        i_lvds_data_valid           => (others => '0'),
        --i_lvds_status               => lvds_status,

        -- inputs  125 (how to sync)------------------------------
        i_sorter_counters           => sorter_counters,

        -- outputs 156--------------------------------------------
        o_mp_datagen_control        => mp_datagen_control_reg,
        o_mp_lvds_link_mask         => lvds_link_mask,
        o_mp_readout_mode           => mp_readout_mode,

        -- outputs 125-------------------------------------------------
        o_sorter_delay              => sorter_delay--;
    );
    o_hotfix_reroute<= lvds_status; --TODO: fix this!!

------------------------------------------------------------------------------------
---------------------- LVDS Receiver part ------------------------------------------
    lvds_block : work.receiver_block_mupix
    port map(
        i_reset_n           => i_reset_n_lvds,
        i_nios_clk          => i_clk156,
        checker_rst_n       => (others => '1'),--TODO: What is this ? M.Mueller
        rx_in               => lvds_data_in,
        rx_inclock_A        => i_lvds_rx_inclock_A,
        rx_inclock_B        => i_lvds_rx_inclock_B,

        o_rx_status         => lvds_status,
        o_rx_ready          => data_valid,
        i_rx_invert         => i_hotfix_backroute,
        rx_data             => rx_data,
        rx_k                => rx_k--,
    );

    -- use a link mask to disable channels from being used in the data processing
    link_enable <= data_valid and not lvds_link_mask;

--------------------------------------------------------------------------------------
--------------------- Unpack the data ------------------------------------------------
    genunpack:
    FOR i in 0 to 35 GENERATE
    -- we currently only use link 0 of each chip (up to 8 possible)
 
    unpacker_single : work.data_unpacker
    generic map(
        COARSECOUNTERSIZE   => COARSECOUNTERSIZE,
        LVDS_ID             => i
    )
    port map(
        reset_n             => reset_125_n,
        clk                 => i_clk125,
        datain              => rx_data(i), 
        kin                 => rx_k(i), 
        readyin             => link_enable_125(i),
        i_mp_readout_mode   => mp_readout_mode,
        o_ts                => ts_unpacker(i),
        o_chip_ID           => chip_ID_unpacker(i),
        o_row               => row_unpacker(i),
        o_col               => col_unpacker(i),
        o_tot               => tot_unpacker(i),
        o_hit_ena           => hits_ena_unpacker(i),
        errorcounter        => unpack_errorcounter(i) -- could be useful!
    );

    END GENERATE genunpack;

    -- count hits_ena
    process(i_clk125)
    begin
        if rising_edge(i_clk125) then
            if (time_counter > x"7735940") then
                hits_ena_count  <= std_logic_vector(rate_counter)(31 downto 0);
            time_counter        <= (others => '0');
            rate_counter        <= (others => '0');
         else
            -- overflow can not happen here
            rate_counter        <= rate_counter + to_unsigned(work.util.count_bits(hits_ena), 32);
            time_counter        <= time_counter + 1;
         end if;
       end if;
    end process;

    process(i_clk125, reset_125_n)
    begin
        if(reset_125_n = '0')then
            counter125 <= (others => '0');
        elsif(rising_edge(i_clk125))then
            if(i_sync_reset_cnt = '1')then
                counter125 <= (others => '0');
            else
                counter125 <= counter125 + 1;
            end if;
            
            if(mp_datagen_control_reg(MP_DATA_GEN_SORT_IN_BIT) = '1') then
                ts      <= ts_gen;
                chip_ID <= Chip_ID_gen;
                row     <= row_gen;
                col     <= col_gen;
                tot     <= tot_gen;
                hits_ena<= hits_ena_gen;
            else
                ts      <= ts_unpacker;
                chip_ID <= Chip_ID_unpacker;
                row     <= row_unpacker;
                col     <= col_unpacker;
                tot     <= tot_unpacker;
                hits_ena<= hits_ena_unpacker;
            end if;
        end if;
    end process;

    gen_hm:
    FOR i IN 11 downto 0 GENERATE
        -- 3->1 multiplexer
        multiplexer: work.hit_multiplexer
        port map(
            reset_n     => sorter_reset_n,
            clk         => i_clk125,

            i_ts(0)             => ts(i*3),
            i_ts(1)             => ts(i*3+1),
            i_ts(2)             => ts(i*3+2),
            i_chip_ID(0)        => chip_ID(i*3),
            i_chip_ID(1)        => chip_ID(i*3+1),
            i_chip_ID(2)        => chip_ID(i*3+2),
            i_row(0)            => row(i*3),
            i_row(1)            => row(i*3+1),
            i_row(2)            => row(i*3+2),
            i_col(0)            => col(i*3),
            i_col(1)            => col(i*3+1),
            i_col(2)            => col(i*3+2),
            i_tot(0)            => tot(i*3),
            i_tot(1)            => tot(i*3+1),
            i_tot(2)            => tot(i*3+2),
            i_hit_ena           => hits_ena(i*3+2) & hits_ena(i*3+1) & hits_ena(i*3),

            o_ts(0)             => ts_hs(i),
            o_chip_ID(0)        => chip_ID_hs(i),
            o_row(0)            => row_hs(i),
            o_col(0)            => col_hs(i),
            o_tot(0)            => tot_hs(i),
            o_hit_ena           => hits_sorter_in_ena(i)--,
        );
        hits_sorter_in(i)       <= row_hs(i) & col_hs(i) & tot_hs(i)(4 downto 0) & ts_hs(i);
    END GENERATE;

    running         <= '1' when i_run_state_125 = RUN_STATE_RUNNING else '0';
    sorter_reset_n  <= '0' when i_run_state_125 = RUN_STATE_IDLE else '1';

    sorter: work.hitsorter_wide
    port map(
        reset_n         => sorter_reset_n,
        writeclk        => i_clk125,
        running         => running,
        currentts       => counter125(TIMESTAMPSIZE-1 downto 0),
        hit_in          => hits_sorter_in,--(others => (others => '0')),
        hit_ena_in      => hits_sorter_in_ena,--(others => '0'),
        readclk         => i_clk125,
        data_out        => fifo_wdata_hs(31 downto 0),
        out_ena         => fifo_write_hs,
        out_type        => fifo_wdata_hs(35 downto 32),
        diagnostic_out  => sorter_counters,
        delay           => sorter_delay--,
    );

    
    output_select : process(i_clk125) -- hitsorter, generator, unsorted ...
    begin
        if(rising_edge(i_clk125))then
            if(mp_datagen_control_reg(MP_DATA_GEN_ENGAGE_BIT)='1') then
                fifo_wdata  <= fifo_wdata_gen;
                fifo_write  <= fifo_write_gen;
            else
                fifo_wdata  <= fifo_wdata_hs;
                fifo_write  <= fifo_write_hs;
            end if;
        end if;
    end process output_select;

    datagen: work.mp_sorter_datagen
    port map(
        i_reset_n           => sorter_reset_n,
        i_clk               => i_clk125,
        i_running           => running,
        i_global_ts         => counter125,
        i_control_reg       => mp_datagen_control_reg,
        i_seed              => gen_seed,
        o_hit_counter       => open,
        o_fifo_wdata        => fifo_wdata_gen,
        o_fifo_write        => fifo_write_gen,

        o_ts                => ts_gen,
        o_chip_ID           => chip_ID_gen,
        o_row               => row_gen,
        o_col               => col_gen,
        o_tot               => tot_gen,
        o_hit_ena           => hits_ena_gen,

        i_evil_register     => (others => '0'),
        o_mischief_managed  => open--,
    );
    gen_seed <= i_fpga_id & not i_fpga_id & i_fpga_id & not i_fpga_id & not i_fpga_id & i_fpga_id & i_fpga_id & not i_fpga_id & '0';


    -- sync some things ..
    sync_fifo_cnt : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4,
        DATA_WIDTH  => 1+36+32,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            => fifo_write & fifo_wdata & hits_ena_count,
        rdclk           => i_clk156,
        rdreq           => '1',
        rdempty         => sync_fifo_empty,
        wrclk           => i_clk125,
        wrreq           => '1',
        q(31 downto 0)  => open,--o_hits_ena_count,
        q(67 downto 32) => sync_fifo_wdata_out,
        q(68)           => sync_fifo_write_out--,
    );
    
    process(i_clk156)
    begin
    if(rising_edge(i_clk156)) then
        if(sync_fifo_empty='0') then 
            o_fifo_wdata <= sync_fifo_wdata_out;
            o_fifo_write <= sync_fifo_write_out;
        else
            o_fifo_write <= '0';
        end if;
    end if;
    end process;

    sync_fifo_2 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4,
        DATA_WIDTH  => 36,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr    => '0',
        data    => link_enable,
        rdclk   => i_clk125,
        rdreq   => '1',
        wrclk   => i_clk156,
        wrreq   => '1',
        q(35 downto 0) => link_enable_125--,
    );

end rtl;
