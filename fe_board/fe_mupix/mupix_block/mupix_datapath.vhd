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
    generic(
        IS_TELESCOPE_g : std_logic := '0'--;
    );
port (
    i_reset_n           : in  std_logic;
    i_reset_n_regs      : in  std_logic;

    i_clk156            : in  std_logic;
    i_clk125            : in  std_logic;

    i_lvds_rx_inclock_A : in  std_logic;
    i_lvds_rx_inclock_B : in  std_logic;
    lvds_data_in        : in  std_logic_vector(35 downto 0);

    i_reg_add           : in  std_logic_vector(15 downto 0);
    i_reg_re            : in  std_logic;
    o_reg_rdata         : out std_logic_vector(31 downto 0);
    i_reg_we            : in  std_logic;
    i_reg_wdata         : in  std_logic_vector(31 downto 0);

    o_fifo_wdata        : out std_logic_vector(35 downto 0);
    o_fifo_write        : out std_logic;

    o_data_bypass       : out std_logic_vector(31 downto 0) := x"000000BC";
    o_data_bypass_we    : out std_logic := '0';

    i_sync_reset_cnt    : in  std_logic;
    i_fpga_id           : in  std_logic_vector(7 downto 0);
    i_run_state_125     : in  run_state_t;
    i_run_state_156     : in  run_state_t--;
);
end mupix_datapath;

architecture rtl of mupix_datapath is

    signal reset_156_n              : std_logic;
    signal reset_125_n              : std_logic;
    signal sorter_reset_n           : std_logic := '0';

    -- signals after mux
    signal rx_data                  : work.util.slv8_array_t(35 downto 0);
    signal rx_k                     : std_logic_vector(35 downto 0);
    signal lvds_status              : work.util.slv32_array_t(35 downto 0);
    signal lvds_invert              : std_logic;
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
    signal hits_sorter_in_buf       : hit_array;
    signal hits_sorter_in_ena_buf   : std_logic_vector(11 downto 0);
    signal data_bypass              : std_logic_vector(31 downto 0);
    signal data_bypass_we           : std_logic;
    signal data_bypass_select       : std_logic_vector(31 downto 0);

    signal running                  : std_logic := '0';

    -- error signal output from unpacker
    signal unpack_errorcounter      : work.util.slv32_array_t(35 downto 0);

    --signal regwritten_reg         : std_logic_vector(NREGISTERS-1 downto 0); 

    signal counter125               : std_logic_vector(63 downto 0);

    signal gen_seed                 : std_logic_vector(64 downto 0);
    signal mp_datagen_control_reg   : std_logic_vector(31 downto 0);

    signal rx_state                 : std_logic_vector(36*4-1 downto 0);

    signal multichip_ro_overflow    : std_logic_vector(31 downto 0);

    signal link_enable              : std_logic_vector(35 downto 0);
    signal lvds_link_mask           : std_logic_vector(35 downto 0);
    signal lvds_link_mask_reg       : std_logic_vector(35 downto 0);

    signal coarsecounters           : reg24array(35 downto 0);
    signal coarsecounter_enas       : std_logic_vector(35 downto 0);
    signal delta_ts_link_select     : std_logic_vector(5 downto 0);

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

    signal last_sorter_hit          : std_logic_vector(31 downto 0);
    signal sorter_out_is_hit        : std_logic;
    signal sorter_inject            : std_logic_vector(31 downto 0);
    signal sorter_inject_prev       : std_logic;

    signal hit_ena_cnt_select       : std_logic_vector( 7 downto 0);
    signal hit_ena_cnt              : std_logic_vector(31 downto 0);
    signal hit_ena_counters         : reg32array(35 downto 0);
    signal hit_ena_counters_reg     : reg32array(35 downto 0);
    signal hitsorter_in_ena_counters_reg: reg32array(11 downto 0);
    signal hitsorter_in_ena_counters: reg32array(11 downto 0);
    signal hitsorter_in_ena_cnt     : std_logic_vector(31 downto 0);
    signal hitsorter_in_ena_cnt_sel : std_logic_vector( 3 downto 0);
    signal hitsorter_out_ena_cnt    : std_logic_vector(31 downto 0);
    signal hitsorter_out_ena_cnt_reg: std_logic_vector(31 downto 0);
    signal reset_n_lvds             : std_logic;

    -- sc
    signal mp_sorter_reg            : work.util.rw_t;
    signal mp_lvds_rx_reg           : work.util.rw_t;
    signal mp_datapath_reg          : work.util.rw_t;

begin

    process(i_clk156)
    begin
        if(rising_edge(i_clk156)) then
            if(i_run_state_156=RUN_STATE_SYNC) then
                reset_156_n <= '0';
            else 
                reset_156_n <=  '1';
            end if;
        end if;
    end process;

    process(i_clk125)
    begin
        if(rising_edge(i_clk125)) then
            if(i_run_state_125=RUN_STATE_SYNC) then
                reset_125_n <= '0';
            else 
                reset_125_n <=  '1';
            end if;
        end if;
    end process;
    
------------------------------------------------------------------------------------
---------------------- sc ----------------------------------------------------------

    e_lvl2_sc_node: entity work.sc_node
    generic map (
        SLAVE1_ADDR_MATCH_g => "00010000--------",
        SLAVE2_ADDR_MATCH_g => "00010001--------"--,
    )
    port map (
        i_clk          => i_clk156,
        i_reset_n      => i_reset_n_regs,

        i_master_addr  => i_reg_add,
        i_master_re    => i_reg_re,
        o_master_rdata => o_reg_rdata,
        i_master_we    => i_reg_we,
        i_master_wdata => i_reg_wdata,

        o_slave0_addr  => mp_datapath_reg.addr(15 downto 0),
        o_slave0_re    => mp_datapath_reg.re,
        i_slave0_rdata => mp_datapath_reg.rdata,
        o_slave0_we    => mp_datapath_reg.we,
        o_slave0_wdata => mp_datapath_reg.wdata,

        o_slave1_addr  => mp_sorter_reg.addr(15 downto 0),
        o_slave1_re    => mp_sorter_reg.re,
        i_slave1_rdata => mp_sorter_reg.rdata,
        o_slave1_we    => mp_sorter_reg.we,
        o_slave1_wdata => mp_sorter_reg.wdata,

        o_slave2_addr  => mp_lvds_rx_reg.addr(15 downto 0),
        o_slave2_re    => mp_lvds_rx_reg.re,
        i_slave2_rdata => mp_lvds_rx_reg.rdata,
        o_slave2_we    => mp_lvds_rx_reg.we,
        o_slave2_wdata => mp_lvds_rx_reg.wdata--,
    );

    mp_lvds_rx_reg_mapping_inst: entity work.mp_lvds_rx_reg_mapping
      port map (
        i_clk156          => i_clk156,
        i_reset_n         => i_reset_n_regs,

        i_reg_add         => mp_lvds_rx_reg.addr(15 downto 0),
        i_reg_re          => mp_lvds_rx_reg.re,
        o_reg_rdata       => mp_lvds_rx_reg.rdata,
        i_reg_we          => mp_lvds_rx_reg.we,
        i_reg_wdata       => mp_lvds_rx_reg.wdata,

        i_lvds_status     => lvds_status--,
      );

    e_mupix_datapath_reg_mapping : work.mupix_datapath_reg_mapping
    port map (
        i_clk156                    => i_clk156,
        i_clk125                    => i_clk125,
        i_reset_n                   => i_reset_n_regs,

        i_reg_add                   => mp_datapath_reg.addr(15 downto 0),
        i_reg_re                    => mp_datapath_reg.re,
        o_reg_rdata                 => mp_datapath_reg.rdata,
        i_reg_we                    => mp_datapath_reg.we,
        i_reg_wdata                 => mp_datapath_reg.wdata,

        -- inputs  125 (how to sync)------------------------------
        --i_coarsecounter_ena         => coarsecounter_enas(MP_LINK_ORDER(to_integer(unsigned(delta_ts_link_select)))),
        --i_coarsecounter             => coarsecounters(MP_LINK_ORDER(to_integer(unsigned(delta_ts_link_select)))),
        i_ts_global                 => counter125(23 downto 0),
        i_last_sorter_hit           => last_sorter_hit,
        i_mp_hit_ena_cnt            => hit_ena_cnt,
        i_mp_sorter_in_hit_ena_cnt  => hitsorter_in_ena_cnt,
        i_mp_sorter_out_hit_ena_cnt => hitsorter_out_ena_cnt_reg,

        -- outputs 156--------------------------------------------
        o_mp_datagen_control        => mp_datagen_control_reg,
        o_mp_lvds_link_mask         => lvds_link_mask,
        o_mp_lvds_invert            => lvds_invert,
        o_mp_readout_mode           => mp_readout_mode,
        o_mp_data_bypass_select     => data_bypass_select,
        o_mp_delta_ts_link_select   => delta_ts_link_select,

        -- outputs 125-------------------------------------------------
        o_sorter_inject             => sorter_inject,
        o_mp_reset_n_lvds           => reset_n_lvds,
        o_mp_hit_ena_cnt_select     => hit_ena_cnt_select,
        o_mp_hit_ena_cnt_sorter_sel => hitsorter_in_ena_cnt_sel--,
    );

------------------------------------------------------------------------------------
---------------------- LVDS Receiver part ------------------------------------------
    lvds_block : work.receiver_block_mupix
    generic map (
        IS_TELESCOPE_g  => IS_TELESCOPE_g--,
    )
    port map(
        i_reset_n           => reset_n_lvds,
        i_nios_clk          => i_clk156,
        i_clk_global        => i_clk125,
        checker_rst_n       => (others => '1'),--TODO: What is this ? M.Mueller
        rx_in               => lvds_data_in,
        rx_inclock_A        => i_lvds_rx_inclock_A,
        rx_inclock_B        => i_lvds_rx_inclock_B,

        o_rx_status         => lvds_status,
        o_rx_ready          => data_valid,
        i_rx_invert         => lvds_invert,
        o_rx_data           => rx_data,
        o_rx_k              => rx_k--,
    );

    -- use a link mask to disable channels from being used in the data processing
    link_enable <= data_valid and not lvds_link_mask_reg;

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
        datain              => rx_data(MP_LINK_ORDER(i)), 
        kin                 => rx_k(MP_LINK_ORDER(i)), 
        readyin             => link_enable(MP_LINK_ORDER(i)),
        i_mp_readout_mode   => mp_readout_mode,
        o_ts                => ts_unpacker(i),
        o_chip_ID           => chip_ID_unpacker(i),
        o_row               => row_unpacker(i),
        o_col               => col_unpacker(i),
        o_tot               => tot_unpacker(i),
        o_hit_ena           => hits_ena_unpacker(i),
        o_coarsecounter     => open,--coarsecounters(i),
        o_coarsecounter_ena => open,--coarsecounter_enas(i),
        o_hit_ena_counter   => hit_ena_counters(i),
        i_run_state_125     => i_run_state_125,
        errorcounter        => unpack_errorcounter(i) -- could be useful!
    );

    END GENERATE genunpack;

    --------------------------------------------
    -- 2 not so interesting processes (one in 156, one in 125 Mhz)
    -- counting stuff, delaying stuff, etc. nothing really happening here
    --------------------------------------------

    process(i_clk156)
    begin
        if(rising_edge(i_clk156)) then
            hit_ena_counters_reg            <= hit_ena_counters;
            hitsorter_in_ena_counters_reg   <= hitsorter_in_ena_counters;
            hitsorter_out_ena_cnt_reg       <= hitsorter_out_ena_cnt;

            if(to_integer(unsigned(hit_ena_cnt_select))<36) then
                hit_ena_cnt <= hit_ena_counters_reg(to_integer(unsigned(hit_ena_cnt_select)));
            else
                hit_ena_cnt <= (others => '0');
            end if;

            if(to_integer(unsigned(hitsorter_in_ena_cnt_sel))<12) then
                hitsorter_in_ena_cnt <= hitsorter_in_ena_counters_reg(to_integer(unsigned(hitsorter_in_ena_cnt_sel)));
            else
                hitsorter_in_ena_cnt <= (others => '0');
            end if;
        end if;
    end process;



    process(i_clk125, reset_125_n)
    begin
        if(reset_125_n = '0')then
            counter125                  <= (others => '0');
            last_sorter_hit             <= (others => '0');
            hitsorter_out_ena_cnt       <= (others => '0');
            hitsorter_in_ena_counters   <= (others => (others => '0'));
            hits_sorter_in_ena          <= (others => '0');

        elsif(rising_edge(i_clk125))then
            lvds_link_mask_reg  <= lvds_link_mask;

            if(i_sync_reset_cnt = '1')then
                counter125 <= (others => '0');
            else
                counter125 <= counter125 + 1;
            end if;

            if(sorter_out_is_hit='1') then
                last_sorter_hit <= fifo_wdata_hs(31 downto 0);
            end if;

            if(i_run_state_125 = RUN_STATE_RUNNING) then
                if(sorter_out_is_hit='1') then 
                        hitsorter_out_ena_cnt <= hitsorter_out_ena_cnt + '1';
                end if;
            end if;

            sorter_inject_prev <= sorter_inject(MP_SORTER_INJECT_ENABLE_BIT);
            if(sorter_inject_prev = '0' and sorter_inject(MP_SORTER_INJECT_ENABLE_BIT) = '1' and (to_integer(unsigned(sorter_inject(MP_SORTER_INJECT_SELECT_RANGE))) < 12)) then
                hits_sorter_in      <= (others => sorter_inject);
                hits_sorter_in_ena  <= (others => '0');
                hits_sorter_in_ena(to_integer(unsigned(sorter_inject(MP_SORTER_INJECT_SELECT_RANGE)))) <= '1';
            else
                hits_sorter_in      <= hits_sorter_in_buf;
                hits_sorter_in_ena  <= hits_sorter_in_ena_buf;
            end if;

            for i in 0 to 11 loop
                if(i_run_state_125 = RUN_STATE_RUNNING) then
                    if(hits_sorter_in_ena(i)='1') then 
                        hitsorter_in_ena_counters(i) <= hitsorter_in_ena_counters(i) + '1';
                    end if;
                end if;
            end loop;

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
            o_hit_ena           => hits_sorter_in_ena_buf(i)--,
        );
        hits_sorter_in_buf(i)       <= row_hs(i) & col_hs(i) & tot_hs(i)(4 downto 0) & ts_hs(i);
    END GENERATE;

    process(i_clk125)
        begin
        if(rising_edge(i_clk125))then
            if(i_run_state_125 = RUN_STATE_RUNNING) then
                running         <= '1';
            else
                running         <= '0';
            end if;
            if(i_run_state_125 = RUN_STATE_IDLE) then
                sorter_reset_n  <= '0';
            else 
                sorter_reset_n  <= '1';
            end if;
        end if;
    end process;
 
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
        out_is_hit      => sorter_out_is_hit,

        i_clk156        => i_clk156,
        i_reset_n_regs  => i_reset_n_regs,
        i_reg_add       => mp_sorter_reg.addr(15 downto 0),
        i_reg_re        => mp_sorter_reg.re,
        o_reg_rdata     => mp_sorter_reg.rdata,
        i_reg_we        => mp_sorter_reg.we,
        i_reg_wdata     => mp_sorter_reg.wdata--,
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
        DATA_WIDTH  => 1+36,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        REGOUT      => 0,
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            => fifo_write & fifo_wdata,
        rdclk           => i_clk156,
        rdreq           => '1',
        rdempty         => sync_fifo_empty,
        wrclk           => i_clk125,
        wrreq           => '1',
        q(35 downto 0)  => sync_fifo_wdata_out,
        q(36)           => sync_fifo_write_out--,
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

    -- bypass hitsorter and put data of a single MP directly on a seperate optical link
    process(i_clk125)
    begin
    if(rising_edge(i_clk125)) then
        if(to_integer(unsigned(data_bypass_select)) <= NCHIPS) then 
            data_bypass     <= hits_sorter_in(to_integer(unsigned(data_bypass_select)));
            data_bypass_we  <= hits_sorter_in_ena(to_integer(unsigned(data_bypass_select)));
        else
            data_bypass     <= x"000000BC";
            data_bypass_we  <= '0';
        end if;
    end if;
    end process;

    sync_fifo_bypass : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4,
        DATA_WIDTH  => 32,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        REGOUT      => 0,
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            => data_bypass,
        rdclk           => i_clk156,
        rdreq           => '1',
        rdempty         => o_data_bypass_we,
        wrclk           => i_clk125,
        wrreq           => data_bypass_we,
        q               => o_data_bypass--,
    );

end rtl;
