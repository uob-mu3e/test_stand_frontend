-- Mupix 10 data path (online repo)
-- M.Mueller, Oktober 2020

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.daq_constants.all;
use work.mupix_constants.all;
use work.mupix_types.all;

entity mupix_datapath is
generic(
    NCHIPS          : integer := 8;
    NLVDS           : integer := 32;
    NSORTERINPUTS   : integer := 1; --up to 4 LVDS links merge to one sorter
    NINPUTS_BANK_A  : integer := 16;
    NINPUTS_BANK_B  : integer := 16--;
);
port (
    i_reset_n           : in  std_logic;
    i_reset_n_lvds      : in  std_logic;

    i_clk156            : in  std_logic;
    i_clk125            : in  std_logic;

    i_lvds_rx_inclock_A : in  std_logic;
    i_lvds_rx_inclock_B : in  std_logic;
    lvds_data_in        : in  std_logic_vector(NLVDS-1 downto 0);

    write_sc_regs       : in  reg32array_t(NREGISTERS_MUPIX_WR-1 downto 0);
    read_sc_regs        : out reg32array_t(NREGISTERS_MUPIX_RD-1 downto 0);

    o_fifo_wdata        : out std_logic_vector(35 downto 0);
    o_fifo_write        : out std_logic;
    o_lvds_data_valid   : out std_logic_vector(NLVDS-1 downto 0);
    o_hits_ena_count    : out std_logic_vector(31 downto 0);

    i_sync_reset_cnt    : in  std_logic;
    
    i_run_state_125     : in  run_state_t;
    i_run_state_156     : in  run_state_t--;
);
end mupix_datapath;

architecture rtl of mupix_datapath is

    signal reset_156_n              : std_logic;
    signal reset_125_n              : std_logic;
    signal sorter_reset_n           : std_logic;

    signal lvds_pll_locked          : std_logic_vector(1 downto 0);
    signal lvds_runcounter          : reg32array_t(NLVDS-1 downto 0);
    signal lvds_errcounter          : reg32array_t(NLVDS-1 downto 0);

    -- signals after mux
    signal rx_data                  : bytearray_t(NLVDS-1 downto 0);
    signal rx_k                     : std_logic_vector(NLVDS-1 downto 0);
    --signal data_valid               : std_logic_vector(NLVDS-1 downto 0);
    signal lvds_data_valid          : std_logic_vector(NLVDS-1 downto 0);

    -- hits + flag to indicate a word as a hit, after unpacker
    signal hits                     : std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
    signal hits_ena                 : std_logic_vector(NLVDS-1 downto 0);

    -- hits after gray-decoding
    signal binhits                  : reg32array_t(NLVDS-1 downto 0);
    signal binhits_ena              : std_logic_vector(NLVDS-1 downto 0);

    -- hits afer 3-1 multiplexing
    signal hits_sorter_in           : hit_array;
    signal hits_sorter_in_ena       : std_logic_vector(NCHIPS-1 downto 0);
    signal running                  : std_logic;

    -- flag to indicate link, after unpacker
    signal link_flag                : std_logic_vector(NCHIPS-1 downto 0);

    -- link flag is pipelined once because hits are gray decoded
    signal link_flag_del            : std_logic_vector(NCHIPS-1 downto 0);

    -- counter + flag to indicate word as a counter, after unpacker
    signal coarsecounters           : std_logic_vector(NCHIPS*COARSECOUNTERSIZE-1 downto 0);
    signal coarsecounters_ena       : std_logic_vector(NCHIPS-1 downto 0);

    -- counter is pipelined once because hits are gray decoded
    signal coarsecounters_del       : std_logic_vector(NCHIPS *COARSECOUNTERSIZE-1 downto 0);
    signal coarsecounters_ena_del   : std_logic_vector(NCHIPS-1 downto 0);

    -- error signal output from unpacker
    signal unpack_errorcounter      : reg32array_t(NLVDS-1 downto 0);

    -- writeregisters are registered once to reduce long combinational paths
    signal writeregs_reg            : reg32array_t(NREGISTERS_MUPIX_WR-1 downto 0);
    signal read_regs                : reg32array_t(NREGISTERS_MUPIX_RD-1 downto 0);
    --signal regwritten_reg         : std_logic_vector(NREGISTERS-1 downto 0); 
    signal sync_fifo_empty          : std_logic;

    signal counter125               : std_logic_vector(63 downto 0);

    signal rx_state                 : std_logic_vector(NLVDS*4-1 downto 0);

    signal multichip_ro_overflow    : std_logic_vector(31 downto 0);

    signal link_enable              : std_logic_vector(NLVDS-1 downto 0);
    signal link_enable_125          : std_logic_vector(NLVDS-1 downto 0);

    -- count hits ena
    signal hits_ena_count           : std_logic_vector(31 downto 0);
    signal time_counter             : unsigned(31 downto 0);
    signal rate_counter             : unsigned(31 downto 0);

    --hit_ts conversion settings
    signal invert_TS                : std_logic;
    signal invert_TS2               : std_logic;
    signal gray_TS                  : std_logic;
    signal gray_TS2                 : std_logic;

    signal fifo_wdata               : std_logic_vector(35 downto 0);
    signal fifo_write               : std_logic;

begin

    reset_156_n <= '0' when (i_run_state_156=RUN_STATE_SYNC) else '1';
    reset_125_n <= '0' when (i_run_state_125=RUN_STATE_SYNC) else '1';
------------------------------------------------------------------------------------
---------------------- registers ---------------------------------------------------
    writregs_clocking : process(i_clk156)
    begin
        if(rising_edge(i_clk156))then
            for I in NREGISTERS_MUPIX_WR-1 downto 0 loop
                writeregs_reg(I) <= write_sc_regs(I);
            end loop;
        end if;
    end process writregs_clocking;

    read_regs_clocking : process(i_clk156)
    begin
        if(rising_edge(i_clk156))then
            for I in NREGISTERS_MUPIX_RD-1 downto 0 loop
                read_sc_regs(I) <= read_regs(I);
            end loop;
        end if;
    end process read_regs_clocking;

    --TODO
    --read_regs(RX_STATE_RECEIVER_0_REGISTER_R)(NLVDS-1 downto 0)   <= rx_state(NLVDS-1 downto 0);
    --read_regs(RX_STATE_RECEIVER_1_REGISTER_R)(NLVDS-1 downto 0)   <= rx_state(NLVDS*2-1 downto NLVDS);
    read_regs(LVDS_PLL_LOCKED_REGISTER_R)(1 downto 0)               <= lvds_pll_locked;
    read_regs(MULTICHIP_RO_OVERFLOW_REGISTER_R)                     <= multichip_ro_overflow;

    GEN_LVDS_REGS:
    FOR I in 0 to NLVDS - 1 GENERATE
        read_regs(LVDS_RUNCOUNTER_REGISTER_R + I) <= lvds_runcounter(I);
        read_regs(LVDS_ERRCOUNTER_REGISTER_R + I) <= lvds_errcounter(I);
    END GENERATE GEN_LVDS_REGS;


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

        rx_state            => open, --rx_state, --TODO
        --o_rx_ready          => data_valid,
        o_rx_ready_nios     => lvds_data_valid,
        rx_data             => rx_data,
        rx_k                => rx_k,
        pll_locked          => lvds_pll_locked--, -- write to some register!

        --rx_runcounter     => lvds_runcounter, -- read_sc_regs
        --rx_errorcounter   => lvds_errcounter, -- would be nice to add some error counter
    );

    o_lvds_data_valid <= lvds_data_valid;

    -- use a link mask to disable channels from being used in the data processing
    gen_link_enable : if NLVDS > 32 GENERATE
        link_enable(31 downto 0)        <= lvds_data_valid(     31 downto  0) and (not writeregs_reg(LINK_MASK_REGISTER_W    )(      31 downto 0));
        link_enable(NLVDS-1 downto 32)  <= lvds_data_valid(NLVDS-1 downto 32) and (not writeregs_reg(LINK_MASK_REGISTER_W + 1)(NLVDS-33 downto 0));
    end generate gen_link_enable;

    gen_link_enable2: if NLVDS < 33 GENERATE
        link_enable                     <= lvds_data_valid                    and (not writeregs_reg(LINK_MASK_REGISTER_W    )(NLVDS- 1 downto 0));
    end generate gen_link_enable2;

--------------------------------------------------------------------------------------
--------------------- Unpack the data ------------------------------------------------
    genunpack:
    FOR i in 0 to NLVDS-1 GENERATE
    -- we currently only use link 0 of each chip (up to 8 possible)
 
    unpacker_single : work.data_unpacker
    generic map(
        COARSECOUNTERSIZE   => COARSECOUNTERSIZE
    )
    port map(
        reset_n             => reset_125_n,
        clk                 => i_clk125,
        datain              => rx_data(i), 
        kin                 => rx_k(i), 
        readyin             => link_enable_125(i),
        hit_out             => hits((i+1)*UNPACKER_HITSIZE-1 downto i*UNPACKER_HITSIZE),
        hit_ena             => hits_ena(i),
        coarsecounter       => open,--coarsecounters((i+1)*COARSECOUNTERSIZE-1 downto i*COARSECOUNTERSIZE),
        coarsecounter_ena   => open,--coarsecounters_ena(i),
        link_flag           => open,--link_flag(i),
        errorcounter        => unpack_errorcounter(i) -- could be useful!
    );

    degray_single : work.hit_ts_conversion 
    port map(
        reset_n     => reset_125_n,
        clk         => i_clk125, 
        invert_TS   => invert_TS,
        invert_TS2  => invert_TS2,
        gray_TS     => gray_TS,
        gray_TS2    => gray_TS2,
        hit_in      => hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
        hit_ena_in  => hits_ena(i),
        hit_out     => binhits(i),--binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
        hit_ena_out => binhits_ena(i)
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

    -- delay cc by one cycle to be in line with hit
    -- Seb: new degray - back to one
    
    -- TODO: Should this be in use somewhere ?
    --    process(i_clk125)
    --    begin
    --        if(i_clk125'event and i_clk125 = '1') then
    --            coarsecounters_del      <= coarsecounters;
    --            coarsecounters_ena_del  <= coarsecounters_ena;
    --            link_flag_del           <= link_flag;
    --        end if;
    --    end process;

    process(i_clk125, reset_125_n)
    begin
        if(reset_125_n = '0' or i_run_state_125 = RUN_STATE_SYNC)then
            counter125 <= (others => '0');
        elsif(rising_edge(i_clk125))then
            if(i_sync_reset_cnt = '1')then
                counter125 <= (others => '0');
            else
                counter125 <= counter125 + 1;
            end if;
        end if;
    end process;

    gen_hm:
    FOR i IN NCHIPS-1 downto 0 GENERATE
        -- 3->1 multiplexer
        multiplexer: work.hit_multiplexer
        port map(
            reset_n     => sorter_reset_n,
            clk         => i_clk125,
            hit_in1     => binhits(i*3),
            hit_ena1    => binhits_ena(i*3),
            hit_in2     => binhits(i*3+1),
            hit_ena2    => binhits_ena(i*3+1),
            hit_in3     => binhits(i*3+2),
            hit_ena3    => binhits_ena(i*3+2),
            hit_out     => hits_sorter_in(i),
            hit_ena     => hits_sorter_in_ena(i)--,
        );
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
        data_out        => fifo_wdata(31 downto 0),
        out_ena         => fifo_write,
        out_type        => fifo_wdata(35 downto 32),
        diagnostic_sel  => (others => '0'),
        diagnostic_out  => open--,
    );

    -- sync some things ..
    sync_fifo_cnt : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
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
        wrclk           => i_clk125,
        wrreq           => '1',
        q(31 downto 0)  => o_hits_ena_count,
        q(67 downto 32) => o_fifo_wdata,
        q(68)           => o_fifo_write--,
    );

    sync_fifo_2 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
        DATA_WIDTH  => 4+NLVDS,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr    => '0',
        data    =>  link_enable &
                    writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_INVERT_BIT) &
                    writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_INVERT_BIT) &
                    writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT) &
                    writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_GRAY_BIT),
                    
        rdclk   => i_clk125,
        rdreq   => '1',
        wrclk   => i_clk156,
        wrreq   => '1',
        q(3)    => invert_TS,
        q(2)    => invert_TS2,
        q(1)    => gray_TS,
        q(0)    => gray_TS2,
        q(NLVDS + 3 downto 4) => link_enable_125--,
    );

end rtl;
