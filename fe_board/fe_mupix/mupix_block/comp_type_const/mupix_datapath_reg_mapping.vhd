-- Last Change: M.Mueller, November 2020 (muellem@uni-mainz.de)
-- there are TWO instances of this entity: one in mp_block, one in mp_datapath
-- TODO: check if things are compiled away correctly in 2nd instance_name .. if not --> new file mupix_reg_mapping_datapath.vhd

-- At some point we might want to generate this file automatically from mupix_registers.vhd

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;


entity mupix_datapath_reg_mapping is
port (
    i_clk156                    : in  std_logic;
    i_clk125                    : in  std_logic := '0';
    i_reset_n                   : in  std_logic;

    i_reg_add                   : in  std_logic_vector(7 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    -- inputs  156--------------------------------------------
    -- ALL INPUTS DEFAULT TO (n*4-1 downto 0 => x"CCC..", others => '1')
    i_lvds_data_valid           : in  std_logic_vector(35 downto 0) := x"CCCCCCCCC"; -- lvds alignment to mupix chips ok
    i_lvds_status               : in  work.util.slv32_array_t    (35 downto 0) := (others => x"CCCCCCCC");

    -- inputs  125 (how to sync)------------------------------
    i_sorter_counters           : in sorter_reg_array   := (others => x"CCCCCCCC");
    i_coarsecounter_ena         : in std_logic := '0';
    i_coarsecounter             : in std_logic_vector(23 downto 0) := (others => '0');
    i_ts_global                 : in std_logic_vector(23 downto 0) := (others => '0');
    i_last_sorter_hit           : in std_logic_vector(31 downto 0) := (others => '0');
    i_mp_hit_ena_cnt            : in std_logic_vector(31 downto 0) := (others => '0');

    -- outputs 156--------------------------------------------
    o_mp_lvds_link_mask         : out std_logic_vector(35 downto 0); -- lvds link mask
    o_mp_lvds_invert            : out std_logic;
    o_mp_datagen_control        : out std_logic_vector(31 downto 0); -- control register for the mupix data gen
    o_mp_readout_mode           : out std_logic_vector(31 downto 0); -- Invert ts, degray, chip ID numbering, tot mode, ..
    o_mp_data_bypass_select     : out std_logic_vector(31 downto 0);
    o_mp_delta_ts_link_select   : out std_logic_vector(5 downto 0);

    -- outputs 125-------------------------------------------------
    o_sorter_delay              : out ts_t;
    o_sorter_inject             : out std_logic_vector(31 downto 0) := (others => '0');
    o_mp_hit_ena_cnt_select     : out std_logic_vector( 7 downto 0) := (others => '0')--;
);
end entity;

architecture rtl of mupix_datapath_reg_mapping is
    signal mp_datagen_control       : std_logic_vector(31 downto 0);
    signal mp_readout_mode          : std_logic_vector(31 downto 0);
    signal mp_lvds_link_mask        : std_logic_vector(35 downto 0);
    signal mp_lvds_link_mask_ordered: std_logic_vector(35 downto 0);
    signal mp_lvds_data_valid       : std_logic_vector(35 downto 0);
    signal mp_lvds_invert           : std_logic;
    signal mp_data_bypass_select    : std_logic_vector(31 downto 0);
    signal mp_sorter_inject         : std_logic_vector(31 downto 0);
    signal mp_hit_ena_cnt_select    : std_logic_vector( 7 downto 0);

    signal reg_delay                : std_logic;
begin

    process(i_clk125)
    begin
        if(rising_edge(i_clk125)) then
            if(reg_delay = '1') then
                o_sorter_inject <= mp_sorter_inject;
            end if;
        end if;
    end process;

    gen_mask_order: for i in 0 to 35 generate
        mp_lvds_link_mask_ordered(i)         <= mp_lvds_link_mask(MP_LINK_ORDER(i));
    end generate;

    process (i_clk156, i_reset_n)
        variable regaddr : integer;
    begin
        if (i_reset_n = '0') then 
            mp_datagen_control        <= (others => '0');
            mp_lvds_link_mask         <= (others => '0');
            mp_sorter_inject          <= (others => '0');
            
        elsif(rising_edge(i_clk156)) then

            --regs for long paths
            o_mp_lvds_link_mask         <= mp_lvds_link_mask_ordered;
            o_mp_lvds_invert            <= mp_lvds_invert;
            o_mp_datagen_control        <= mp_datagen_control;
            o_mp_readout_mode           <= mp_readout_mode;
            mp_lvds_data_valid          <= i_lvds_data_valid;
            o_mp_data_bypass_select     <= mp_data_bypass_select;
            o_mp_hit_ena_cnt_select     <= mp_hit_ena_cnt_select;

            regaddr             := to_integer(unsigned(i_reg_add(7 downto 0)));
            o_reg_rdata         <= x"CCCCCCCC";

            if(i_reg_we = '1') then
                reg_delay <= '0';
            else
                reg_delay <= '1';
            end if;

            -----------------------------------------------------------------
            ---- datapath ---------------------------------------------------
            -----------------------------------------------------------------

            if ( regaddr = MP_READOUT_MODE_REGISTER_W and i_reg_we = '1' ) then
                mp_readout_mode <= i_reg_wdata;
            end if;
            if ( regaddr = MP_READOUT_MODE_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_readout_mode;
            end if;

            if ( regaddr = MP_LVDS_LINK_MASK_REGISTER_W and i_reg_we = '1' ) then
                mp_lvds_link_mask(31 downto 0) <= i_reg_wdata;
            end if;
            if ( regaddr = MP_LVDS_LINK_MASK_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_lvds_link_mask_ordered(31 downto 0);
            end if;
            if ( regaddr = MP_LVDS_LINK_MASK2_REGISTER_W and i_reg_we = '1' ) then
                mp_lvds_link_mask(35 downto 32) <= i_reg_wdata(3 downto 0);
            end if;
            if ( regaddr = MP_LVDS_LINK_MASK2_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata(3 downto 0) <= mp_lvds_link_mask_ordered(35 downto 32);
                o_reg_rdata(31 downto 4)<= (others => '0');
            end if;

            if ( regaddr = MP_LVDS_DATA_VALID_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata <= mp_lvds_data_valid(31 downto 0);
            end if;
            if ( regaddr = MP_LVDS_DATA_VALID2_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata(3 downto 0) <= mp_lvds_data_valid(35 downto 32);
                o_reg_rdata(31 downto 4)<= (others => '0');
            end if;

            if ( regaddr = MP_DATA_GEN_CONTROL_REGISTER_W and i_reg_we = '1' ) then
                mp_datagen_control <= i_reg_wdata;
            end if;
            if ( regaddr = MP_DATA_GEN_CONTROL_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_datagen_control;
            end if;

            for I in 0 to MUPIX_LVDS_STATUS_BLOCK_LENGTH-1 loop 
                if ( regaddr = I + MP_LVDS_STATUS_START_REGISTER_W and i_reg_re = '1' ) then
                    o_reg_rdata <= i_lvds_status(MP_LINK_ORDER(I));
                end if;
            end loop;

            if ( regaddr = MP_LVDS_INVERT_REGISTER_W and i_reg_we = '1' ) then
                mp_lvds_invert <= i_reg_wdata(0);
            end if;
            if ( regaddr = MP_LVDS_INVERT_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= (0 => mp_lvds_invert, others => '0');
            end if;

            if ( regaddr = MP_SORTER_DELAY_W and i_reg_we = '1' ) then
                o_sorter_delay <= i_reg_wdata(TSRANGE);
            end if;

            for I in 0 to NSORTERCOUNTERS-1 loop 
                if ( regaddr = I + MP_SORTER_COUNTER_R and i_reg_re = '1' ) then
                    o_reg_rdata <= i_sorter_counters(I);
                end if;
            end loop;

            if ( regaddr = MP_DATA_BYPASS_SELECT_REGISTER_W and i_reg_we = '1' ) then
                mp_data_bypass_select <= i_reg_wdata;
            end if;
            if ( regaddr = MP_DATA_BYPASS_SELECT_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_data_bypass_select;
            end if;

            if ( regaddr = MP_LAST_SORTER_HIT_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata <= i_last_sorter_hit;
            end if;

            if ( regaddr = MP_SORTER_INJECT_REGISTER_W and i_reg_we = '1' ) then
                mp_sorter_inject <= i_reg_wdata;
            end if;
            if ( regaddr = MP_SORTER_INJECT_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_sorter_inject;
            end if;

            if ( regaddr = MP_HIT_ENA_CNT_SELECT_REGISTER_W and i_reg_we = '1' ) then
                mp_hit_ena_cnt_select <= i_reg_wdata(7 downto 0);
            end if;
            if ( regaddr = MP_HIT_ENA_CNT_SELECT_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata( 7 downto 0) <= mp_hit_ena_cnt_select;
                o_reg_rdata(31 downto 8) <= (others => '0');
            end if;

            if ( regaddr = MP_HIT_ENA_CNT_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata             <= i_mp_hit_ena_cnt;
                mp_hit_ena_cnt_select   <= std_logic_vector(to_unsigned(to_integer(unsigned(mp_hit_ena_cnt_select)) + 1,8));
            end if;
        end if;
    end process;


    -- histogram of delta ts between mp timestamp and fpga timestamp
--    delta_ts_histo : work.histogram_generic
--    generic map(
--        DATA_WIDTH   => ,
--        ADDR_WIDTH   => 
--    )
--    port map(
--        rclk:         in std_logic;
--        wclk:         in std_logic;
--        rst_n:        in std_logic;
--        zeromem:      in std_logic;
--        ena:          in std_logic;
--        can_overflow: in std_logic;
--        data_in:      in std_logic_vector(ADDR_WIDTH-1 downto 0);
--        valid_in:     in std_logic;
--        busy_n:       out std_logic;
--
--        raddr_in:     in std_logic_vector(ADDR_WIDTH-1 downto 0);
--        q_out:        out std_logic_vector(DATA_WIDTH-1 downto 0)
--    );
end architecture;
