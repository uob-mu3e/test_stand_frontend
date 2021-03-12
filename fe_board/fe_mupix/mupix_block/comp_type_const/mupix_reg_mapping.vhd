-- Last Change: M.Mueller, November 2020 (muellem@uni-mainz.de)
-- there are TWO instances of this entity: one in mp_block, one in mp_datapath
-- TODO: check if things are compiled away correctly in 2nd instance_name .. if not --> new file mupix_reg_mapping_datapath.vhd

-- At some point we might want to generate this file automatically from mupix_registers.vhd

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;


entity mupix_reg_mapping is
port (
    i_clk156                    : in  std_logic;
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

    -- outputs 156--------------------------------------------
    o_mp_lvds_link_mask         : out std_logic_vector(35 downto 0); -- lvds link mask
    o_mp_lvds_invert            : out std_logic;
    o_mp_datagen_control        : out std_logic_vector(31 downto 0); -- control register for the mupix data gen
    o_mp_readout_mode           : out std_logic_vector(31 downto 0); -- Invert ts, degray, chip ID numbering, tot mode, ..

    o_mp_ctrl_data              : out std_logic_vector(32*5 + 31 downto 0);
    o_mp_fifo_write             : out std_logic_vector( 5 downto 0);
    o_mp_fifo_clear             : out std_logic_vector( 5 downto 0);
    o_mp_ctrl_enable            : out std_logic_vector( 5 downto 0);
    o_mp_ctrl_chip_config_mask  : out std_logic_vector(11 downto 0);
    o_mp_ctrl_invert_29         : out std_logic;
    o_mp_ctrl_invert_csn        : out std_logic;
    o_mp_ctrl_slow_down         : out std_logic_vector(31 downto 0)--;
);
end entity;

architecture rtl of mupix_reg_mapping is
    signal mp_datagen_control       : std_logic_vector(31 downto 0);
    signal mp_readout_mode          : std_logic_vector(31 downto 0);
    signal mp_lvds_link_mask        : std_logic_vector(35 downto 0);
    signal mp_lvds_data_valid       : std_logic_vector(35 downto 0);
    signal mp_ctrl_slow_down        : std_logic_vector(31 downto 0);
    signal mp_ctrl_chip_config_mask : std_logic_vector(31 downto 0);
    signal mp_ctrl_invert_29        : std_logic;
    signal mp_ctrl_invert_csn       : std_logic;
    signal mp_lvds_invert           : std_logic;
begin

    process (i_clk156, i_reset_n)
        variable regaddr : integer;
    begin
        if (i_reset_n = '0') then 
            mp_datagen_control        <= (others => '0');
            o_mp_ctrl_enable          <= (others => '0');
            mp_lvds_link_mask         <= (others => '0');
            mp_ctrl_invert_csn        <= '0';
            
        elsif(rising_edge(i_clk156)) then

            --regs for long paths
            o_mp_lvds_link_mask         <= mp_lvds_link_mask;
            o_mp_lvds_invert            <= mp_lvds_invert;
            o_mp_datagen_control        <= mp_datagen_control;
            o_mp_readout_mode           <= mp_readout_mode;
            mp_lvds_data_valid          <= i_lvds_data_valid;
            o_mp_ctrl_slow_down         <= mp_ctrl_slow_down;
            o_mp_ctrl_chip_config_mask  <= mp_ctrl_chip_config_mask(11 downto 0);
            o_mp_ctrl_invert_29         <= mp_ctrl_invert_29;
            o_mp_ctrl_invert_csn        <= mp_ctrl_invert_csn;

            regaddr             := to_integer(unsigned(i_reg_add(7 downto 0)));
            o_reg_rdata         <= x"CCCCCCCC";
            o_mp_fifo_write     <= (others => '0');

            -----------------------------------------------------------------
            ---- mupix ctrl -------------------------------------------------
            -----------------------------------------------------------------

            if ( regaddr = MP_CTRL_ENABLE_REGISTER_W and i_reg_we = '1' ) then
                o_mp_fifo_clear  <= i_reg_wdata(CLEAR_TDAC_FIFO_BIT downto CLEAR_BIAS_FIFO_BIT);
                o_mp_ctrl_enable <= i_reg_wdata(WR_TDAC_BIT downto WR_BIAS_BIT);
            end if;

            if ( regaddr = MP_CTRL_CONF_REGISTER_W and i_reg_we = '1' ) then
                o_mp_ctrl_data(WR_CONF_BIT*32 + 31 downto WR_CONF_BIT*32) <= i_reg_wdata;
                o_mp_fifo_write(WR_CONF_BIT) <= '1';
            end if;
            if ( regaddr = MP_CTRL_VDAC_REGISTER_W and i_reg_we = '1' ) then
                o_mp_ctrl_data(WR_VDAC_BIT*32 + 31 downto WR_VDAC_BIT*32) <= i_reg_wdata;
                o_mp_fifo_write(WR_VDAC_BIT) <= '1';
            end if;
            if ( regaddr = MP_CTRL_BIAS_REGISTER_W and i_reg_we = '1' ) then
                o_mp_ctrl_data(WR_BIAS_BIT*32 + 31 downto WR_BIAS_BIT*32) <= i_reg_wdata;
                o_mp_fifo_write(WR_BIAS_BIT) <= '1';
            end if;
            if ( regaddr = MP_CTRL_TDAC_REGISTER_W and i_reg_we = '1' ) then
                o_mp_ctrl_data(WR_TDAC_BIT*32 + 31 downto WR_TDAC_BIT*32) <= i_reg_wdata;
                o_mp_fifo_write(WR_TDAC_BIT) <= '1';
            end if;
            if ( regaddr = MP_CTRL_TEST_REGISTER_W and i_reg_we = '1' ) then
                o_mp_ctrl_data(WR_test_BIT*32 + 31 downto WR_test_BIT*32) <= i_reg_wdata;
                o_mp_fifo_write(WR_test_BIT) <= '1';
            end if;
            if ( regaddr = MP_CTRL_COL_REGISTER_W and i_reg_we = '1' ) then
                o_mp_ctrl_data(WR_COL_BIT*32 + 31 downto WR_COL_BIT*32) <= i_reg_wdata;
                o_mp_fifo_write(WR_COL_BIT)  <= '1';
            end if;

            if ( regaddr = MP_CTRL_SLOW_DOWN_REGISTER_W and i_reg_we = '1' ) then
                mp_ctrl_slow_down <= i_reg_wdata;
            end if;
            if ( regaddr = MP_CTRL_SLOW_DOWN_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_ctrl_slow_down;
            end if;

            if ( regaddr = MP_CTRL_CHIP_MASK_REGISTER_W and i_reg_we = '1' ) then
                mp_ctrl_chip_config_mask <= i_reg_wdata;
            end if;
            if ( regaddr = MP_CTRL_CHIP_MASK_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_ctrl_chip_config_mask;
            end if;

            if ( regaddr = MP_CTRL_INVERT_REGISTER_W and i_reg_we = '1' ) then
                mp_ctrl_invert_29   <= i_reg_wdata(MP_CTRL_INVERT_29_BIT);
                mp_ctrl_invert_csn  <= i_reg_wdata(MP_CTRL_INVERT_CSN_BIT);
            end if;
            if ( regaddr = MP_CTRL_INVERT_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata(MP_CTRL_INVERT_29_BIT)  <= mp_ctrl_invert_29;
                o_reg_rdata(MP_CTRL_INVERT_CSN_BIT) <= mp_ctrl_invert_csn;
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
                o_reg_rdata <= mp_lvds_link_mask(31 downto 0);
            end if;
            if ( regaddr = MP_LVDS_LINK_MASK2_REGISTER_W and i_reg_we = '1' ) then
                mp_lvds_link_mask(35 downto 32) <= i_reg_wdata(3 downto 0);
            end if;
            if ( regaddr = MP_LVDS_LINK_MASK2_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata(3 downto 0) <= mp_lvds_link_mask(35 downto 32);
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

        end if;
    end process;
end architecture;
