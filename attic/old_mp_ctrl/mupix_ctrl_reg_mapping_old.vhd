-- Last Change: M.Mueller, November 2020 (muellem@uni-mainz.de)
-- there are TWO instances of this entity: one in mp_block, one in mp_datapath
-- TODO: check if things are compiled away correctly in 2nd instance_name .. if not --> new file mupix_reg_mapping_datapath.vhd

-- At some point we might want to generate this file automatically from mupix_registers.vhd

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;

entity mupix_ctrl_reg_mapping_old is
port (
    i_clk156                    : in  std_logic;
    i_reset_n                   : in  std_logic;

    i_reg_add                   : in  std_logic_vector(15 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    -- inputs  156--------------------------------------------
    i_mp_spi_busy               : in std_logic;
    o_use_old                   : out std_logic;

    -- outputs 156--------------------------------------------

    o_mp_ctrl_data              : out std_logic_vector(32*5 + 31 downto 0);
    o_mp_fifo_write             : out std_logic_vector( 5 downto 0) := (others => '0');
    o_mp_ctrl_data_all          : out std_logic_vector(31 downto 0);
    o_mp_ctrl_data_all_we       : out std_logic;
    o_mp_fifo_clear             : out std_logic_vector( 5 downto 0) := (others => '0');
    o_mp_ctrl_enable            : out std_logic_vector( 5 downto 0);
    o_mp_ctrl_chip_config_mask  : out std_logic_vector(11 downto 0);
    o_mp_ctrl_invert_29         : out std_logic;
    o_mp_ctrl_invert_csn        : out std_logic;
    o_mp_ctrl_slow_down         : out std_logic_vector(31 downto 0)--;
);
end entity;

architecture rtl of mupix_ctrl_reg_mapping_old is
    signal mp_ctrl_slow_down        : std_logic_vector(31 downto 0);
    signal mp_ctrl_chip_config_mask : std_logic_vector(31 downto 0);
    signal mp_ctrl_invert_29        : std_logic;
    signal mp_ctrl_invert_csn       : std_logic;
    signal mp_fifo_clear            : std_logic_vector( 5 downto 0);
    signal mp_ctrl_enable           : std_logic_vector( 5 downto 0);
    signal mp_ctrl_data_all         : std_logic_vector(31 downto 0);
    signal mp_ctrl_data_all_we      : std_logic;
    signal mp_spi_busy              : std_logic;

    begin

    process (i_clk156, i_reset_n)
        variable regaddr : integer;
    begin
        if (i_reset_n = '0') then
            o_mp_ctrl_enable          <= (others => '0');
            mp_ctrl_invert_csn        <= '0';
            o_mp_ctrl_data_all_we     <= '0';
            o_mp_fifo_write           <= (others => '0');
            o_mp_fifo_clear           <= (others => '0');
            o_mp_ctrl_data            <= (others => '0');
            o_mp_ctrl_data_all        <= (others => '0');
            o_mp_ctrl_chip_config_mask<= (others => '0');
            o_mp_ctrl_invert_29       <= '0';
            o_mp_ctrl_invert_csn      <= '0';
            o_mp_ctrl_slow_down       <= (others => '0');
            
        elsif(rising_edge(i_clk156)) then

            --regs for long paths
            o_mp_ctrl_slow_down         <= mp_ctrl_slow_down;
            o_mp_ctrl_chip_config_mask  <= mp_ctrl_chip_config_mask(11 downto 0);
            o_mp_ctrl_invert_29         <= mp_ctrl_invert_29;
            o_mp_ctrl_invert_csn        <= mp_ctrl_invert_csn;
            regaddr                     := to_integer(unsigned(i_reg_add));
            o_reg_rdata                 <= x"CCCCCCCC";
            o_mp_fifo_write             <= (others => '0');
            --o_mp_ctrl_data_all_we       <= '0';
            o_mp_fifo_clear             <= mp_fifo_clear;
            o_mp_ctrl_enable            <= mp_ctrl_enable;
            o_mp_ctrl_data_all          <= mp_ctrl_data_all;
            o_mp_ctrl_data_all_we       <= mp_ctrl_data_all_we;
            mp_spi_busy                 <= i_mp_spi_busy;

            -----------------------------------------------------------------
            ---- mupix ctrl -------------------------------------------------
            -----------------------------------------------------------------

            if ( regaddr = MP_CTRL_ENABLE_REGISTER_W and i_reg_we = '1' ) then
                mp_fifo_clear  <= i_reg_wdata(CLEAR_TDAC_FIFO_BIT downto CLEAR_BIAS_FIFO_BIT);
                mp_ctrl_enable <= i_reg_wdata(WR_TDAC_BIT downto WR_BIAS_BIT);
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

            if ( regaddr = MP_CTRL_ALL_REGISTER_W and i_reg_we = '1' ) then
                mp_ctrl_data_all      <= i_reg_wdata;
                mp_ctrl_data_all_we   <= '1';
            else
                mp_ctrl_data_all_we   <= '0';
            end if;

            if ( regaddr = MP_CTRL_SPI_BUSY_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata(0) <= mp_spi_busy;
                o_reg_rdata(31 downto 1) <= (others => '0');
            end if;

            if ( regaddr = MP_CTRL_USE_OLD_REGISTER_W and i_reg_we = '1' ) then
                o_use_old <= i_reg_wdata(0);
            end if;

        end if;
    end process;

end architecture;
