-- M. Mueller

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;


entity mupix_ctrl_reg_mapping is
generic( 
    N_CHIPS_PER_SPI_g       : positive := 3;
    N_SPI_g                 : positive := 4 
);
port (
    i_clk156                    : in  std_logic;
    i_reset_n                   : in  std_logic;
    o_reset_n                   : out std_logic;

    i_reg_add                   : in  std_logic_vector(15 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    ----------------------------------------

    i_mp_spi_busy               : in std_logic := '0';

    o_chip_cvb                  : out std_logic_vector(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0);
    o_chip_tdac                 : out integer range 0 to N_CHIPS_PER_SPI_g*N_SPI_g-1;

    o_conf_reg_data             : out reg32;
    o_conf_reg_we               : out std_logic;
    o_vdac_reg_data             : out reg32;
    o_vdac_reg_we               : out std_logic;
    o_bias_reg_data             : out reg32;
    o_bias_reg_we               : out std_logic;

    o_combined_data             : out reg32;
    o_combined_data_we          : out std_logic;

    o_tdac_data                 : out reg32;
    o_tdac_we                   : out std_logic;
    o_run_tdac_test             : out std_logic;

    o_mp_ctrl_slow_down         : out std_logic_vector(31 downto 0);
    o_mp_direct_spi_data        : out reg32array(N_SPI_g-1 downto 0);
    o_mp_direct_spi_data_wr     : out std_logic_vector(N_SPI_g-1 downto 0);
    i_mp_direct_spi_busy        : in  std_logic_vector(N_SPI_g-1 downto 0);
    o_mp_ctrl_direct_spi_enable : out std_logic;
    o_mp_ctrl_spi_enable        : out std_logic--;
);
end entity;

architecture rtl of mupix_ctrl_reg_mapping is
    signal mp_ctrl_slow_down        : std_logic_vector(31 downto 0);
    signal mp_spi_busy              : std_logic;
    signal mp_ctrl_direct_spi_enable: std_logic;
    signal mp_ctrl_spi_enable       : std_logic;
    signal conf_write_chip_select   : std_logic_vector(63 downto 0);

    begin

    process (i_clk156, i_reset_n)
        variable regaddr : integer;
    begin
        if (i_reset_n = '0') then
            o_reset_n                   <= '0';
            o_mp_ctrl_slow_down         <= (others => '0');
            mp_ctrl_slow_down           <= (others => '0');
            o_mp_direct_spi_data_wr     <= (others => '0');
            o_mp_direct_spi_data        <= (others => (others => '0'));
            o_reg_rdata                 <= x"CCCCCCCC";
            mp_ctrl_direct_spi_enable   <= '0';
            mp_ctrl_spi_enable          <= '0';
            o_mp_ctrl_direct_spi_enable <= '0';
            o_mp_ctrl_spi_enable        <= '0';
            o_conf_reg_we               <= '0';
            o_combined_data_we          <= '0';
            o_tdac_we                   <= '0';
            o_vdac_reg_we               <= '0';
            o_bias_reg_we               <= '0';
            o_conf_reg_data             <= (others => '0');
            o_bias_reg_data             <= (others => '0');
            o_vdac_reg_data             <= (others => '0');
            o_tdac_data                 <= (others => '0');
            o_chip_cvb                  <= (others => '0');
            o_combined_data             <= (others => '0');
            conf_write_chip_select      <= (others => '0');
            o_run_tdac_test             <= '0';

        elsif(rising_edge(i_clk156)) then

            o_reset_n                   <= '1';
            o_mp_ctrl_slow_down         <= mp_ctrl_slow_down;
            regaddr                     := to_integer(unsigned(i_reg_add));
            o_reg_rdata                 <= x"CCCCCCCC";
            o_mp_direct_spi_data_wr     <= (others => '0');
            mp_spi_busy                 <= i_mp_spi_busy;
            o_mp_ctrl_direct_spi_enable <= mp_ctrl_direct_spi_enable;
            o_mp_ctrl_spi_enable        <= mp_ctrl_spi_enable;

            o_chip_cvb                  <= conf_write_chip_select(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0); -- o_chip_cvb is Overwritten in case of regaddr match with MP_CTRL_COMBINED_START_REGISTER_W !!!
            o_combined_data_we          <= '0';
            o_tdac_we                   <= '0';
            o_conf_reg_we               <= '0';
            o_vdac_reg_we               <= '0';
            o_bias_reg_we               <= '0';
            o_run_tdac_test             <= '0';


            -----------------------------------------------------------------
            ---- mupix ctrl -------------------------------------------------
            -----------------------------------------------------------------

            loopctrlALL: for I in 0 to N_SPI_g*N_CHIPS_PER_SPI_g-1 loop
                if ( regaddr = MP_CTRL_COMBINED_START_REGISTER_W + I and i_reg_we = '1' ) then
                    o_combined_data     <= i_reg_wdata;
                    o_combined_data_we  <= '1';
                    o_chip_cvb          <= (I => '1', others => '0');
                end if;
            end loop;

            loopTDACs: for I in 0 to N_SPI_g*N_CHIPS_PER_SPI_g-1 loop
                if ( regaddr = MP_CTRL_TDAC_START_REGISTER_W + I and i_reg_we = '1' ) then
                    o_tdac_data         <= i_reg_wdata;
                    o_tdac_we           <= '1';
                    o_chip_tdac         <= I;
                end if;
            end loop;

            if ( regaddr = MP_CTRL_CHIP_SELECT1_REGISTER_W and i_reg_we = '1' ) then
                conf_write_chip_select(31 downto 0) <= i_reg_wdata;
            end if;
            if ( regaddr = MP_CTRL_CHIP_SELECT1_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= conf_write_chip_select(31 downto 0);
            end if;
            if ( regaddr = MP_CTRL_CHIP_SELECT2_REGISTER_W and i_reg_we = '1' ) then
                conf_write_chip_select(63 downto 32) <= i_reg_wdata;
            end if;
            if ( regaddr = MP_CTRL_CHIP_SELECT2_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= conf_write_chip_select(63 downto 32);
            end if;

            if ( regaddr = MP_CTRL_VDAC_REGISTER_W and i_reg_we = '1' ) then
                o_vdac_reg_data <= i_reg_wdata;
                o_vdac_reg_we   <= '1';
            end if;
            if ( regaddr = MP_CTRL_BIAS_REGISTER_W and i_reg_we = '1' ) then
                o_bias_reg_data <= i_reg_wdata;
                o_bias_reg_we   <= '1';
            end if;

            if ( regaddr = MP_CTRL_SLOW_DOWN_REGISTER_W and i_reg_we = '1' ) then
                mp_ctrl_slow_down <= i_reg_wdata;
            end if;
            if ( regaddr = MP_CTRL_SLOW_DOWN_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata <= mp_ctrl_slow_down;
            end if;

            if ( regaddr = MP_CTRL_SPI_BUSY_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata(0) <= mp_spi_busy;
                o_reg_rdata(31 downto 1) <= (others => '0');
            end if;

            loopdirectspi: for I in 0 to N_SPI_g-1 loop
            if ( regaddr = MP_CTRL_DIRECT_SPI_START_REGISTER_W+I and i_reg_we = '1' ) then
                o_mp_direct_spi_data(I) <= i_reg_wdata;
                o_mp_direct_spi_data_wr(I) <= '1';
            end if;
            end loop;

            if ( regaddr = MP_CTRL_DIRECT_SPI_ENABLE_REGISTER_W and i_reg_we = '1' ) then
                mp_ctrl_direct_spi_enable   <= i_reg_wdata(0);
            end if;
            if ( regaddr = MP_CTRL_DIRECT_SPI_ENABLE_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata(0)  <= mp_ctrl_direct_spi_enable;
                o_reg_rdata(31 downto 1) <= (others => '0');
            end if;

            if ( regaddr = MP_CTRL_SPI_ENABLE_REGISTER_W and i_reg_we = '1' ) then
                mp_ctrl_spi_enable   <= i_reg_wdata(0);
            end if;
            if ( regaddr = MP_CTRL_SPI_ENABLE_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata(0)  <= mp_ctrl_spi_enable;
                o_reg_rdata(31 downto 1) <= (others => '0');
            end if;

            if ( regaddr = MP_CTRL_DIRECT_SPI_BUSY_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata(N_SPI_g-1 downto 0) <= i_mp_direct_spi_busy;
                o_reg_rdata(31 downto N_SPI_g) <= (others => '0');
            end if;

            if ( regaddr = MP_CTRL_RESET_REGISTER_W and i_reg_we = '1' ) then
                o_reset_n <= '0';
            end if;

            if ( regaddr = MP_CTRL_RUN_TEST_REGISTER_W and i_reg_we = '1' ) then
                o_run_tdac_test <= '1';
            end if;
        end if;
    end process;

end architecture;
