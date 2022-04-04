----------------------------------------------------------------------------
-- Mupix control
-- M. Mueller
-- Feb 2022
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;


entity mupix_ctrl is
    generic( 
        DIRECT_SPI_FIFO_SIZE_g  : positive := 5;
        N_CHIPS_PER_SPI_g       : positive := 3;
        N_SPI_g                 : positive := 4 
    );
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;

        i_reg_add           : in  std_logic_vector(15 downto 0);
        i_reg_re            : in  std_logic;
        o_reg_rdata         : out std_logic_vector(31 downto 0) := (others => '0');
        i_reg_we            : in  std_logic;
        i_reg_wdata         : in  std_logic_vector(31 downto 0);

        o_SIN               : out std_logic_vector( 3 downto 0) := (others => '0');

        o_clock             : out std_logic_vector(N_SPI_g-1 downto 0) := (others => '0');
        o_mosi              : out std_logic_vector(N_SPI_g-1 downto 0) := (others => '0');
        o_csn               : out std_logic_vector(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0) := (others => '0')--;
        );
end entity mupix_ctrl;

architecture RTL of mupix_ctrl is

    signal reset_n                      : std_logic;

    signal slow_down                    : std_logic_vector(15 downto 0) := (others => '0');
    signal slow_down_buf                : std_logic_vector(31 downto 0) := (others => '0');
    signal spi_chip_select_mask         : std_logic_vector(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0) := (others => '0'); -- SPI chip select mask
    signal spi_chip_select_mask_sc      : std_logic_vector(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0) := (others => '0');
    signal spi_chip_select_mask_mp_ctrl : std_logic_vector(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0) := (others => '0');
    signal direct_spi_fifo_full         : std_logic_vector(N_SPI_g-1 downto 0) := (others => '0');
    signal mp_ctrl_to_direct_spi        : reg32array(N_SPI_g-1 downto 0); -- Write data to direct spi from rest of ctrl firmware
    signal mp_ctrl_to_direct_spi_wr     : std_logic_vector(N_SPI_g-1 downto 0) := (others => '0');
    signal sc_to_direct_spi             : reg32array(N_SPI_g-1 downto 0); -- Write data to direct spi from slowcontrol (needs to be enabled using mp_ctrl_direct_spi_ena first)
    signal sc_to_direct_spi_wr          : std_logic_vector(N_SPI_g-1 downto 0) := (others => '0');
    signal mp_direct_spi_busy           : std_logic_vector(N_SPI_g-1 downto 0) := (others => '0');
    signal mp_direct_spi_busy_n         : std_logic_vector(N_SPI_g-1 downto 0) := (others => '0');
    signal mp_ctrl_direct_spi_ena       : std_logic := '0';
    signal mp_ctrl_spi_ena              : std_logic;

    signal signals_from_storage         : mp_conf_array_out(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0) := (others => (rdy => (others => '0'), spi_data => (others => '0'), conf => (others => '0'), bias => (others => '0'), vdac => (others => '0'), tdac => (others => '0')));
    signal signals_to_storage           : mp_conf_array_in(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0);

    signal direct_conf_reg_data         : reg32 := (others => '0');
    signal direct_bias_reg_data         : reg32 := (others => '0');
    signal direct_vdac_reg_data         : reg32 := (others => '0');
    signal direct_conf_reg_we           : std_logic := '0';
    signal direct_bias_reg_we           : std_logic := '0';
    signal direct_vdac_reg_we           : std_logic := '0';

    signal chip_select_cvb              : std_logic_vector(N_CHIPS_PER_SPI_g*N_SPI_g-1 downto 0) := (others => '0');
    signal chip_select_tdac             : integer range 0 to N_CHIPS_PER_SPI_g*N_SPI_g-1 := 0;

    signal combined_data                : reg32 := (others => '0');
    signal combined_data_we             : std_logic := '0';

    signal tdac_data                    : reg32 := (others => '0');
    signal tdac_we                      : std_logic := '0';

    signal run_test                     : std_logic;
    signal n_free_pages                 : reg32;

begin

    slow_down                   <= slow_down_buf(15 downto 0);
    spi_chip_select_mask        <= spi_chip_select_mask_sc when mp_ctrl_direct_spi_ena = '1' else spi_chip_select_mask_mp_ctrl;
    mp_direct_spi_busy_n        <= not mp_direct_spi_busy;



    ------------------------------------------------------
    -- SC regs
    ------------------------------------------------------
    e_mupix_ctrl_reg_mapping : entity work.mupix_ctrl_reg_mapping
    generic map (
        N_CHIPS_PER_SPI_g      => N_CHIPS_PER_SPI_g,
        N_SPI_g                => N_SPI_g--,
    )
    port map (
        i_clk156                    => i_clk,
        i_reset_n                   => i_reset_n,
        o_reset_n                   => reset_n,

        i_reg_add                   => i_reg_add,
        i_reg_re                    => i_reg_re,
        o_reg_rdata                 => o_reg_rdata,
        i_reg_we                    => i_reg_we,
        i_reg_wdata                 => i_reg_wdata,

        i_mp_spi_busy               => '0',

        o_chip_cvb                  => chip_select_cvb,
        o_chip_tdac                 => chip_select_tdac,

        o_conf_reg_data             => direct_conf_reg_data,
        o_conf_reg_we               => direct_conf_reg_we,
        o_vdac_reg_data             => direct_vdac_reg_data,
        o_vdac_reg_we               => direct_vdac_reg_we,
        o_bias_reg_data             => direct_bias_reg_data,
        o_bias_reg_we               => direct_bias_reg_we,

        o_combined_data             => combined_data,
        o_combined_data_we          => combined_data_we,

        o_tdac_data                 => tdac_data,
        o_tdac_we                   => tdac_we,
        o_run_tdac_test             => run_test,

        o_mp_ctrl_slow_down         => slow_down_buf,
        o_mp_direct_spi_data        => sc_to_direct_spi,
        o_mp_direct_spi_data_wr     => sc_to_direct_spi_wr,
        i_mp_direct_spi_busy        => mp_direct_spi_busy,
        o_mp_ctrl_direct_spi_enable => mp_ctrl_direct_spi_ena,
        o_mp_ctrl_spi_enable        => mp_ctrl_spi_ena,

        i_n_free_pages              => n_free_pages--,
    );

    ------------------------------------------------------
    -- config storage
    ------------------------------------------------------

    mupix_ctrl_config_storage_inst: entity work.mupix_ctrl_config_storage
      generic map (
        N_CHIPS_g => N_CHIPS_PER_SPI_g * N_SPI_g
      )
      port map (
        i_clk              => i_clk,
        i_reset_n          => reset_n,

        --inputs to write storage data
        i_chip_cvb         => chip_select_cvb,
        i_chip_tdac        => chip_select_tdac,

        i_conf_reg_data    => direct_conf_reg_data,
        i_conf_reg_we      => direct_conf_reg_we,
        i_vdac_reg_data    => direct_vdac_reg_data,
        i_vdac_reg_we      => direct_vdac_reg_we,
        i_bias_reg_data    => direct_bias_reg_data,
        i_bias_reg_we      => direct_bias_reg_we,

        i_combined_data    => combined_data,
        i_combined_data_we => combined_data_we,
        i_tdac_data        => tdac_data,
        i_tdac_we          => tdac_we,

        --connections to SPI and custom protocol writing
        o_data             => signals_from_storage,
        i_read             => signals_to_storage,

        o_n_free_pages     => n_free_pages--,
      );


    gen_spi: for I in 0 to N_SPI_g-1 generate

    ------------------------------------------------------
    -- custom protocol (aka "mu3e slowcontrol") writing
    ------------------------------------------------------
        mupix_slowcontrol_inst: entity work.mupix_slowcontrol
          generic map (
            N_CHIPS_PER_SPI_g => N_CHIPS_PER_SPI_g
          )
          port map (
            i_clk     => i_clk,
            i_reset_n => reset_n,
            o_read    => open, -- connect to signals_to_storage with reg switch
            i_data    => signals_from_storage(I*N_CHIPS_PER_SPI_g+N_CHIPS_PER_SPI_g-1 downto N_CHIPS_PER_SPI_g*I),
            i_enable  => mp_ctrl_spi_ena,
            o_SIN     => open--,--o_SIN
          );
    ------------------------------------------------------
    -- SPI writing
    ------------------------------------------------------

        mp_ctrl_spi_inst: entity work.mp_ctrl_spi
        generic map (
            N_CHIPS_PER_SPI_g      => N_CHIPS_PER_SPI_g
        )
        port map (
            i_clk                   => i_clk,
            i_reset_n               => reset_n,
            
            o_read                  => signals_to_storage(I*N_CHIPS_PER_SPI_g+N_CHIPS_PER_SPI_g-1 downto N_CHIPS_PER_SPI_g*I),
            i_data                  => signals_from_storage(I*N_CHIPS_PER_SPI_g+N_CHIPS_PER_SPI_g-1 downto N_CHIPS_PER_SPI_g*I),
            o_data_to_direct_spi    => mp_ctrl_to_direct_spi(I),
            o_data_to_direct_spi_we => mp_ctrl_to_direct_spi_wr(I),
            i_direct_spi_fifo_full  => direct_spi_fifo_full(I),
            i_direct_spi_fifo_empty => mp_direct_spi_busy_n(I),
            o_spi_chip_selct_mask   => spi_chip_select_mask_mp_ctrl(I*N_CHIPS_PER_SPI_g+N_CHIPS_PER_SPI_g-1 downto N_CHIPS_PER_SPI_g*I),
            i_run_test              => run_test
        );

        mp_ctrl_direct_spi_inst: entity work.mp_ctrl_direct_spi
        generic map (
            DIRECT_SPI_FIFO_SIZE_g => DIRECT_SPI_FIFO_SIZE_g,
            N_CHIPS_PER_SPI_g      => N_CHIPS_PER_SPI_g--,
        )
        port map (
            i_clk                => i_clk,
            i_reset_n            => reset_n,

            i_fifo_write_mp_ctrl => mp_ctrl_to_direct_spi_wr(I),
            i_fifo_data_mp_ctrl  => mp_ctrl_to_direct_spi(I),
            o_fifo_almost_full   => direct_spi_fifo_full(I),

            i_direct_spi_enable  => mp_ctrl_direct_spi_ena,
            i_fifo_write_direct  => sc_to_direct_spi_wr(I),
            i_fifo_data_direct   => sc_to_direct_spi(I),
            o_direct_spi_busy    => mp_direct_spi_busy(I),

            i_spi_slow_down      => slow_down,
            i_chip_mask          => spi_chip_select_mask(I*N_CHIPS_PER_SPI_g+N_CHIPS_PER_SPI_g-1 downto I*N_CHIPS_PER_SPI_g),

            o_spi                => o_mosi(I),
            o_spi_clk            => o_clock(I),
            o_csn                => o_csn(I*N_CHIPS_PER_SPI_g+N_CHIPS_PER_SPI_g-1 downto I*N_CHIPS_PER_SPI_g)
        );
    end generate;
    

end RTL;