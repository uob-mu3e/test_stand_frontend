----------------------------------------------------------------------------
-- storage logic of MP configuration
-- M. Mueller
-- Feb 2022

-- everything related to storing mupix configuration on the FEB
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.math_real.all;
use ieee.std_logic_misc.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;


entity mupix_ctrl_config_storage is
    generic( 
        N_CHIPS_g                 : positive := 4
    );
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;

        i_chip_cvb          : in  std_logic_vector(N_CHIPS_g-1 downto 0);   -- used for conf, vdac, bias and combined, 1 hot encoding
        i_chip_tdac         : in  integer range 0 to N_CHIPS_g-1;           -- used for TDACs (will stay like this if i dont implement a way to write equal TDACs in parallel)

        i_conf_reg_data     : in  reg32;
        i_conf_reg_we       : in  std_logic;
        i_vdac_reg_data     : in  reg32;
        i_vdac_reg_we       : in  std_logic;
        i_bias_reg_data     : in  reg32;
        i_bias_reg_we       : in  std_logic;

        i_combined_data     : in  reg32;
        i_combined_data_we  : in  std_logic;

        i_tdac_data         : in  reg32;
        i_tdac_we           : in  std_logic;

        o_data              : out mp_conf_array_out(N_CHIPS_g-1 downto 0);
        i_read              : in  mp_conf_array_in(N_CHIPS_g-1 downto 0)--;
    );

end entity mupix_ctrl_config_storage;

architecture RTL of mupix_ctrl_config_storage is

    signal conf_dpf_full : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal vdac_dpf_full : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal bias_dpf_full : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal tdac_dpf_full : std_logic_vector(N_CHIPS_g-1 downto 0);

    signal conf_dpf_empty : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal vdac_dpf_empty : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal bias_dpf_empty : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal tdac_dpf_empty : std_logic_vector(N_CHIPS_g-1 downto 0);

    signal conf_dpf_we : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal vdac_dpf_we : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal bias_dpf_we : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal tdac_dpf_we : std_logic_vector(N_CHIPS_g-1 downto 0);

    signal conf_dpf_wdata : reg32array(N_CHIPS_g-1 downto 0);
    signal vdac_dpf_wdata : reg32array(N_CHIPS_g-1 downto 0);
    signal bias_dpf_wdata : reg32array(N_CHIPS_g-1 downto 0);
    signal tdac_dpf_wdata : reg32array(N_CHIPS_g-1 downto 0);

    signal conf_dpf_we_splitter : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal vdac_dpf_we_splitter : std_logic_vector(N_CHIPS_g-1 downto 0);
    signal bias_dpf_we_splitter : std_logic_vector(N_CHIPS_g-1 downto 0);

    signal conf_dpf_wdata_splitter : reg32array(N_CHIPS_g-1 downto 0);
    signal vdac_dpf_wdata_splitter : reg32array(N_CHIPS_g-1 downto 0);
    signal bias_dpf_wdata_splitter : reg32array(N_CHIPS_g-1 downto 0);

begin

    mp_conf_splitter: entity work.mp_conf_splitter
      generic map (
        N_CHIPS_g => N_CHIPS_g
      )
      port map (
        i_clk            => i_clk,
        i_reset_n        => i_reset_n,

        i_data           => i_combined_data,
        i_data_we        => i_combined_data_we,
        i_chip_cvb       => i_chip_cvb,

        o_conf_dpf_we    => conf_dpf_we_splitter,
        o_vdac_dpf_we    => vdac_dpf_we_splitter,
        o_bias_dpf_we    => bias_dpf_we_splitter,
        o_conf_dpf_wdata => conf_dpf_wdata_splitter,
        o_vdac_dpf_wdata => vdac_dpf_wdata_splitter,
        o_bias_dpf_wdata => bias_dpf_wdata_splitter
      );


    gen_dp_fifos: for I in 0 to N_CHIPS_g generate

        o_data(I).rdy <= tdac_dpf_full(I) & bias_dpf_full(I) & vdac_dpf_full(I) & conf_dpf_full(I);

        conf_dpf_we(I) <= (i_conf_reg_we and i_chip_cvb(I)) or conf_dpf_we_splitter;
        vdac_dpf_we(I) <= (i_vdac_reg_we and i_chip_cvb(I)) or vdac_dpf_we_splitter;
        bias_dpf_we(I) <= (i_bias_reg_we and i_chip_cvb(I)) or bias_dpf_we_splitter;

        conf_dpf_wdata(I) <= conf_dpf_wdata_splitter when conf_dpf_we_splitter='1' else i_conf_reg_data;
        vdac_dpf_wdata(I) <= vdac_dpf_wdata_splitter when vdac_dpf_we_splitter='1' else i_vdac_reg_data;
        bias_dpf_wdata(I) <= bias_dpf_wdata_splitter when bias_dpf_we_splitter='1' else i_bias_reg_data;

        conf: entity work.dual_port_fifo
          generic map (
            N_BITS_g       => MP_CONFIG_REGS_LENGTH(0),
            WDATA_WIDTH_g  => 32,
            RDATA1_WIDTH_g => 1,
            RDATA2_WIDTH_g => 53
          )
          port map (
            i_clk     => i_clk,
            i_reset_n => i_reset_n,
            o_full    => conf_dpf_full(I),
            o_empty   => conf_dpf_empty(I),
            i_we      => conf_dpf_we(I),
            i_wdata   => conf_dpf_wdata(I),
            i_re1     => i_read(I).spi_read(0),
            o_rdata1  => o_data(I).spi_data(0 downto 0),
            i_re2     => i_read(I).mu3e_read(0),
            o_rdata2  => o_data(I).conf
          );

        vdac: entity work.dual_port_fifo
          generic map (
            N_BITS_g       => MP_CONFIG_REGS_LENGTH(1),
            WDATA_WIDTH_g  => 32,
            RDATA1_WIDTH_g => 1,
            RDATA2_WIDTH_g => 53
          )
          port map (
            i_clk     => i_clk,
            i_reset_n => i_reset_n,
            o_full    => vdac_dpf_full(I),
            o_empty   => vdac_dpf_empty(I),
            i_we      => vdac_dpf_we(I),
            i_wdata   => vdac_dpf_wdata(I),
            i_re1     => i_read(I).spi_read(1),
            o_rdata1  => o_data(I).spi_data(1 downto 1),
            i_re2     => i_read(I).mu3e_read(1),
            o_rdata2  => o_data(I).vdac
          );

        bias: entity work.dual_port_fifo
          generic map (
            N_BITS_g       => MP_CONFIG_REGS_LENGTH(2),
            WDATA_WIDTH_g  => 32,
            RDATA1_WIDTH_g => 1,
            RDATA2_WIDTH_g => 53
          )
          port map (
            i_clk     => i_clk,
            i_reset_n => i_reset_n,
            o_full    => bias_dpf_full(I),
            o_empty   => bias_dpf_empty(I),
            i_we      => bias_dpf_we(I),
            i_wdata   => bias_dpf_wdata(I),
            i_re1     => i_read(I).spi_read(2),
            o_rdata1  => o_data(I).spi_data(2 downto 2),
            i_re2     => i_read(I).mu3e_read(2),
            o_rdata2  => o_data(I).bias
          );
        
        tdac: entity work.dual_port_fifo
          generic map (
            N_BITS_g       => MP_CONFIG_REGS_LENGTH(3),
            WDATA_WIDTH_g  => 28,
            RDATA1_WIDTH_g => 1,
            RDATA2_WIDTH_g => 53
          )
          port map (
            i_clk     => i_clk,
            i_reset_n => i_reset_n,
            o_full    => tdac_dpf_full(I),
            o_empty   => tdac_dpf_empty(I),
            i_we      => tdac_dpf_we(I),
            i_wdata   => tdac_dpf_wdata(I),
            i_re1     => i_read(I).spi_read(3),
            o_rdata1  => o_data(I).spi_data(3 downto 3),
            i_re2     => i_read(I).mu3e_read(3),
            o_rdata2  => o_data(I).tdac
          );

    end generate;

    tdac_memory: entity work.tdac_memory
      generic map (
        N_CHIPS_g => N_CHIPS_g
      )
      port map (
        i_clk            => i_clk,
        i_reset_n        => i_reset_n,

        o_tdac_dpf_we    => tdac_dpf_we,
        o_tdac_dpf_wdata => tdac_dpf_wdata,
        i_tdac_dpf_empty => tdac_dpf_empty,

        i_data           => i_tdac_data,
        i_we             => i_tdac_we,
        i_chip           => i_chip_tdac
      );

end RTL;
