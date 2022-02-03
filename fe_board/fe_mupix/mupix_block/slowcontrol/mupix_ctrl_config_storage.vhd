----------------------------------------------------------------------------
-- storage logic of MP configuration
-- M. Mueller
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

        i_chip_cvb          : in  std_logic_vector(N_CHIPS_g-1 downto 0);   -- used for conf, vdac, bias and combined
        i_chip_tdac         : in  integer range 0 to N_CHIPS_g;             -- used for TDACs (will stay like this if i dont implement a way to write equal TDACs in parallel)

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

        -- i_clr_fifo          : in  std_logic_vector(5 downto 0);

        -- -- data written from sc regs
        -- i_data              : in  std_logic_vector(32*6-1 downto 0);
        -- i_wrreq             : in  std_logic_vector(5 downto 0);

        -- -- data written from sc regs (for 1 complete configuration, all 6 mupix shift regs in the order BIAS, CONF, VDAC, COL, TEST, TDAC)
        -- i_data_all          : in  std_logic_vector(31 downto 0);
        -- i_wrreq_all         : in  std_logic;

        -- -- output data, 1 bit for each of the 6 shift-regs on the mupix
        -- o_data              : out std_logic_vector(5 downto 0);
        -- o_is_writing        : out std_logic_vector(5 downto 0); -- active as long as the full size of the shift reg is not reached
        -- i_enable            : in  std_logic_vector(5 downto 0); -- triggers write to shift reg on mupix on trans. 0->1 (make sure the full data was stored or will be stored very soon for this shift reg)
        -- i_rdreq             : in  std_logic--; -- spi writing requests the next bits  only 1 cycle active !!
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

    -- signal fifo_read                : std_logic_vector(5 downto 0) := (others => '0');
    -- signal fifo_clear               : std_logic_vector(5 downto 0) := (others => '0');
    -- signal fifo_write               : std_logic_vector(5 downto 0) := (others => '0');
    -- signal fifo_wdata               : reg32array(5 downto 0);
    -- signal fifo_wdata_final         : reg32array(5 downto 0);
    -- signal fifo_write_final         : std_logic_vector(5 downto 0) := (others => '0');
    -- signal data_buffer              : std_logic_vector(32*6-1 downto 0);
    -- type bitpos_t                   is array (5 downto 0) of integer range 31 downto 0;
    -- type bitpos_global_t            is array (5 downto 0) of integer range 1000 downto 0; -- TODO: how to max(MP_CONFIG_REGS_LENGTH) in vhdl ?
    -- signal bitpos                   : bitpos_t;
    -- signal bitpos_global            : bitpos_global_t;
    -- signal is_writing               : std_logic_vector(5 downto 0);
    -- signal enable_prev              : std_logic_vector(5 downto 0);
    -- signal data_in_all_position_32  : integer range 84 downto 0;
    -- signal data_in_leftovers        : std_logic_vector(31 downto 0);
    -- signal internal_enable          : std_logic;


begin


    gen_dp_fifos: for I in 0 to N_CHIPS_g generate

        o_data(I).rdy <= tdac_dpf_full(I) & bias_dpf_full(I) & vdac_dpf_full(I) & conf_dpf_full(I);

        conf_dpf_we(I) <= i_conf_reg_we and i_chip_cvb(I);
        vdac_dpf_we(I) <= i_vdac_reg_we and i_chip_cvb(I);
        bias_dpf_we(I) <= i_bias_reg_we and i_chip_cvb(I);
        tdac_dpf_we -- TODO

        conf_dpf_wdata(I) <= i_conf_reg_data;
        vdac_dpf_wdata(I) <= i_vdac_reg_data;
        bias_dpf_wdata(I) <= i_bias_reg_data;
        tdac_dpf_wdata -- TODO

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


    -- process(i_clk, i_reset_n) -- read process
    -- begin
    --     if(i_reset_n = '0')then
    --         fifo_read       <= (others => '0');
    --         bitpos          <= (others => 0);
    --         bitpos_global   <= (others => 0);
    --         is_writing      <= (others => '0');
    --         o_is_writing    <= (others => '0');
    --         o_data          <= (others => '0');

    --     elsif(rising_edge(i_clk))then
    --         for i in 0 to 5 loop
    --             enable_prev(i) <= i_enable(i) or internal_enable;
    --         end loop;
    --         fifo_read   <= (others => '0');
    --         o_is_writing <= is_writing;

    --         for I in 0 to 5 loop
    --             if(i_rdreq='1') then
    --                 if(is_writing(I)='1') then
    --                     o_data(I)           <= data_buffer((I+1)*32-1-bitpos(I)); -- TODO: leave this invert here ?, makes live easier in software at the moment M.Mueller FEB2021
    --                     bitpos_global(I)    <= bitpos_global(I)+1;
    --                     if(bitpos(I)=31) then
    --                         bitpos(I)       <= 0;
    --                         fifo_read(I)    <= '1';
    --                     else
    --                         bitpos(I)       <= bitpos(I)+1;
    --                     end if;
    --                 end if;

    --             elsif(bitpos_global(I)=MP_CONFIG_REGS_LENGTH(I)) then
    --                 is_writing(I)           <= '0';
    --                 bitpos_global(I)        <= 0;
    --                 bitpos(I)               <= 0;
    --             end if;

    --             if(enable_prev(I)='0' and (i_enable(I)='1' or internal_enable='1')) then
    --                 is_writing(I)           <= '1';
    --             end if;
    --         end loop;
    --     end if;
    -- end process;

    -- process(i_clk, i_reset_n) -- write process

    -- begin
    --     if(i_reset_n = '0')then
    --         fifo_wdata              <= (others => (others => '0'));
    --         fifo_write              <= (others => '0');
    --         data_in_all_position_32 <= 0;
    --         data_in_leftovers       <= (others => '0');
    --         internal_enable         <= '0';
    --     elsif(rising_edge(i_clk))then
    --         fifo_write              <= (others => '0');
    --         fifo_wdata              <= (others => (others => '0'));
    --         internal_enable         <= '0';

    --         --Bits: BIAS 210, CONF 90, VDAC 80, COL 896, TDAC 512, TEST 896

    --         if(or_reduce(i_clr_fifo) = '1') then
    --             data_in_all_position_32 <= 0;

    --         elsif(i_wrreq_all='1') then
    --             data_in_all_position_32 <= data_in_all_position_32 + 1;
                
    --             case data_in_all_position_32 is
    --                 when 0 to 5 =>
    --                     -- 192 BIAS Bits
    --                     fifo_write(WR_BIAS_BIT) <= '1';
    --                     fifo_wdata(WR_BIAS_BIT) <= i_data_all;
    --                     data_in_leftovers       <= (others => '0');
    --                 when 6 =>
    --                     -- 18 Bias Bits
    --                     fifo_write(WR_BIAS_BIT) <= '1';
    --                     fifo_wdata(WR_BIAS_BIT) <= i_data_all(31 downto 16) & x"0000";

    --                     -- 16 leftover
    --                     data_in_leftovers(15 downto 0) <= i_data_all(15 downto 0);
    --                 when 7 to 8 =>
    --                     -- 64 CONF Bits
    --                     fifo_write(WR_CONF_BIT) <= '1';
    --                     fifo_wdata(WR_CONF_BIT) <= data_in_leftovers(15 downto 0) & i_data_all(31 downto 16);
                        
    --                     -- 16 leftover
    --                     data_in_leftovers(15 downto 0) <= i_data_all(15 downto 0);
    --                  when 9 =>
    --                     -- 26 CONF Bits  (16 from leftover, 10 from input --> 22 new leftover)
    --                     fifo_write(WR_CONF_BIT) <= '1';
    --                     fifo_wdata(WR_CONF_BIT) <= data_in_leftovers(15 downto 0) & i_data_all(31 downto 22) & "000000";
                        
    --                     -- 22 leftover
    --                     data_in_leftovers(21 downto 0) <= i_data_all(21 downto 0);
    --                  when 10 to 11 => 
    --                     -- 64 VDAC Bits  (22 from leftover, 10 from input --> 22 new leftover)
    --                     fifo_write(WR_VDAC_BIT) <= '1';
    --                     fifo_wdata(WR_VDAC_BIT) <= data_in_leftovers(21 downto 0) & i_data_all(31 downto 22);
                        
    --                     -- 22 leftover
    --                     data_in_leftovers(21 downto 0) <= i_data_all(21 downto 0);
    --                  when 12 =>
    --                     -- 16 VDAC Bits  (16 from leftover)
    --                     fifo_write(WR_VDAC_BIT) <= '1';
    --                     fifo_wdata(WR_VDAC_BIT) <= data_in_leftovers(21 downto 6) & x"0000";
    --                     -- 6 new leftover ... but will throw them away here since this probably makes the software easier when col, tdac and test are read from file instead of odb
    --                     -- --> col tac and test are 32 bit aligned and do not overlap with odb config
                        
    --                     -- 32 COL Bits
    --                     fifo_write(WR_COL_BIT) <= '1';
    --                     fifo_wdata(WR_COL_BIT) <= i_data_all;
    --                  when 13 to 39 => 
    --                     -- 864 COL Bits
    --                     fifo_write(WR_COL_BIT) <= '1';
    --                     fifo_wdata(WR_COL_BIT) <= i_data_all;
    --                  when 40 to 55 =>
    --                     -- 512 TDAC Bits
    --                     fifo_write(WR_TDAC_BIT) <= '1';
    --                     fifo_wdata(WR_TDAC_BIT) <= i_data_all;
    --                  when 56 to 82 =>
    --                     -- 896 Test Bits
    --                     fifo_write(WR_TEST_BIT) <= '1';
    --                     fifo_wdata(WR_TEST_BIT) <= i_data_all;
    --                  when 83 =>
    --                     fifo_write(WR_TEST_BIT) <= '1';
    --                     fifo_wdata(WR_TEST_BIT) <= i_data_all;
                        
    --                     -- All data is there --> trigger start of spi writing
    --                     internal_enable         <= '1';
                        
    --                     data_in_all_position_32 <= 0;
    --                  when others =>
    --                     data_in_all_position_32 <= 0;
    --             end case;
    --         end if;
    --     end if;
    -- end process;

    -- gen_config_storage: for I in 0 to 5 generate
    --     fifo_clear(I)       <= i_clr_fifo(I) or (not i_reset_n);
    --     fifo_wdata_final(I) <= i_data(I*32 + 31 downto I*32) when i_wrreq(I)='1' else fifo_wdata(I);
    --     fifo_write_final(I) <= i_wrreq(I) or fifo_write(I);
        
    --     mp_ctrl_storage_fifo: entity work.ip_scfifo
    --     generic map(
    --         ADDR_WIDTH      => integer(ceil(log2(real(MP_CONFIG_REGS_LENGTH(I))))),
    --         DATA_WIDTH      => 32,
    --         SHOWAHEAD       => "ON",
    --         REGOUT          => 0,
    --         DEVICE          => "ARRIA V"--,
    --     )
    --     port map (
    --         clock           => i_clk,
    --         sclr            => fifo_clear(I),
    --         data            => fifo_wdata_final(I),
    --         wrreq           => fifo_write_final(I),
    --         q               => data_buffer(I*32 + 31 downto I*32),
    --         rdreq           => fifo_read(I)--,
    --     );
    -- end generate gen_config_storage;

end RTL;
