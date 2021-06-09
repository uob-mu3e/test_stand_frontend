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
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;
        i_clr_fifo          : in  std_logic_vector(5 downto 0);

        -- data written from sc regs
        i_data              : in  std_logic_vector(32*6-1 downto 0);
        i_wrreq             : in  std_logic_vector(5 downto 0);

        -- data written from sc regs (for 1 complete configuration, all 6 mupix shift regs in the order BIAS, CONF, VDAC, COL, TEST, TDAC)
        i_data_all          : in  std_logic_vector(31 downto 0);
        i_wrreq_all         : in  std_logic;

        -- output data, 1 bit for each of the 6 shift-regs on the mupix
        o_data              : out std_logic_vector(5 downto 0);
        o_is_writing        : out std_logic_vector(5 downto 0); -- active as long as the full size of the shift reg is not reached
        i_enable            : in  std_logic_vector(5 downto 0); -- triggers write to shift reg on mupix on trans. 0->1 (make sure the full data was stored or will be stored very soon for this shift reg)
        i_rdreq             : in  std_logic--; -- spi writing requests the next bits  only 1 cycle active !!
    );
end entity mupix_ctrl_config_storage;

architecture RTL of mupix_ctrl_config_storage is

    signal fifo_read                : std_logic_vector(5 downto 0);
    signal fifo_clear               : std_logic_vector(5 downto 0);
    signal fifo_write               : std_logic_vector(5 downto 0);
    signal fifo_wdata               : reg32array(5 downto 0);
    signal fifo_wdata_final         : reg32array(5 downto 0);
    signal fifo_write_final         : std_logic_vector(5 downto 0);
    signal data_buffer              : std_logic_vector(32*6-1 downto 0);
    type bitpos_t                   is array (5 downto 0) of integer range 31 downto 0;
    type bitpos_global_t            is array (5 downto 0) of integer range 1000 downto 0; -- TODO: how to max(MP_CONFIG_REGS_LENGTH) in vhdl ?
    signal bitpos                   : bitpos_t;
    signal bitpos_global            : bitpos_global_t;
    signal is_writing               : std_logic_vector(5 downto 0);
    signal enable_prev              : std_logic_vector(5 downto 0);
    signal data_in_all_position_32  : integer range 84 downto 0;
    signal data_in_leftovers        : std_logic_vector(31 downto 0);
    signal internal_enable          : std_logic;


begin

    process(i_clk, i_reset_n) -- read process
    begin
        if(i_reset_n = '0')then
            fifo_read       <= (others => '0');
            bitpos          <= (others => 0);
            bitpos_global   <= (others => 0);
            is_writing      <= (others => '0');
            o_is_writing    <= (others => '0');
            o_data          <= (others => '0');

        elsif(rising_edge(i_clk))then
            for i in 0 to 5 loop
                enable_prev(i) <= i_enable(i) or internal_enable;
            end loop;
            fifo_read   <= (others => '0');
            o_is_writing <= is_writing;

            for I in 0 to 5 loop
                if(i_rdreq='1') then
                    if(is_writing(I)='1') then
                        o_data(I)           <= data_buffer((I+1)*32-1-bitpos(I)); -- TODO: leave this invert here ?, makes live easier in software at the moment M.Mueller FEB2021
                        bitpos_global(I)    <= bitpos_global(I)+1;
                        if(bitpos(I)=31) then
                            bitpos(I)       <= 0;
                            fifo_read(I)    <= '1';
                        else
                            bitpos(I)       <= bitpos(I)+1;
                        end if;
                    end if;

                elsif(bitpos_global(I)=MP_CONFIG_REGS_LENGTH(I)) then
                    is_writing(I)           <= '0';
                    bitpos_global(I)        <= 0;
                    bitpos(I)               <= 0;
                end if;

                if(enable_prev(I)='0' and (i_enable(I)='1' or internal_enable='1')) then
                    is_writing(I)           <= '1';
                end if;
            end loop;
        end if;
    end process;

    process(i_clk, i_reset_n) -- write process

    begin
        if(i_reset_n = '0')then
            fifo_wdata              <= (others => (others => '0'));
            fifo_write              <= (others => '0');
            data_in_all_position_32 <= 0;
            data_in_leftovers       <= (others => '0');
            internal_enable         <= '0';
        elsif(rising_edge(i_clk))then
            fifo_write              <= (others => '0');
            fifo_wdata              <= (others => (others => '0'));
            internal_enable         <= '0';

            --Bits: BIAS 210, CONF 90, VDAC 80, COL 896, TDAC 512, TEST 896

            if(or_reduce(i_clr_fifo) = '1') then
                data_in_all_position_32 <= 0;

            elsif(i_wrreq_all='1') then
                data_in_all_position_32 <= data_in_all_position_32 + 1;
                
                case data_in_all_position_32 is
                    when 0 to 5 =>
                        -- 192 BIAS Bits
                        fifo_write(WR_BIAS_BIT) <= '1';
                        fifo_wdata(WR_BIAS_BIT) <= i_data_all;
                        data_in_leftovers       <= (others => '0');
                    when 6 =>
                        -- 18 Bias Bits
                        fifo_write(WR_BIAS_BIT) <= '1';
                        fifo_wdata(WR_BIAS_BIT) <= i_data_all(31 downto 14) & x"000" & "00";

                        -- 14 leftover
                        data_in_leftovers(13 downto 0) <= i_data_all(13 downto 0);
                    when 7 to 8 =>
                        -- 64 CONF Bits
                        fifo_write(WR_CONF_BIT) <= '1';
                        fifo_wdata(WR_CONF_BIT) <= data_in_leftovers(13 downto 0) & i_data_all(31 downto 14);
                        
                        -- 14 leftover
                        data_in_leftovers(13 downto 0) <= i_data_all(13 downto 0);
                     when 9 =>
                        -- 26 CONF Bits  (14 from leftover, 12 from input --> 20 new leftover)
                        fifo_write(WR_CONF_BIT) <= '1';
                        fifo_wdata(WR_CONF_BIT) <= data_in_leftovers(31 downto 18) & i_data_all(31 downto 20) & "000000";
                        
                        -- 20 leftover
                        data_in_leftovers(19 downto 0) <= i_data_all(19 downto 0);
                     when 10 to 11 => 
                        -- 64 VDAC Bits  (20 from leftover, 12 from input --> 20 new leftover)
                        fifo_write(WR_VDAC_BIT) <= '1';
                        fifo_wdata(WR_VDAC_BIT) <= data_in_leftovers(19 downto 0) & i_data_all(31 downto 20);
                        
                        -- 20 leftover
                        data_in_leftovers(19 downto 0) <= i_data_all(19 downto 0);
                     when 12 =>
                        -- 16 VDAC Bits  (16 from leftover --> no new leftover)
                        fifo_write(WR_VDAC_BIT) <= '1';
                        fifo_wdata(WR_VDAC_BIT) <= data_in_leftovers(19 downto 4) & x"0000";
                        
                        -- 32 COL Bits
                        fifo_write(WR_COL_BIT) <= '1';
                        fifo_wdata(WR_COL_BIT) <= i_data_all;
                     when 13 to 39 => 
                        -- 864 COL Bits
                        fifo_write(WR_COL_BIT) <= '1';
                        fifo_wdata(WR_COL_BIT) <= i_data_all;
                     when 40 to 55 =>
                        -- 512 TDAC Bits
                        fifo_write(WR_TDAC_BIT) <= '1';
                        fifo_wdata(WR_TDAC_BIT) <= i_data_all;
                     when 56 to 82 =>
                        -- 896 Test Bits
                        fifo_write(WR_TEST_BIT) <= '1';
                        fifo_wdata(WR_TEST_BIT) <= i_data_all;
                     when 83 =>
                        fifo_write(WR_TEST_BIT) <= '1';
                        fifo_wdata(WR_TEST_BIT) <= i_data_all;
                        
                        -- All data is there --> trigger start of spi writing
                        internal_enable         <= '1';
                        
                        data_in_all_position_32 <= 0;
                     when others =>
                        data_in_all_position_32 <= 0;
                end case;
            end if;
        end if;
    end process;

    gen_config_storage: for I in 0 to 5 generate
        fifo_clear(I)       <= i_clr_fifo(I) or (not i_reset_n);
        fifo_wdata_final(I) <= i_data(I*32 + 31 downto I*32) when i_wrreq(I)='1' else fifo_wdata(I);
        fifo_write_final(I) <= i_wrreq(I) or fifo_write(I);
        
        mp_ctrl_storage_fifo: entity work.ip_scfifo
        generic map(
            ADDR_WIDTH      => integer(ceil(log2(real(MP_CONFIG_REGS_LENGTH(I))))),
            DATA_WIDTH      => 32,
            SHOWAHEAD       => "ON",
            DEVICE          => "ARRIA V"--,
        )
        port map (
            clock           => i_clk,
            sclr            => fifo_clear(I),
            data            => fifo_wdata_final(I),
            wrreq           => fifo_write_final(I),
            q               => data_buffer(I*32 + 31 downto I*32),
            rdreq           => fifo_read(I)--,
        );
    end generate gen_config_storage;

end RTL;
