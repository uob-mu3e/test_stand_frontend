----------------------------------------------------------------------------
-- storage logic of MP configuration
-- M. Mueller
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.math_real.all;
use work.mupix_constants.all;
use work.mupix_registers.all;
use work.daq_constants.all;

entity mupix_ctrl_config_storage is
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;
        i_clr_all           : in  std_logic;

        -- data written from sc regs
        i_data              : in  std_logic_vector(32*6-1 downto 0);
        i_wrreq             : in  std_logic_vector(5 downto 0);

        -- output data, 1 bit for each of the 6 shift-regs on the mupix
        o_data              : out std_logic_vector(5 downto 0);
        o_is_writing        : out std_logic_vector(5 downto 0); -- active as long as the full size of the shift reg is not reached
        i_enable            : in  std_logic_vector(5 downto 0); -- triggers write to shift reg on mupix on trans. 0->1 (make sure the full data was stored or will be stored very soon for this shift reg)
        i_rdreq             : in  std_logic--; -- spi writing requests the next bits  only 1 cycle active !!
        );
end entity mupix_ctrl_config_storage;

architecture RTL of mupix_ctrl_config_storage is

    signal fifo_read                : std_logic_vector(5 downto 0);
    signal data_buffer              : std_logic_vector(32*6-1 downto 0);
    type bitpos_t                   is array (5 downto 0) of integer range 31 downto 0;
    type bitpos_global_t            is array (5 downto 0) of integer range 1000 downto 0; -- TODO: how to max(MP_CONFIG_REGS_LENGTH) in vhdl ?
    signal bitpos                   : bitpos_t;
    signal bitpos_global            : bitpos_global_t;
    signal is_writing               : std_logic_vector(5 downto 0);
    signal enable_prev              : std_logic_vector(5 downto 0);
begin



    process(i_clk, i_reset_n)
    begin
        if(i_reset_n = '0')then
            fifo_read       <= (others => '0');
            bitpos          <= (others => 0);
            bitpos_global   <= (others => 0);
            is_writing      <= (others => '0');

        elsif(rising_edge(i_clk))then
            enable_prev <= i_enable;
            fifo_read   <= (others => '0');
            o_is_writing <= is_writing;

            for I in 0 to 5 loop
                if(i_rdreq='1') then
                    if(is_writing(I)='1') then
                        o_data(I)           <= data_buffer(I*32+bitpos(I));
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
                end if;

                if(enable_prev(I)='0' and i_enable(I)='1') then
                    is_writing(I)           <= '1';
                end if;
            end loop;
        end if;
    end process;


    gen_config_storage: for I in 0 to 5 generate
        mp_ctrl_storage_fifo: entity work.ip_scfifo
        generic map(
            ADDR_WIDTH      => integer(ceil(log2(real(MP_CONFIG_REGS_LENGTH(I))))),
            DATA_WIDTH      => 32,
            SHOWAHEAD       => "ON",
            DEVICE          => "ARRIA V"--,
        )
        port map (
            clock           => i_clk,
            sclr            => i_clr_all or (not i_reset_n),
            data            => i_data(I*32 + 31 downto I*32),
            wrreq           => i_wrreq(I),
            q               => data_buffer(I*32 + 31 downto I*32),
            rdreq           => fifo_read(I)--,
        );
    end generate gen_config_storage;

end RTL;
