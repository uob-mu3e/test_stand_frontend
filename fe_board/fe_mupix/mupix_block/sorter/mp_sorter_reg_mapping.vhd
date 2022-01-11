-- mupix sorter reg mapping
-- M. Mueller, Nov 2021

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;
entity mp_sorter_reg_mapping is
port (
    i_clk156                    : in  std_logic;
    i_reset_n                   : in  std_logic;

    i_reg_add                   : in  std_logic_vector(15 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    i_nintime                   : in  reg_array;
    i_noutoftime                : in  reg_array;
    i_noverflow                 : in  reg_array;
    i_nout                      : in  reg32;
    i_credit                    : in  reg32;

    o_sorter_delay              : out ts_t--;
);
end entity;

architecture rtl of mp_sorter_reg_mapping is
    signal sorter_delay         : ts_t;

    signal nintime                   : reg_array;
    signal noutoftime                : reg_array;
    signal noverflow                 : reg_array;
    signal nout                      : reg32;
    signal credit                    : reg32;

    begin
    process (i_clk156, i_reset_n)
        variable regaddr : integer;
    begin
        if (i_reset_n = '0') then 
            o_sorter_delay              <= (others => '0');
            sorter_delay                <= (others => '0');

        elsif(rising_edge(i_clk156)) then
            o_sorter_delay              <= sorter_delay;

            -- register sorter signals once in 156 Mhz domain and put a false path between i_nintime and nintime, ...
            nintime                     <= i_nintime;
            noutoftime                  <= i_noutoftime;
            noverflow                   <= i_noverflow;
            nout                        <= i_nout;
            credit                      <= i_credit;

            regaddr                     := to_integer(unsigned(i_reg_add));
            -----------------------------------------------------------------
            ---- sorter regs ------------------------------------------------
            -----------------------------------------------------------------
            for I in 0 to 11 loop 
                if ( regaddr = I + MP_SORTER_NINTIME_REGISTER_R and i_reg_re = '1' ) then
                    o_reg_rdata <= nintime(I);
                end if;
            end loop;

            for I in 0 to 11 loop 
                if ( regaddr = I + MP_SORTER_NOUTOFTIME_REGISTER_R and i_reg_re = '1' ) then
                    o_reg_rdata <= noutoftime(I);
                end if;
            end loop;

            for I in 0 to 11 loop 
                if ( regaddr = I + MP_SORTER_NOVERFLOW_REGISTER_R and i_reg_re = '1' ) then
                    o_reg_rdata <= noverflow(I);
                end if;
            end loop;

            if ( regaddr = MP_SORTER_NOUT_REGISTER_R and i_reg_re = '1' ) then
                o_reg_rdata <= nout;
            end if;

            if ( regaddr = MP_SORTER_CREDIT_REGISTER_R and i_reg_we = '1' ) then
                o_reg_rdata <= credit;
            end if;

            if ( regaddr = MP_SORTER_DELAY_REGISTER_W and i_reg_we = '1' ) then
                sorter_delay <= i_reg_wdata(TSRANGE);
            end if;
            if ( regaddr = MP_SORTER_DELAY_REGISTER_W and i_reg_re = '1' ) then
                o_reg_rdata(TSRANGE) <= sorter_delay;
            end if;
        end if;
    end process;
end architecture;
