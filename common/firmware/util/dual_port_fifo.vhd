----------------------------------------------------------------------------
-- Dual port FIFO
-- M. Mueller
-- Feb 2022
-- TODO: make this behave more "fifo-like" using dual port RAM with output width > max(RDATA1_WIDTH, RDATA2_WIDTH). Buffer RAM output once, shift readpointer on ram output and buffer by RDATA1_WIDTH, RDATA2_WIDTH
-- UNTIL THAT IS DONE : START READING ONLY WHEN FULL, START WRITING ONLY WHEN EMPTY, NO READ & WRITE AT THE SAME TIME !!!
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;

entity dual_port_fifo is
    generic( 
        N_BITS_g                : positive := 2047;
        N_BITS_ACTUAL_g         : positive := 2047; -- TODO find more elegant solution
        WDATA_WIDTH_g           : positive := 32;
        RDATA1_WIDTH_g          : positive := 32;
        RDATA2_WIDTH_g          : positive := 32;
        IS_TDAC_DPF_g           : boolean  := false--;
    );
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;

        o_full              : out std_logic;
        o_empty             : out std_logic;

        i_we                : in  std_logic;
        i_wdata             : in  std_logic_vector(WDATA_WIDTH_g-1 downto 0);

        i_re1               : in  std_logic;
        o_rdata1            : out std_logic_vector(RDATA1_WIDTH_g-1 downto 0);

        i_re2               : in  std_logic;
        o_rdata2            : out std_logic_vector(RDATA2_WIDTH_g-1 downto 0)--;
        );
end entity dual_port_fifo;

architecture RTL of dual_port_fifo is

    signal shift_reg            : std_logic_vector(N_BITS_g-1 downto 0);
    signal bits_used            : integer range 0 to N_BITS_g + WDATA_WIDTH_g;

begin

    o_rdata1        <= shift_reg(RDATA1_WIDTH_g-1 downto 0);
    o_rdata2        <= shift_reg(RDATA2_WIDTH_g-1 downto 0);

    o_full  <= '1' when (bits_used = N_BITS_ACTUAL_g or (i_we='1' and bits_used + WDATA_WIDTH_g >= N_BITS_g)) else '0';
    o_empty <= '1' when (bits_used = 0 or (i_re1 = '1' and bits_used - RDATA1_WIDTH_g <= 0) or (i_re2 = '1' and bits_used - RDATA2_WIDTH_g <=0)) else '0';

    process(i_clk, i_reset_n)
    begin
    if(i_reset_n = '0') then 
        shift_reg       <= (others => '0');
        bits_used       <= 0;
    elsif(rising_edge(i_clk))then

        if(i_re1 = '1') then
            shift_reg(N_BITS_g-1-RDATA1_WIDTH_g downto 0) <= shift_reg(N_BITS_g-1 downto RDATA1_WIDTH_g);
            shift_reg(N_BITS_g-1 downto N_BITS_g-RDATA1_WIDTH_g) <= (others => '0');
            bits_used <= bits_used - RDATA1_WIDTH_g;
        elsif(i_re2 = '1') then
            shift_reg(N_BITS_g-1-RDATA2_WIDTH_g downto 0) <= shift_reg(N_BITS_g-1 downto RDATA2_WIDTH_g);
            shift_reg(N_BITS_g-1 downto N_BITS_g-RDATA2_WIDTH_g) <= (others => '0');
            bits_used <= bits_used - RDATA2_WIDTH_g;
        elsif(i_we = '1') then
            if(IS_TDAC_DPF_g = false) then 
                shift_reg(N_BITS_g-1 downto N_BITS_g-WDATA_WIDTH_g) <= i_wdata;
                shift_reg(N_BITS_g-1-WDATA_WIDTH_g downto 0) <= shift_reg(N_BITS_g-1 downto WDATA_WIDTH_g);
            else
                -- this is the by far easiest point to do tdac row conversion to digital addresses
                for I in 508 to 511 loop
                    shift_reg(tdac_conversion_index(I)) <= i_wdata(I-508);
                end loop;
                for I in 0 to 507 loop
                    shift_reg(tdac_conversion_index(I)) <= shift_reg(tdac_conversion_index(I+4));
                end loop;
                -- tdac row conversion done ...

            end if;

            if(bits_used + WDATA_WIDTH_g < N_BITS_ACTUAL_g) then 
                bits_used <= bits_used + WDATA_WIDTH_g;
            else
                bits_used <= N_BITS_ACTUAL_g;
            end if;
        end if;
    end if;
    end process;

end RTL;