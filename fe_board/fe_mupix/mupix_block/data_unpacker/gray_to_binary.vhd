-- Convert gray code to a binary code
-- Sebastian Dittmeier 
-- September 2017
-- dittmeier@physi.uni-heidelberg.de
-- based on code by Niklaus Berger
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity gray_to_binary is 
    generic(MAXBITS : integer :=16);
    port (
        reset_n                 : in  std_logic;
        clk                     : in  std_logic;
        NBITS                   : in  integer;
        gray_in                 : in  std_logic_vector (MAXBITS-1 DOWNTO 0);
        bin_out                 : out std_logic_vector (MAXBITS-1 DOWNTO 0)
    );
end gray_to_binary;

architecture rtl of gray_to_binary is

begin

process(reset_n, clk)
    variable decoding: std_logic_vector(MAXBITS-1 downto 0);
begin
    if(reset_n = '0') then
        bin_out <= (others => '0');
        decoding := (others => '0');
    elsif(clk'event and clk = '1') then
        decoding := (others => '0');
        if(NBITS > 0 and NBITS <= MAXBITS)then      -- sanity check, otherwise vector out of range or nullvector
            decoding(NBITS-1) := gray_in(NBITS-1);  -- Most significant bit is simply copied
            for i in MAXBITS-2 downto 0 loop        -- loop has to have a constant width
                if(i<(NBITS-1))then                 -- so we have to check that we are actually within range
                    decoding(i)    := gray_in(i) xor decoding(i+1);
                end if;
            end loop;
        end if;
        bin_out     <= decoding;
    end if;
end process;

end rtl;