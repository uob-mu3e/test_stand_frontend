-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Hit multiplexer 3-1
-- Assume hits at maximum every third cycle
-- Niklaus Berger, Feb 2014
-- 
-- nberger@physi.uni-heidelberg.de
--
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_constants.all;



entity hit_multiplexer is 
    port (
        reset_n     : in  std_logic;
        clk         : in  std_logic;
        hit_in1     : IN  STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
        hit_ena1    : IN  STD_LOGIC;
        hit_in2     : IN  STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
        hit_ena2    : IN  STD_LOGIC;
        hit_in3     : IN  STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
        hit_ena3    : IN  STD_LOGIC;
        hit_out     : OUT STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
        hit_ena     : OUT STD_LOGIC
    );
end hit_multiplexer;

architecture RTL of hit_multiplexer is

    signal ena              : std_logic_vector(2 downto 0);
    signal ena_del1         : std_logic_vector(2 downto 0);
    signal ena_del2         : std_logic_vector(2 downto 0);

    signal ena_del1_nors    : std_logic_vector(2 downto 0);
    signal ena_del2_nors    : std_logic_vector(2 downto 0);

    signal hit1             : STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
    signal hit2             : STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
    signal hit3             : STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);

begin

process(clk, reset_n)
begin
    if(reset_n = '0') then
        hit_ena         <= '0';
        ena             <= (others => '0');
        ena_del1        <= (others => '0');
        ena_del2        <= (others => '0');
        ena_del1_nors   <= (others => '0');
        ena_del2_nors   <= (others => '0');
    elsif(clk'event and clk = '1') then

        ena         <= ((hit_ena3 & hit_ena2 & hit_ena1) and (not ena)) and (not ena_del1_nors);
        ena_del1    <= ena;
        ena_del2    <= ena_del1;

        ena_del1_nors   <= ena;
        ena_del2_nors   <= ena_del1_nors;

        if(hit_ena1 = '1' and ena(0) = '0' and ena_del1_nors(0) = '0') then
            hit1    <= hit_in1;
        end if;
        
        if(hit_ena2 = '1' and ena(1) = '0' and ena_del1_nors(1) = '0') then
            hit2    <= hit_in2;
        end if;
        
        if(hit_ena3 = '1' and ena(2) = '0' and ena_del1_nors(2) = '0') then
            hit3    <= hit_in3;
        end if;
        
        if(ena_del2(0) = '1') then
            hit_out     <= hit1;
            hit_ena     <= '1';
        elsif (ena_del2(1) = '1') then
            hit_out     <= hit2;
            hit_ena     <= '1';
        elsif (ena_del2(2) = '1') then
            hit_out     <= hit3;
            hit_ena     <= '1';
        elsif(ena_del1(0) = '1') then
            hit_out     <= hit1;
            hit_ena     <= '1';
            ena_del2(0) <= '0';
        elsif (ena_del1(1) = '1') then
            hit_out     <= hit2;
            hit_ena     <= '1';
            ena_del2(1) <= '0';
        elsif (ena_del1(2) = '1') then
            hit_out     <= hit3;
            hit_ena     <= '1';
            ena_del2(2) <= '0';
        elsif(ena(0) = '1') then
            hit_out     <= hit1;
            hit_ena     <= '1';
            ena_del1(0) <= '0';
        elsif (ena(1) = '1') then
            hit_out     <= hit2;
            hit_ena     <= '1';
            ena_del1(1) <= '0';
        elsif (ena(2) = '1') then
            hit_out     <= hit3;
            hit_ena     <= '1';
            ena_del1(2) <= '0';
        else
            hit_ena     <= '0';
        end if;

    end if;
end process;

end rtl;