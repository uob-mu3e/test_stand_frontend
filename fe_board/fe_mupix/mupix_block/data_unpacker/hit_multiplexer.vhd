-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Hit multiplexer 3-1
-- Assume hits at maximum every third cycle
-- Niklaus Berger, Feb 2014
-- 
-- nberger@physi.uni-heidelberg.de
--
-- added clk transition 125 -> 156.25 
-- M.Mueller
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_constants.all;

entity hit_multiplexer is 
    port (
        i_reset_n     : in  std_logic;
        i_clk125      : in  std_logic;
        i_clk156      : in  std_logic;
        i_hit_in1     : in  std_logic_vector(HITSIZE-1 DOWNTO 0);
        i_hit_ena1    : in  std_logic;
        i_hit_in2     : in  std_logic_vector(HITSIZE-1 DOWNTO 0);
        i_hit_ena2    : in  std_logic;
        i_hit_in3     : in  std_logic_vector(HITSIZE-1 DOWNTO 0);
        i_hit_ena3    : in  std_logic;
        o_hit_out     : out std_logic_vector(HITSIZE-1 DOWNTO 0);
        o_hit_ena     : out std_logic
    );
end hit_multiplexer;

architecture RTL of hit_multiplexer is

    signal ena      : std_logic_vector(2 downto 0);
    signal ena_del1 : std_logic_vector(2 downto 0);
    signal ena_del2 : std_logic_vector(2 downto 0);

    signal ena_del1_nors : std_logic_vector(2 downto 0);
    signal ena_del2_nors : std_logic_vector(2 downto 0);

    signal hit1     : STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
    signal hit2     : STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
    signal hit3     : STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);

    signal hit_in1  : std_logic_vector(HITSIZE-1 DOWNTO 0);
    signal hit_in2  : std_logic_vector(HITSIZE-1 DOWNTO 0);
    signal hit_in3  : std_logic_vector(HITSIZE-1 DOWNTO 0);

    signal hit_ena1 : std_logic;
    signal hit_ena2 : std_logic;
    signal hit_ena3 : std_logic;

begin

-----------------------------------
-- clock-in into 156.25 MHz 
-----------------------------------
-- TODO: drive everything below these fifo's with the respective lvds_rx clk instead of 125 global ?

    sync_fifo1 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
        DATA_WIDTH  => 33,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            => i_hit_in1 & i_hit_ena1,
        rdclk           => i_clk156,
        rdreq           => '1',
        wrclk           => i_clk125,
        wrreq           => '1',
        q(0)            => hit_ena1,
        q(32 downto 1)  => hit_in1--,
    );

    sync_fifo2 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
        DATA_WIDTH  => 33,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            => i_hit_in2 & i_hit_ena2,
        rdclk           => i_clk156,
        rdreq           => '1',
        wrclk           => i_clk125,
        wrreq           => '1',
        q(0)            => hit_ena2,
        q(32 downto 1)  => hit_in2--,
    );

    sync_fifo3 : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2,
        DATA_WIDTH  => 33,
        SHOWAHEAD   => "OFF",
        OVERFLOW    => "ON",
        DEVICE      => "Arria V"--,
    )
    port map(
        aclr            => '0',
        data            => i_hit_in3 & i_hit_ena3,
        rdclk           => i_clk156,
        rdreq           => '1',
        wrclk           => i_clk125,
        wrreq           => '1',
        q(0)            => hit_ena3,
        q(32 downto 1)  => hit_in3--,
    );


process(i_clk156, i_reset_n)
begin
if(i_reset_n = '0') then
    o_hit_ena       <= '0';
    ena             <= (others => '0');
    ena_del1        <= (others => '0');
    ena_del2        <= (others => '0');
    ena_del1_nors   <= (others => '0');
    ena_del2_nors   <= (others => '0');
elsif(i_clk156'event and i_clk156 = '1') then

    ena             <= ((hit_ena3 & hit_ena2 & hit_ena1) and (not ena)) and (not ena_del1_nors);
    ena_del1        <= ena;
    ena_del2        <= ena_del1;

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
        o_hit_out   <= hit1;
        o_hit_ena   <= '1';
    elsif (ena_del2(1) = '1') then
        o_hit_out   <= hit2;
        o_hit_ena   <= '1';
    elsif (ena_del2(2) = '1') then
        o_hit_out   <= hit3;
        o_hit_ena   <= '1';
    elsif(ena_del1(0) = '1') then
        o_hit_out   <= hit1;
        o_hit_ena   <= '1';
        ena_del2(0) <= '0';
    elsif (ena_del1(1) = '1') then
        o_hit_out   <= hit2;
        o_hit_ena   <= '1';
        ena_del2(1) <= '0';
    elsif (ena_del1(2) = '1') then
        o_hit_out   <= hit3;
        o_hit_ena   <= '1';
        ena_del2(2) <= '0';
    elsif(ena(0) = '1') then
        o_hit_out   <= hit1;
        o_hit_ena   <= '1';
        ena_del1(0) <= '0';
    elsif (ena(1) = '1') then
        o_hit_out   <= hit2;
        o_hit_ena   <= '1';
        ena_del1(1) <= '0';
    elsif (ena(2) = '1') then
        o_hit_out   <= hit3;
        o_hit_ena   <= '1';
        ena_del1(2) <= '0';
    else
        o_hit_ena   <= '0';
    end if;

end if;
end process;

end rtl;