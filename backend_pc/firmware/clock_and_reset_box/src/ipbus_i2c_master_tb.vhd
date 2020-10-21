-- ipbus_i2c_master_tb
--
-- Test benach for adapted I2C master
--
-- Niklaus Berger, July 2020

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
use work.ipbus.all;


entity ipbus_i2c_master_tb is
end entity;

architecture rtl of ipbus_i2c_master_tb is

   component ipbus_i2c_master is
        generic(addr_width: natural := 0);
        port(
            clk: in std_logic;
            rst: in std_logic;
            ipbus_in: in ipb_wbus;
            ipbus_out: out ipb_rbus;
            ipbus_in_fast: in ipb_wbus;
            ipbus_out_fast: out ipb_rbus;
            ipbus_in_mem: in ipb_wbus;
            ipbus_out_mem: out ipb_rbus;       
            scl_o: out std_logic;
            scl_i: in std_logic;
            sda_o: out std_logic;
            sda_i: in std_logic
        );
   end component;    

signal clk : std_logic;
signal rst : std_logic;
signal ipbus_in : ipb_wbus;
signal ipbus_out : ipb_rbus;
signal ipbus_in_fast : ipb_wbus;
signal ipbus_out_fast : ipb_rbus;
signal ipbus_in_mem : ipb_wbus;
signal ipbus_out_mem : ipb_rbus;
signal scl_o: std_logic;
signal scl_i: std_logic;
signal sda_o: std_logic;
signal sda_i: std_logic;

begin

    dut:ipbus_i2c_master
        port map(
            clk => clk,
            rst => rst,
            ipbus_in => ipbus_in,
            ipbus_out => ipbus_out,
            ipbus_in_fast => ipbus_in_fast,
            ipbus_out_fast => ipbus_out_fast,
            ipbus_in_mem  => ipbus_in_mem,
            ipbus_out_mem => ipbus_out_mem,  
            scl_o   => scl_o,
            scl_i   => scl_i,
            sda_o   => sda_o,
            sda_i   => sda_i
        );

sda_i <= '1'; 
scl_i <= '1';

clkgen: process
begin
    clk <= '0';
    wait for 5 ns;
    clk <= '1';
    wait for 5 ns;
end process clkgen;        


resetgen: process
begin
    rst <= '1';
    wait for 50 ns;
    rst <= '0';
    wait;
end process resetgen;    

inputgen: process
begin
    ipbus_in.ipb_strobe         <= '0';
    ipbus_in_fast.ipb_strobe    <= '0';
    ipbus_in_mem.ipb_strobe     <= '0';
    ipbus_in.ipb_write          <= '0';   
    ipbus_in_fast.ipb_write     <= '0';
    wait for 100 ns;
    -- set prescale register
    ipbus_in.ipb_strobe         <= '1';
    ipbus_in.ipb_write          <= '1';
    ipbus_in.ipb_addr      <= X"00000000";
    ipbus_in.ipb_wdata     <= X"00000004";
    wait for 10 ns;
    ipbus_in.ipb_strobe         <= '0';
    ipbus_in.ipb_write          <= '0';
    ipbus_in.ipb_addr      <= X"00000000";
    ipbus_in.ipb_wdata     <= X"00000000";
    wait for 30 ns;
    -- set prescale register// upper part
    ipbus_in.ipb_strobe         <= '1';
    ipbus_in.ipb_write          <= '1';
    ipbus_in.ipb_addr      <= X"00000001";
    ipbus_in.ipb_wdata     <= X"00000000";
    wait for 10 ns;
    ipbus_in.ipb_strobe         <= '0';
    ipbus_in.ipb_write          <= '0';
    ipbus_in.ipb_addr      <= X"00000000";
    ipbus_in.ipb_wdata     <= X"00000000";
    wait for 30 ns;
    -- enable the i2c core
    ipbus_in.ipb_strobe         <= '1';
    ipbus_in.ipb_write          <= '1';
    ipbus_in.ipb_addr      <= X"00000002";
    ipbus_in.ipb_wdata     <= X"00000080";
    wait for 10 ns;
    ipbus_in.ipb_strobe         <= '0';
    ipbus_in.ipb_write          <= '0';
    ipbus_in.ipb_addr      <= X"00000000";
    ipbus_in.ipb_wdata     <= X"00000000";
    wait for 40 ns;
    ipbus_in_fast.ipb_strobe    <= '1';
    ipbus_in_fast.ipb_write     <= '1';
    ipbus_in_fast.ipb_addr      <= X"A0B0C0D0";
    ipbus_in_fast.ipb_wdata     <= X"CCCCCCCC";
    wait for 10 ns;
    ipbus_in_fast.ipb_strobe    <= '0';
    wait;
end process inputgen;

end architecture rtl;

