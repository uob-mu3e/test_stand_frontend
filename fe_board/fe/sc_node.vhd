----------------------------------------------------------------------------
--
-- Slow Control Node
-- Oktober 2021, M.Mueller
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.mudaq.all;

entity sc_node is
generic (
    -- todo: generic how many slaves, max.4 ; (an 4 nichts anschliessen etc.)
    ADD_SLAVE0_DELAY_g      : positive := 1; -- Delay to introduce for i_slave0_rdata
    ADD_SLAVE1_DELAY_g      : positive := 1; -- Delay to introduce for i_slave1_rdata
    N_REPLY_CYCLES_g        : positive := 2; -- cycles between i_master_re and arrival of o_master_rdata
    SLAVE0_ADDR_MATCH_g     : std_ulogic_vector(15 downto 0) := "----------------"--;
        -- Pattern to match with i_master_addr in order to connect re/we to slave0 ("-" is don't care)
        -- connects to slave1 if no match
        -- Default "--------" connects to slave0
);
port (
    i_clk           : in    std_logic;
    i_reset_n       : in    std_logic;

    i_master_addr   : in    std_logic_vector(15 downto 0);
    i_master_re     : in    std_logic;
    o_master_rdata  : out   reg32;
    i_master_we     : in    std_logic;
    i_master_wdata  : in    reg32;

    o_slave0_addr   : out   std_logic_vector(15 downto 0);
    o_slave0_re     : out   std_logic;
    i_slave0_rdata  : in    reg32;
    o_slave0_we     : out   std_logic;
    o_slave0_wdata  : out   reg32;

    o_slave1_addr   : out   std_logic_vector(15 downto 0);
    o_slave1_re     : out   std_logic;
    i_slave1_rdata  : in    reg32;
    o_slave1_we     : out   std_logic;
    o_slave1_wdata  : out   reg32--;
);
end entity;

architecture arch of sc_node is
    signal s0_return_queue      : reg32array(ADD_SLAVE0_DELAY_g downto 0);
    signal s1_return_queue      : reg32array(ADD_SLAVE1_DELAY_g downto 0);
    signal slave0_re            : std_logic;
    signal slave1_re            : std_logic;

    -- in this vector 0: slave0, 1: slave1
    signal return_queue_S01_switch : std_logic_vector(N_REPLY_CYCLES_g downto 0);

begin
    assert ( ADD_SLAVE0_DELAY_g <= N_REPLY_CYCLES_g ) report "sc_node Delay mismatch, N_REPLY_CYCLES_g is not allowed to be smaller than ADD_SLAVE0_DELAY_g" severity error;
    assert ( ADD_SLAVE1_DELAY_g <= N_REPLY_CYCLES_g ) report "sc_node Delay mismatch, N_REPLY_CYCLES_g is not allowed to be smaller than ADD_SLAVE1_DELAY_g" severity error;

    -- return part ------------------------------------
    o_slave0_re <= slave0_re;
    o_slave1_re <= slave1_re;

    return_queue_S01_switch(N_REPLY_CYCLES_g) <= slave1_re;

    s0_return_queue(ADD_SLAVE0_DELAY_g) <= i_slave0_rdata; -- move delay to request part (delay re, we, addr)
    s1_return_queue(ADD_SLAVE1_DELAY_g) <= i_slave1_rdata;

    o_master_rdata           <= s1_return_queue(0) when return_queue_S01_switch(0)='1' else s0_return_queue(0);

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        return_queue_S01_switch(N_REPLY_CYCLES_g-1 downto 0) <= (others => '0');
    elsif rising_edge(i_clk) then
        s0_return_queue(ADD_SLAVE0_DELAY_g-1 downto 0) <= s0_return_queue(ADD_SLAVE0_DELAY_g downto 1);
        s1_return_queue(ADD_SLAVE1_DELAY_g-1 downto 0) <= s1_return_queue(ADD_SLAVE1_DELAY_g downto 1);
        return_queue_S01_switch(N_REPLY_CYCLES_g-1 downto 0) <= return_queue_S01_switch(N_REPLY_CYCLES_g downto 1);
    end if;
    end process;

    -- request part -----------------------------------
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        slave0_re   <= '0';
        slave1_re   <= '0';
        o_slave0_we <= '0';
        o_slave1_we <= '0';
    elsif rising_edge(i_clk) then

        slave0_re   <= '0';
        slave1_re   <= '0';
        o_slave0_we <= '0';
        o_slave1_we <= '0';

        o_slave0_addr   <= i_master_addr;
        o_slave1_addr   <= i_master_addr;

        o_slave0_wdata  <= i_master_wdata;
        o_slave1_wdata  <= i_master_wdata;

        if( to_stdulogicvector(i_master_addr) ?= SLAVE0_ADDR_MATCH_g) then
            slave0_re   <= i_master_re;
            o_slave0_we <= i_master_we;
        else
            slave1_re   <= i_master_re;
            o_slave1_we <= i_master_we;
        end if;
    end if;
    end process;

end architecture;
