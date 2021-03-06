----------------------------------------------------------------------------
-- Slow Control Main Unit for Switching Board
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-- Marius Koeppel, Mainz University
-- makoeppe@students.uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity swb_sc_main is
generic (
    NLINKS : positive := 4
);
port (
    i_clk           : in    std_logic;
    i_reset_n       : in    std_logic;
    i_length_we     : in    std_logic;
    i_length        : in    std_logic_vector(15 downto 0);
    i_mem_data      : in    std_logic_vector(31 downto 0);
    o_mem_addr      : out   std_logic_vector(15 downto 0);
    o_mem_data      : out   work.mu3e.link_array_t(NLINKS-1 downto 0);
    o_done          : out   std_logic;
    o_state         : out   std_logic_vector(27 downto 0)
);
end entity;

architecture arch of swb_sc_main is

    signal addr_reg    : std_logic_vector(15 downto 0) := (others => '0');
    signal wren_reg    : std_logic_vector(15 downto 0);--)(NLINKS-1 downto 0);

    type state_type is (idle, read_fpga_id, read_data);
    signal state : state_type;

    signal length_we, length_we_reg : std_logic;
    signal wait_cnt : std_logic_vector(2 downto 0);
    signal mem_data  : work.mu3e.link_t;
    signal mask_addr, length_s, cur_length : std_logic_vector(15 downto 0);
    signal mem_data_reg : std_logic_vector(31 downto 0);

begin

    o_mem_addr <= addr_reg;

    process(i_clk, i_reset_n)
    begin
        if ( i_reset_n = '0' ) then
            length_we       <= '0';
            length_we_reg   <= '0';
        elsif ( rising_edge(i_clk) ) then
            length_we       <= '0';
            length_we_reg   <= i_length_we;
            if ( i_length_we = '1' and length_we_reg = '0' ) then
                length_we <= '1';
            end if;
        end if;
    end process;

    gen_output:
    for I in 0 to NLINKS-1 generate
        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n = '0' ) then
            o_mem_data(I)       <= work.mu3e.LINK_IDLE;
        elsif ( rising_edge(i_clk) ) then
            if ( wren_reg(I) = '1' ) then
                o_mem_data(I)   <= mem_data;
            else
                o_mem_data(I)   <= work.mu3e.LINK_IDLE;
            end if;
        end if;
        end process;
    end generate;

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        addr_reg    <= (others => '0');
        wren_reg    <= (others => '0');
        state       <= idle;
        o_done      <= '0';
        o_state     <= (others => '0');
        wait_cnt    <= (others => '0');
        mem_data    <= work.mu3e.LINK_ZERO;
        length_s    <= (others => '0');
        cur_length  <= (others => '0');
        mem_data_reg<= (others => '0');
        --
    elsif ( rising_edge(i_clk) ) then
        wait_cnt        <= wait_cnt + '1';
        mem_data_reg    <= i_mem_data;
        mem_data.data   <= mem_data_reg;
        wren_reg        <= (others => '0');

        -- reset link
        mem_data.datak  <= "0000";
        mem_data.t0     <= '0';
        mem_data.t1     <= '0';
        mem_data.err    <= '0';
        mem_data.eop    <= '0';
        mem_data.sbhdr  <= '0';
        mem_data.dthdr  <= '0';
        mem_data.sop    <= '0';
        mem_data.idle   <= '0';

        if ( addr_reg = x"FFFF" ) then
            addr_reg <= (others => '0');
        end if;

        case state is
        when idle =>
            o_state         <= x"0000001";
            o_done          <= '1';
            if ( length_we = '1' ) then
                state       <= read_fpga_id;
                length_s    <= i_length;
                o_done      <= '0';
                wait_cnt    <= (others => '0');
            end if;
            --
        when read_fpga_id =>
            o_state <= x"0000002";
            if ( wait_cnt = "111" ) then
                if ( mem_data_reg(7 downto 0) = x"BC" ) then
                    state           <= read_data;
                    addr_reg        <= addr_reg + '1';
                    cur_length      <= cur_length + '1';
                    mem_data.datak  <= "0001";
                    mask_addr       <= mem_data_reg(23 downto 8); -- get fpga id if x"FFFF" write to all links, if 1 first link and so on
                    if ( mem_data_reg(23 downto 8) = x"FFFF" ) then
                        wren_reg    <= (others => '1');
                    else
                        wren_reg    <= mem_data_reg(23 downto 8); -- todo fix me to more the 16 addr one hot
                    end if;
                end if;
            end if;
            --
        when read_data =>
            o_state <= x"0000003";
            if ( wait_cnt = "111" ) then
                if(mask_addr(15 downto 0) = x"FFFF") then
                    wren_reg    <= (others => '1');
                else
                    wren_reg    <= mask_addr(15 downto 0);
                end if;
                if (length_s + '1' = cur_length ) then
                    mem_data.datak  <= "0001";
                    state           <= idle;
                    wait_cnt        <= (others => '0');
                    addr_reg        <= (others => '0');
                    length_s        <= (others => '0');
                    cur_length      <= (others => '0');
                else
                    cur_length  <= cur_length + '1';
                    addr_reg    <= addr_reg + '1';
                end if;
            end if;
            --
        when others =>
            state       <= idle;
            addr_reg    <= (others => '0');
            wren_reg    <= (others => '0');
            cur_length  <= (others => '0');
            length_s    <= (others => '0');
            --
        end case;

    end if;
    end process;

end architecture;
