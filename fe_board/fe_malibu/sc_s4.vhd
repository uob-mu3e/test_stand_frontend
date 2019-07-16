----------------------------------------------------------------------------
--
-- Slow Control Unit
-- Marius Koeppel, Mainz University
-- makoeppe@students.uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sc_s4 is
port (
    i_link_data     : in    std_logic_vector(31 downto 0);
    i_link_datak    : in    std_logic_vector(3 downto 0);

    o_fifo_we       : out   std_logic;
    o_fifo_wdata    : out   std_logic_vector(35 downto 0);

    o_ram_addr      : out   std_logic_vector(31 downto 0);
    o_ram_re        : out   std_logic;
    i_ram_rdata     : in    std_logic_vector(31 downto 0);
    i_ram_rvalid    : in    std_logic;
    o_ram_we        : out   std_logic;
    o_ram_wdata     : out   std_logic_vector(31 downto 0);

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic
);
end entity;

architecture arch of sc_s4 is

    type state_t is (
        S_IDLE, S_ADDR, S_LENGTH, S_READ, S_WRITE,
        S_TIMEOUT, S_ERROR--,
    );
    signal state : state_t;

    signal ram_addr, ram_addr_end : unsigned(31 downto 0);
    signal ram_read_nreq : unsigned(7 downto 0);

    signal idle_counter : unsigned(15 downto 0);
    signal sc_type : std_logic_vector(1 downto 0);

begin

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        state <= S_IDLE;

        o_fifo_we <= '0';
        o_ram_re <= '0';
        o_ram_we <= '0';

        ram_read_nreq <= (others => '0');
        idle_counter <= (others => '0');
        --
    elsif rising_edge(i_clk) then
        o_fifo_we <= '0';
        o_ram_re <= '0';
        o_ram_we <= '0';

        if ( i_link_data(7 downto 0) = x"BC" and i_link_datak = "0001"
            and i_link_data(31 downto 26) = "000111"
        ) then
            -- preamble
            sc_type <= i_link_data(25 downto 24);
--            fpga_id <= i_link_data(23 downto 8);
            state <= S_ADDR;
            --
        elsif ( i_link_data = x"0000009C" and i_link_datak = "0001"
            and ( state /= S_IDLE )
        ) then
            -- trailer
            -- write end of packet
            o_fifo_we <= '1';
            o_fifo_wdata <= "0011" & X"00000000";
            state <= S_IDLE;
            --
        elsif ( state = S_ADDR and i_link_datak = "0000" ) then
            -- ack addr
            o_fifo_we <= '1';
            o_fifo_wdata <= sc_type & "10" & i_link_data;

            ram_addr <= unsigned(i_link_data);

            state <= S_LENGTH;
            --
        elsif ( state = S_LENGTH and i_link_datak = "0000" ) then
            -- ack length
            o_fifo_we <= '1';
            o_fifo_wdata <= "0000" & X"0001" & i_link_data(15 downto 0);

            ram_addr_end <= ram_addr + unsigned(i_link_data(15 downto 0));

            if (sc_type = "10") then
                state <= S_READ;
            elsif (sc_type = "11") then
                state <= S_WRITE;
            end if;
            --
        elsif ( state = S_WRITE and i_link_datak = "0000" ) then
            if ( ram_addr /= ram_addr_end ) then
                -- write to ram
                o_ram_addr <= std_logic_vector(ram_addr);
                o_ram_we <= '1';
                o_ram_wdata <= i_link_data;
                ram_addr <= ram_addr + 1;
            end if;

            if ( ram_addr = ram_addr_end ) then
                -- write end of packet
                o_fifo_we <= '1';
                o_fifo_wdata <= "0011" & X"00000000";
                state <= S_IDLE;
            end if;
            --
        elsif ( state = S_READ ) then
            if ( ram_addr /= ram_addr_end and ram_read_nreq /= (ram_read_nreq'range => '1') ) then
                -- read from ram
                o_ram_addr <= std_logic_vector(ram_addr);
                o_ram_re <= '1';
                ram_addr <= ram_addr + 1;
                ram_read_nreq <= ram_read_nreq + 1;
            end if;

            if ( ram_read_nreq /= 0 and i_ram_rvalid = '1' ) then
                -- write to fifo
                o_fifo_we <= '1';
                o_fifo_wdata <= "0000" & i_ram_rdata;
                if ( ram_addr /= ram_addr_end and ram_read_nreq /= (ram_read_nreq'range => '1') ) then
                    ram_read_nreq <= ram_read_nreq;
                else
                    ram_read_nreq <= ram_read_nreq - 1;
                end if;
            end if;

            if ( ram_addr = ram_addr_end and ram_read_nreq = 0 ) then
                -- write end of packet
                o_fifo_we <= '1';
                o_fifo_wdata <= "0011" & X"00000000";
                state <= S_IDLE;
            end if;
            --
        elsif ( i_link_data = x"000000BC" and i_link_datak = "0001" ) then
            idle_counter <= idle_counter + 1;

            if ( state = S_IDLE ) then
                idle_counter <= (others => '0');
            end if;

            if ( idle_counter = (idle_counter'range => '1') ) then
                -- timeout
                state <= S_TIMEOUT;
            end if;
            --
        elsif ( state = S_IDLE ) then
            --
        else
            state <= S_ERROR;
            --
        end if;

        --
    end if;
    end process;

end architecture;
