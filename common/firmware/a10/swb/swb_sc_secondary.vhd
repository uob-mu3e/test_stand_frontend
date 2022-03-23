----------------------------------------------------------------------------
-- Slow Control Secondary Unit for Switching Board
-- Marius Koeppel, Mainz University
-- mkoeppel@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity swb_sc_secondary is
generic (
    NLINKS      : positive := 4;
    skip_init   : std_logic := '0'
);
port (
    i_link_enable               : in    std_logic_vector(NLINKS-1 downto 0);
    i_link_data                 : in    work.mu3e.link_array_t(NLINKS-1 downto 0);

    mem_data_out                : out   std_logic_vector(31 downto 0);
    mem_addr_out                : out   std_logic_vector(15 downto 0);
    mem_addr_finished_out       : out   std_logic_vector(15 downto 0);
    mem_wren                    : out   std_logic;
    stateout                    : out   std_logic_vector(3 downto 0);

    i_reset_n                   : in    std_logic;
    i_clk                       : in    std_logic--;
);
end entity;

architecture arch of swb_sc_secondary is

    signal link_data : work.mu3e.link_array_t(NLINKS-1 downto 0);
    signal rdempty, wen, wrfull, ren : std_logic_vector(NLINKS-1 downto 0);

    signal mem_data_o : std_logic_vector(31 downto 0);
    signal mem_addr_o : std_logic_vector(15 downto 0);
    signal mem_wren_o : std_logic;
    signal current_link : integer range 0 to NLINKS - 1;

    type state_type is (init, waiting, startup, starting);
    signal state : state_type;

begin

    mem_data_out <= mem_data_o;
    mem_addr_out <= mem_addr_o;
    mem_wren     <= mem_wren_o;

    gen_buffer_sc : FOR i in 0 to NLINKS - 1 GENERATE

        wen(i) <= '1' when i_link_data(i).idle = '0' else '0';

        e_fifo : entity work.link_scfifo
        generic map (
            g_ADDR_WIDTH=> 8,
            g_WREG_N    => 1,
            g_RREG_N    => 1--,
        )
        port map (
            i_wdata     => i_link_data(i),
            i_we        => wen(i),
            o_wfull     => wrfull(i),

            o_rdata     => link_data(i),
            i_rack      => ren(i),
            o_rempty    => rdempty(i),

            i_clk       => i_clk,
            i_reset_n   => i_reset_n--;
        );

    END GENERATE;

    process(i_reset_n, i_clk)
    begin
    if ( i_reset_n = '0' ) then
        mem_data_o <= (others => '0');
        mem_addr_o <= (others => '1');
        mem_addr_finished_out <= (others => '1');
        stateout <= (others => '0');
        ren <= (others => '0');
        mem_wren_o <= '0';
        current_link <= 0;
        if ( skip_init = '0' ) then
            state <= init;
        else
            state <= waiting;
        end if;

    elsif rising_edge(i_clk) then
        stateout <= (others => '0');
        mem_data_o <= (others => '0');
        ren <= (others => '0');
        mem_wren_o <= '0';
        mem_wren_o <= '0';

        case state is
        when init =>
            stateout(3 downto 0) <= x"1";
            mem_addr_o <= mem_addr_o + '1';
            mem_data_o <= (others => '0');
            mem_wren_o <= '1';
            if ( mem_addr_o = x"FFFE" ) then
                mem_addr_finished_out <= (others => '1');
                state <= waiting;
            end if;
            --
        when waiting =>
            stateout(3 downto 0) <= x"2";
            --LOOP link mux take the last one for prio
            link_mux:
            FOR i in 0 to NLINKS - 1 LOOP
                if ( i_link_enable(i) = '1' and link_data(i).sop = '1' and rdempty(i) = '0' ) then
                    state <= starting;
                    current_link <= i;
                end if;
            END LOOP;

        when startup =>
            mem_addr_o <= mem_addr_o + '1';
            mem_data_o <= link_data(current_link).data;
            mem_wren_o <= '1';
            ren(current_link) <= '1';
            state <= starting;

        when starting =>
            stateout(3 downto 0) <= x"3";
            if ( rdempty(current_link) = '0' ) then
                ren(current_link) <= '1';
                mem_addr_o <= mem_addr_o + '1';
                mem_data_o <= link_data(current_link).data;
                mem_wren_o <= '1';
                if ( link_data(current_link).eop = '1' ) then
                    mem_addr_finished_out <= mem_addr_o + '1';
                    state <= waiting;
                end if;
            end if;
            --
        when others =>
            stateout(3 downto 0) <= x"E";
            mem_data_o <= (others => '0');
            mem_wren_o <= '0';
            state <= waiting;
            --
        end case;

    end if;
    end process;

end architecture;
