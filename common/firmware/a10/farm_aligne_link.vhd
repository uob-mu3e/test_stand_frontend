-------------------------------------------------------
--! farm_aligne_link.vhd
--! @brief the @farm_aligne_link syncs the link data
--! to the pcie clk
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity farm_aligne_link is
generic (
    g_NLINKS_SWB_TOTL : positive :=  8;
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    i_rx            : in  std_logic_vector(35 downto 0);
    i_sop           : in  std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    i_sop_cur       : in  std_logic;
    i_eop           : in  std_logic;
    i_skip          : in  std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    o_skip          : out std_logic;

    i_empty         : in  std_logic_vector(g_NLINKS_SWB_TOTL-1 downto 0);
    i_empty_cur     : in  std_logic;
    o_ren           : out std_logic;

    o_tx            : out std_logic_vector(31 downto 0);
    o_tx_k          : out std_logic_vector(3 downto 0);

    --! error counters
    --! 0: fifo sync_almost_full
    --! 1: fifo sync_wrfull
    --! 2: # of next farm event
    --! 3: cnt_sub_header
    o_counter       : out work.util.slv32_array_t(3 downto 0);
    o_data          : out std_logic_vector(35 downto 0);
    o_empty         : out std_logic;
    i_ren           : in  std_logic;

    o_error         : out std_logic;

    i_reset_n_250   : in std_logic;
    i_clk_250       : in std_logic--;
);
end entity;

architecture arch of farm_aligne_link is

    type link_to_fifo_type is (idle, write_data, skip_event);
    signal link_to_fifo_state : link_to_fifo_type;
    signal cnt_skip_event, cnt_event : std_logic_vector(31 downto 0);

    signal f_data : std_logic_vector(35 downto 0);
    signal f_almost_full, f_wrfull, f_wen : std_logic;
    signal f_wrusedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH - 1 downto 0);

begin

    e_cnt_link_fifo_almost_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(0), i_ena => f_almost_full, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    e_cnt_dc_link_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(1), i_ena => f_wrfull, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    o_counter(2) <= cnt_skip_event;
    o_counter(3) <= cnt_event;

    o_ren <= '1' when link_to_fifo_state  = idle and i_sop = check_ones and work.util.or_reduce(i_empty) = '0' else
             '1' when link_to_fifo_state /= idle and i_empty_cur = '0' else
             '0';

    -- we skip the events which are already tagged and if fifo is almost_full = '1'
    o_skip <= '1' when link_to_fifo_state = idle and f_almost_full = '1' else
              '1' when link_to_fifo_state = idle and i_sop_cur(i) = '1' and i_rx(25) = '1' else
              '0';

    --! sync link data to pcie clk and buffer events
    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 /= '1' ) then
        f_data              <= (others => '0');
        f_wen               <= '0';
        o_error             <= '0';
        cnt_skip_event      <= (others => '0');
        cnt_event           <= (others => '0');
        link_to_fifo_state  <= idle;
        o_tx                <= x"000000BC";
        o_tx_k              <= x"0001";
        --
    elsif ( rising_edge(i_clk_250) ) then

        f_wen   <= '0';
        f_data  <= i_rx;

        o_tx    <= x"000000BC";
        o_tx_k  <= x"0001";

        case link_to_fifo_state is

        when idle =>
            -- TODO: also check if all are in state idle here
            if ( i_sop = check_ones and work.util.or_reduce(i_empty) = '0' ) then
                o_tx   <= i_rx;
                o_tx_k <= i_rx_k;
                if ( f_wrfull = '1' ) then
                    -- TODO: not good stop run state
                elsif ( f_almost_full = '1' or work.util.or_reduce(i_skip) = '1' ) then
                    -- skip hits of sub header and set overflow
                    link_to_fifo_state  <= skip_event;
                    cnt_skip_event      <= cnt_skip_event + '1';
                else
                    -- tag event
                    o_tx(25)            <= '1';
                    f_wen               <= '1';
                    link_to_fifo_state  <= write_data;
                    cnt_event           <= cnt_event + '1';
                end if;
            end if;

        when write_data =>
            -- write the event
            if ( i_empty_cur = '0' ) then
                o_tx   <= i_rx;
                o_tx_k <= i_rx_k;
                f_wen  <= '1';
                if ( i_eop = '1' ) then
                    link_to_fifo_state <= idle;
                end if;
            end if;

        when skip_event =>
            -- skip the event
            if ( i_empty_cur = '0' ) then
                o_tx   <= i_rx;
                o_tx_k <= i_rx_k;
                if ( i_eop = '1' ) then
                    link_to_fifo_state <= idle;
                end if;
            end if;

        when others =>
            link_to_fifo_state <= idle;

        end case;
    --
    end if;
    end process;

    e_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH  => 35--,
    )
    port map (
        data        => f_data,
        wrreq       => f_wen,
        rdreq       => i_ren,
        wrclk       => i_clk_250,
        rdclk       => i_clk_250,
        q           => o_data,
        rdempty     => o_empty,
        rdusedw     => open,
        wrfull      => f_wrfull,
        wrusedw     => f_wrusedw,
        aclr        => not i_reset_n_250--,
    );

    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 = '0' ) then
        f_almost_full       <= '0';
    elsif ( rising_edge(i_clk_250) ) then
        if ( f_wrusedw(LINK_FIFO_ADDR_WIDTH - 1) = '1' ) then
            f_almost_full <= '1';
        else
            f_almost_full <= '0';
        end if;
    end if;
    end process;

end architecture;
