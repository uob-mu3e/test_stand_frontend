-------------------------------------------------------
--! @farm_link_to_fifo.vhd
--! @brief the farm_link_to_fifo sorts out the data from the 
--! link and provides it as a fifo output
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity farm_link_to_fifo is
generic (
    g_NLINKS_SWB_TOTL    : positive :=  16,
    N_PIXEL              : positive :=  8,
    N_SCIFI              : positive :=  4,
    N_TILE               : positive :=  4--,
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    i_rx            : in  work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    i_rx_k          : in  work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    
    o_data_pixel    : out std_logic_vector(N_PIXEL * 36 - 1 downto 0);
    o_data_scifi    : out std_logic_vector(N_SCIFI * 36 - 1 downto 0);
    o_data_tile     : out std_logic_vector(N_TILE  * 36 - 1 downto 0);
    i_ren           : in  std_logic;
    o_rdempty       : out std_logic;

    --! error counters 
    --! 0: fifo f_almost_full
    --! 1: fifo f_wrfull
    --! 2: # of skip event
    o_counter       : out work.util.slv32_array_t(2 downto 0);

    i_reset_n_250   : in std_logic;
    i_clk_250       : in std_logic--;
);
end entity;

architecture arch of farm_link_to_fifo is

    constant check_ones   : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0) := (others => '1');
    constant check_zeros  : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0) := (others => '0');
   
    type link_to_fifo_type is (idle, write_data, skip_data);
    signal link_to_fifo_state : link_to_fifo_type;
    signal cnt_skip_event : std_logic_vector(31 downto 0);

    signal rx_data, rx_q : work.util.slv36_array_t(g_NLINKS_SWB_TOTL - 1 downto 0);
    signal rx_wen, sync_rdempty, sync_ren, sop, eop : std_logic_vector(g_NLINKS_SWB_TOTL - 1 downto 0);

    signal f_data, f_q : std_logic_vector(g_NLINKS_SWB_TOTL * 36 - 1 downto 0);
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

    o_data_pixel <= f_q(N_PIXEL * 36 - 1 downto 0);
    o_data_scifi <= f_q((N_SCIFI + N_PIXEL) * 36 - 1 downto N_PIXEL * 36);
    o_data_tile  <= f_q((N_SCIFI + N_PIXEL + N_TILE) * 36 - 1 downto (N_SCIFI + N_PIXEL) * 36);

    --! buffer link data and sortout x"BC"
    gen_link_to_fifo : FOR i in 0 to g_NLINKS_SWB_TOTL - 1 GENERATE
        
        process(i_clk_250, i_reset_n_250)
        begin
            if ( i_reset_n_250 = '0' ) then
                rx_data(i)  <= (others => '0');
                rx_wen(i)   <= '0';
            elsif ( rising_edge(i_clk_250) ) then
                rx_data(i) <= i_rx(i) & i_rx_k(i);
                if ( i_rx(i) = x"000000BC" and i_rx_k(i) = "0001" ) then
                    rx_wen(i) <= '0';
                else
                    rx_wen(i) <= '1';
                end if;
            end if;
        end process;
            
        e_sync_fifo : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => LINK_FIFO_ADDR_WIDTH,
            DATA_WIDTH  => 36,
            DEVICE      => "Arria 10"--,
        )
        port map (
            data        => rx_data(i),
            wrreq       => rx_wen(i),
            rdreq       => sync_ren(i),
            wrclk       => i_clk_250,
            rdclk       => i_clk_250,
            q           => rx_q(i),
            rdempty     => sync_rdempty(i),
            aclr        => not i_reset_n_250--,
        );

        sop(i)      <= '1' when rx_q(i)(11 downto 4) = x"7C" and rx_q(i)(3 downto 0) = "0001" else
                       '0';

        eop(i)      <= '1' when rx_q(i)(11 downto 4) = K28_4 and rx_q(i)(3 downto 0) = "0001" else
                       '0';

        sync_ren(i) <= '1' when link_to_fifo_state  = idle and sop(i) = '0' and sync_rdempty(i) = '0' else
                       '1' when link_to_fifo_state /= idle and eop(i) = '0' and sync_rdempty    = check_zeros else
                       '0';

    END GENERATE;

    --! Aligne Link data
    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 /= '1' ) then
        f_data              <= (others => '0');
        f_wen              <= '0';
        cnt_skip_event      <= (others => '0');
        link_to_fifo_state  <= idle;
        --
    elsif rising_edge(i_clk_250) then

        f_wen <= '0';
        for I in 0 to g_NLINKS_SWB_TOTL - 1 loop
            f_data(36 * I + 35 downto 36 * I) <= rx_q(I);
        end loop;
                
        case link_to_fifo_state is 

            when idle =>
                if ( sop = check_ones ) then
                    if ( f_almost_full = '1' ) then
                        link_to_fifo_state  <= skip_data;
                        cnt_skip_event      <= cnt_skip_event + '1';
                    else
                        link_to_fifo_state  <= write_data;
                        f_wen               <= '1';
                    end if;
                end if;

            when write_data =>
                if ( eop = check_ones ) then
                    link_to_fifo_state <= idle;
                end if;

                if ( sync_rdempty = check_zeros ) then
                    f_wen <= '1';
                end if;

            when skip_data =>
                if ( eop = check_ones ) then
                    link_to_fifo_state <= idle;
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
        DATA_WIDTH  => g_NLINKS_SWB_TOTL * 36,
        DEVICE      => "Arria 10"--,
    )
    port map (
        data        => f_data,
        wrreq       => f_wen,
        rdreq       => i_ren,
        wrclk       => i_clk_250,
        rdclk       => i_clk_250,
        q           => f_q,
        rdempty     => o_rdempty,
        rdusedw     => open,
        wrfull      => f_wrfull,
        wrusedw     => f_wrusedw,
        aclr        => not i_reset_n_250--,
    );

    process(i_clk_250, i_reset_n_250)
    begin
        if(i_reset_n_250 = '0') then
            f_almost_full       <= '0';
        elsif(rising_edge(i_clk_250)) then
            if ( f_wrusedw(LINK_FIFO_ADDR_WIDTH - 1) = '1' ) then
                f_almost_full <= '1';
            else 
                f_almost_full <= '0';
            end if;
        end if;
    end process;

end architecture;
