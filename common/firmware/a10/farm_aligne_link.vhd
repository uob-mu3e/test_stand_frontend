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
    N : positive :=  8;
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    i_rx            : in work.util.slv34_array_t(N - 1 downto 0);
    i_sop           : in std_logic_vector(N - 1 downto 0);
    i_eop           : in std_logic_vector(N - 1 downto 0);

    --! error counters 
    --! 0: fifo sync_almost_full
    --! 1: fifo sync_wrfull
    --! 2: cnt_skip_sub_header
    --! 3: cnt_sub_header
    o_counter       : out work.util.slv32_array_t(3 downto 0);
    o_data          : out std_logic_vector(N * 32 + 1 downto 0);
    o_empty         : out std_logic;
    i_ren           : in  std_logic;
    
    i_empty         : in  std_logic_vector(N - 1 downto 0);
    o_ren           : out std_logic_vector(N - 1 downto 0);
    
    o_error         : out std_logic;

    i_reset_n_250   : in std_logic;
    i_clk_250       : in std_logic--;
);
end entity;

architecture arch of farm_aligne_link is

    type link_to_fifo_type is (idle, write_data, skip_hits_of_sub_header, error_state);
    signal link_to_fifo_state : link_to_fifo_type;
    signal cnt_skip_sub_header, cnt_sub_header : std_logic_vector(31 downto 0);

    signal f_data : std_logic_vector(N * 32 + 1 downto 0); -- 32b x links plus 2 bits for sop, eop
    signal f_almost_full, f_wrfull, f_wen : std_logic;
    signal f_wrusedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH - 1 downto 0);
    constant check_zeros : std_logic_vector(N - 1 downto 0) := (others => '0');
    constant check_ones : std_logic_vector(N - 1 downto 0) := (others => '1');

begin

    e_cnt_link_fifo_almost_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(0), i_ena => f_almost_full, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    e_cnt_dc_link_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(1), i_ena => f_wrfull, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    o_counter(2) <= cnt_skip_sub_header;
    o_counter(3) <= cnt_sub_header;
    
    gen_link_to_fifo : FOR i in 0 to N - 1 GENERATE
        o_ren(i) <=  '1' when link_to_fifo_state  = idle and i_sop(i) = '0' and i_empty(i) = '0' else
                     '1' when link_to_fifo_state /= idle and i_empty = check_zeros else
                     '0';
    END GENERATE;
    
    --! sync link data to pcie clk and buffer events
    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 /= '1' ) then
        f_data              <= (others => '0');
        f_wen               <= '0';
        o_error             <= '0';
        cnt_skip_sub_header <= (others => '0');
        cnt_sub_header      <= (others => '0');
        link_to_fifo_state  <= idle;
        --
    elsif ( rising_edge(i_clk_250) ) then
    
        f_wen <= '0';
        
        for I in 0 to N - 1 loop
            f_data(I * 32 + 31 downto I * 32) <= i_rx(I)(31 downto 0);
        end loop;

        case link_to_fifo_state is 

            when idle =>
                if ( i_sop = check_ones and i_empty = check_zeros ) then
                    if ( f_wrfull = '1' ) then
                        -- not good stop run state
                        link_to_fifo_state  <= error_state;
                    elsif ( f_almost_full = '1' ) then
                        -- skip hits of sub header and set overflow
                        link_to_fifo_state  <= skip_hits_of_sub_header;
                        cnt_skip_sub_header <= cnt_skip_sub_header + '1';
                    else
                        link_to_fifo_state  <= write_data;
                        cnt_sub_header      <= cnt_sub_header + '1';
                    end if;
                -- if we come from skip_hits_of_sub_header we need to check also for eop
                elsif ( i_eop = check_ones and i_empty = check_zeros ) then
                    f_data(N * 32 + 1 downto N * 32) <= "10"; -- trailer
                    f_wen <= '1';
                end if;

            when write_data =>
                if ( i_sop = check_ones and i_empty = check_zeros ) then
                    f_data(N * 32 + 1 downto N * 32) <= "01"; -- header
                    f_wen               <= '1';
                elsif ( i_eop = check_ones and i_empty = check_zeros ) then
                    link_to_fifo_state <= idle;
                    f_data(N * 32 + 1 downto N * 32) <= "10"; -- trailer
                    f_wen <= '1';
                elsif ( i_empty = check_zeros ) then
                    f_data(N * 32 + 1 downto N * 32) <= "00"; -- data
                    f_wen <= '1';
                end if;

            when skip_hits_of_sub_header =>
                -- skip the hits here and write overflow
                if ( i_sop = check_ones and i_empty = check_zeros ) then
                    link_to_fifo_state <= idle;
                    f_data(3 * 32 + 31 downto 32 * 3 + 8) <= x"FFFFFF";
                    f_data(4 * 32 + 31 downto 32 * 4 + 8) <= x"FFFFFF";
                    f_data(N * 32 + 1 downto N * 32) <= "01"; -- header
                    f_wen <= '1';
                elsif ( i_eop = check_ones and i_empty = check_zeros ) then
                    link_to_fifo_state <= idle;
                    f_data(N * 32 + 1 downto N * 32) <= "10"; -- trailer
                    f_wen <= '1';
                end if;
                
            when error_state =>
                if ( f_wrfull = '0' ) then
                    link_to_fifo_state <= idle;
                    f_data(N * 32 + 1 downto N * 32) <= "11"; -- error
                    o_error <= '1';
                    f_wen <= '1';
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
        DATA_WIDTH  => N * 32 + 2,
        DEVICE      => "Arria 10"--,
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
