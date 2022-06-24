-- zero suppression for Mupix Data frames as descibed in the scpec book (might need adaptation for scifi/tile)
-- assuming sop and eop of link_t to be set correctly, not assuming anything about subhdr of link_t (so we can insert it also before link_to_fifo entity) 
-- M. Mueller, June 2022


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity zero_suppression is
port (
    i_reset_n               : in    std_logic;
    i_clk                   : in    std_logic;

    i_ena_subh_suppression  : in    std_logic := '0';
    i_ena_head_suppression  : in    std_logic := '0';

    i_data                  : in    work.mu3e.link_t;
    o_data                  : out   work.mu3e.link_t--;
);
end entity;

architecture arch of zero_suppression is

    signal subh_suppression : std_logic;
    signal head_suppression : std_logic;

    signal data_subh_suppressed : work.mu3e.link_t;
    signal data_head_suppressed : work.mu3e.link_t;

    signal data_buffer      : work.mu3e.link_array_t(5 downto 0);
    signal next_subhdr      : work.mu3e.link_t;
    signal next_hit         : work.mu3e.link_t;
    type   subhdr_suppression_state_t is (head, ts0,ts1,d0,d1, waitingForFirstHit, waitingForMoreHits);
    signal subhdr_suppression_state   : subhdr_suppression_state_t;
    signal sending_data_buffer        : std_logic;
    signal is_subh          : std_logic;
    signal event_had_hit    : std_logic;

begin

    subh_suppression <= i_ena_subh_suppression or i_ena_head_suppression;
    head_suppression <= i_ena_head_suppression;

    is_subh <= '1' when (i_data.data(31 downto 26) = "111111" and i_data.datak = "0000") else '0';
    
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        o_data                      <= work.mu3e.LINK_IDLE;
        subhdr_suppression_state    <= head;
        event_had_hit               <= '0';
        data_buffer                 <= (others => work.mu3e.LINK_IDLE);
        data_subh_suppressed        <= work.mu3e.LINK_IDLE;
        data_head_suppressed        <= work.mu3e.LINK_IDLE;
        next_hit                    <= work.mu3e.LINK_IDLE;
        next_subhdr                 <= work.mu3e.LINK_IDLE;
        sending_data_buffer         <= '0';

    elsif rising_edge(i_clk) then
        -- default to idle
        o_data <= work.mu3e.LINK_IDLE;
        next_hit <= work.mu3e.LINK_IDLE;
        data_subh_suppressed <= work.mu3e.LINK_IDLE;
        data_head_suppressed <= work.mu3e.LINK_IDLE;

        -- assign output based on ena signals 
        if(subh_suppression = '0' and head_suppression = '0') then
            o_data <= i_data;
        elsif(head_suppression = '1') then 
            o_data <= data_head_suppressed;
        elsif(subh_suppression = '1' and head_suppression = '0') then
            o_data <= data_subh_suppressed;
        end if;

        -- -----------------------------------------------------------------
        -- generate subhdr zero-suppressed datastream data_subh_suppressed

        if(i_data.idle = '0') then -- we have a non-idle input
            case subhdr_suppression_state is
                when head => 
                    data_subh_suppressed <= i_data;
                    if(i_data.sop = '1') then
                        subhdr_suppression_state <= ts0;
                    end if;
                    -- else nonsense, but we still transmitt it
                when ts0 =>
                    subhdr_suppression_state <= ts1;
                    data_subh_suppressed <= i_data;
                when ts1 =>
                    subhdr_suppression_state <= d0;
                    data_subh_suppressed <= i_data;
                when d0 =>
                    subhdr_suppression_state <= d1;
                    data_subh_suppressed <= i_data;
                when d1 =>
                    subhdr_suppression_state <= waitingForFirstHit;
                    data_subh_suppressed <= i_data;
                when waitingForFirstHit =>
                    if(i_data.eop = '1') then 
                        data_subh_suppressed <= i_data;
                        subhdr_suppression_state <= head;
                    elsif(is_subh = '1') then  -- new next subh, skip prev. one
                        next_subhdr <= i_data;
                        subhdr_suppression_state <= waitingForFirstHit;
                    else -- --> we have a hit --> send saved subh, save hit
                        subhdr_suppression_state <= waitingForMoreHits;
                        data_subh_suppressed <= next_subhdr;
                        next_hit <= i_data;
                        event_had_hit <= '1';
                    end if;
                when waitingForMoreHits =>
                    if(i_data.eop = '1') then 
                        data_subh_suppressed <= i_data;
                        subhdr_suppression_state <= head;
                    elsif(is_subh = '1') then  -- subh with hits done --> send last hit of prev. subh
                        next_subhdr <= i_data;
                        subhdr_suppression_state <= waitingForFirstHit;
                        data_subh_suppressed <= next_hit;
                    else -- --> we have a hit --> send saved hit, save new hit
                        subhdr_suppression_state <= waitingForMoreHits;
                        data_subh_suppressed <= next_hit;
                        next_hit <= i_data;
                    end if;
                when others => 
                    subhdr_suppression_state <= head;
            end case;
        elsif(next_hit.idle = '0') then 
            data_subh_suppressed <= next_hit;
        end if;

        -- --------------------------------------------------
        -- gernerate header suppressed datastream

        if(data_buffer(0).sop='0') then 
            for I in 0 to 4 loop
                data_buffer(I) <= data_buffer(I+1);
            end loop;
            data_buffer(5) <= data_subh_suppressed;
            data_head_suppressed <= data_buffer(0);
            event_had_hit <= '0';
        else 
            -- start of packet waiting at the end of data_buffer
            -- wait for eop or hit

            if(data_subh_suppressed.eop = '1') then -- eop --> throw everything away 
                data_buffer <= (others => work.mu3e.LINK_IDLE);
            end if;

            if(event_had_hit = '1') then -- hit --> start sending again until the next sop is at the end of data_buffer
                for I in 0 to 4 loop
                    data_buffer(I) <= data_buffer(I+1);
                end loop;
                data_buffer(5) <= data_subh_suppressed;
                data_head_suppressed <= data_buffer(0);
            end if; 
            
        end if;


    end if;
    end process;

end architecture;
