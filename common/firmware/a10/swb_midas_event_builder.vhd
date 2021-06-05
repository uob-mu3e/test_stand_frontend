--
-- Marius Koeppel, November 2020
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity swb_midas_event_builder is
generic(
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE : std_logic_vector(7 downto 0) := x"01"--;
);
port(
    i_rx                : in  std_logic_vector (31 downto 0);
    i_rempty            : in  std_logic;
    i_header            : in  std_logic;
    i_trailer           : in  std_logic;

    i_get_n_words       : in  std_logic_vector (31 downto 0);
    i_dmamemhalffull    : in  std_logic;
    i_wen               : in  std_logic;
    o_data              : out std_logic_vector (255 downto 0);
    o_wen               : out std_logic;
    o_ren               : out std_logic;
    o_dma_cnt_words     : out std_logic_vector (31 downto 0);
    o_endofevent        : out std_logic;
    o_done              : out std_logic;
    o_state_out         : out std_logic_vector (3 downto 0);
    
    --! status counters 
    --! 0: bank_builder_idle_not_header
    --! 1: bank_builder_skip_event_dma
    --! 2: bank_builder_ram_full
    --! 3: bank_builder_tag_fifo_full
    o_counters          : out work.util.slv32_array_t(3 downto 0);

    i_reset_n_250       : in  std_logic;
    i_clk_250           : in  std_logic--;
);
end entity;

architecture arch of swb_midas_event_builder is

    -- tagging fifo
    type event_tagging_state_type is (
        event_head, event_num, event_tmp, event_size, bank_size, bank_flags, bank_name, bank_type, bank_length, bank_data, bank_set_length, event_set_size, bank_set_size, write_tagging_fifo, set_algin_word, bank_reserved, EVENT_IDLE--,
    );
    signal event_tagging_state : event_tagging_state_type;
    signal e_size_add, b_size_add, b_length_add, w_ram_add_reg, w_ram_add, w_fifo_data, r_fifo_data, last_event_add, align_event_size : std_logic_vector(11 downto 0);
    signal w_fifo_en, r_fifo_en, tag_fifo_empty, tag_fifo_full : std_logic;

    -- ram 
    signal w_ram_en : std_logic;
    signal r_ram_add : std_logic_vector(8 downto 0);
    signal w_ram_data : std_logic_vector(31 downto 0);
    signal r_ram_data : std_logic_vector(255 downto 0);

    -- midas event 
    signal event_id, trigger_mask : std_logic_vector(15 downto 0);
    signal serial_number, time_tmp, type_bank, flags, bank_size_cnt, event_size_cnt : std_logic_vector(31 downto 0);
    
    -- event readout state machine
    type event_counter_state_type is (waiting, get_data, runing, skip_event);
    signal event_counter_state : event_counter_state_type;
    signal event_last_ram_add : std_logic_vector(8 downto 0);
    signal word_counter, word_counter_endofevent : std_logic_vector(31 downto 0);

    -- error cnt
    signal cnt_tag_fifo_full : std_logic_vector(31 downto 0);
    signal cnt_ram_full : std_logic_vector(31 downto 0);
    signal cnt_skip_event_dma : std_logic_vector(31 downto 0);
    signal cnt_idle_not_header : std_logic_vector(31 downto 0);

begin

    --! counter
    o_counters(0) <= cnt_idle_not_header;
    o_counters(1) <= cnt_skip_event_dma;
    --! TODO implement bank_builder_ram_full
    o_counters(2) <= (others => '0');
    e_cnt_tag_fifo : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(3), i_ena => tag_fifo_full, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    --! data out
    o_data <= r_ram_data;

    e_ram_32_256 : entity work.ip_ram
    generic map (
        ADDR_WIDTH_A    => 12,
        ADDR_WIDTH_B    => 9,
        DATA_WIDTH_A    => 32,
        DATA_WIDTH_B    => 256,
        DEVICE          => "Arria 10"--,
    )
    port map (
        address_a       => w_ram_add,
        address_b       => r_ram_add,
        clock_a         => i_clk_250,
        clock_b         => i_clk_250,
        data_a          => w_ram_data,
        data_b          => (others => '0'),
        wren_a          => w_ram_en,
        wren_b          => '0',
        q_a             => open,
        q_b             => r_ram_data--,
    );

    e_tagging_fifo_event : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 12,
        DATA_WIDTH      => 12,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_fifo_data,
        wrreq           => w_fifo_en,
        rdreq           => r_fifo_en,
        clock           => i_clk_250,
        q               => r_fifo_data,
        full            => tag_fifo_full,
        empty           => tag_fifo_empty,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => not i_reset_n_250--,
    );

    o_ren <=
        '1' when ( event_tagging_state = bank_data and i_rempty = '0' ) else
        '1' when ( event_tagging_state = EVENT_IDLE and i_rempty = '0' and i_header = '0' ) else
        '0';

    -- write link data to event ram
    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 = '0' ) then
        e_size_add          <= (others => '0');
        b_size_add          <= (others => '0');
        b_length_add        <= (others => '0');
        w_ram_add_reg       <= (others => '0');
        last_event_add      <= (others => '0');
        align_event_size    <= (others => '0');

        -- ram and tagging fifo write signals
        w_ram_en            <= '0';
        w_ram_data          <= (others => '0');
        w_ram_add           <= (others => '1');
        w_fifo_en           <= '0';
        w_fifo_data         <= (others => '0');

        -- midas signals
        event_id            <= x"0001";
        trigger_mask        <= (others => '0');
        serial_number       <= x"00000001";
        time_tmp            <= (others => '0');
        flags               <= x"00000031";
        type_bank           <= x"00000006"; -- MIDAS Bank Type TID_DWORD
    
        -- for size counting in bytes
        bank_size_cnt       <= (others => '0');
        event_size_cnt      <= (others => '0');
        cnt_idle_not_header <= (others => '0');

        -- state machine singals
        event_tagging_state <= EVENT_IDLE;

    --
    elsif rising_edge(i_clk_250) then
        flags           <= x"00000031";
        trigger_mask    <= (others => '0');
        event_id        <= x"0001";
        type_bank       <= x"00000006";
        w_ram_en        <= '0';
        w_fifo_en       <= '0';

        if ( event_tagging_state /= EVENT_IDLE ) then
            -- count time for midas event header
            time_tmp <= time_tmp + '1';
        end if;

        case event_tagging_state is
        when EVENT_IDLE =>
            -- start if at least one not masked link has data
            if ( i_rempty = '0' and i_header = '1' ) then
                event_tagging_state <= event_head;
            elsif ( i_rempty = '0' and i_header = '0' ) then
                cnt_idle_not_header <= cnt_idle_not_header + 1;
            end if;

        when event_head =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= trigger_mask & event_id;
            last_event_add      <= w_ram_add + 1;
            event_tagging_state <= event_num;

        when event_num =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= serial_number;
            event_tagging_state <= event_tmp;

        when event_tmp =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= time_tmp;
            event_tagging_state <= event_size;

        when event_size =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            event_tagging_state <= bank_size;
            e_size_add          <= w_ram_add + 1;

        when bank_size =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            event_size_cnt      <= event_size_cnt + 4;
            b_size_add          <= w_ram_add + 1;
            event_tagging_state <= bank_flags;

        when bank_flags =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_add_reg       <= w_ram_add + 1;
            w_ram_data          <= flags;
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= bank_name;

        when bank_name =>
            -- here we check if the link is empty and if we saw a header
            if ( i_rempty = '0' and i_header = '1' ) then
                w_ram_en    <= '1';
                w_ram_add   <= w_ram_add_reg + 1;
                -- MIDAS expects bank names in ascii:
                -- For the run 2021 
                -- PCD1 = PixelCentralDebug1
                -- SCD1 = ScifiCentralDebug1
                -- TCD1 = TileCentralDebug1
                -- NONE else
                if ( DATA_TYPE = x"01" ) then
                    w_ram_data <= x"31444350";
                elsif ( DATA_TYPE = x"02" ) then
                    w_ram_data <= x"31444353";
                elsif ( DATA_TYPE = x"03" ) then
                    w_ram_data <= x"31444354";
                else
                    w_ram_data <= x"454E4F4E";
                end if;
                event_size_cnt      <= event_size_cnt + 4;
                event_tagging_state <= bank_type;
            end if;

        when bank_type =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= type_bank;
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= bank_length;

        when bank_length =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            event_size_cnt      <= event_size_cnt + 4;
            b_length_add        <= w_ram_add + 1;
            event_tagging_state <= bank_reserved;
            
        when bank_reserved =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= bank_data;            

        when bank_data =>
            -- check again if the fifo is empty
            if ( i_rempty = '0' and i_rx /= x"FFFFFFFF" ) then
                w_ram_en            <= '1';
                w_ram_add           <= w_ram_add + 1;
                if ( i_trailer = '1' ) then
                    w_ram_data(31 downto 12)    <= x"0FC00";
                    w_ram_data(11 downto 8)     <= "00" & i_rx(9 downto 8);
                    w_ram_data(7 downto 0)      <= x"9C";
                else
                    w_ram_data      <= i_rx;
                end if;
                event_size_cnt      <= event_size_cnt + 4;
                bank_size_cnt       <= bank_size_cnt + 4;
                if ( i_trailer = '1' ) then
                    event_tagging_state <= set_algin_word;
                    align_event_size    <= w_ram_add + 1 - last_event_add;
                end if;
            end if;

        when set_algin_word =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_add_reg       <= w_ram_add + 1;
            w_ram_data          <= x"AFFEAFFE";
            align_event_size    <= align_event_size + 1;
            -- check if the size of the bank data 
            -- is in 64 bit and 256 bit
            -- if not add a dummy words
            if ( align_event_size(2 downto 0) + '1' = "000" ) then
                event_tagging_state <= bank_set_length;
            else
                bank_size_cnt   <= bank_size_cnt + 4;
                event_size_cnt  <= event_size_cnt + 4;
            end if;

        when bank_set_length =>
            w_ram_en            <= '1';
            w_ram_add           <= b_length_add;
            w_ram_add_reg       <= w_ram_add;
            w_ram_data          <= bank_size_cnt;
            bank_size_cnt       <= (others => '0');
            event_tagging_state <= event_set_size;

        when event_set_size =>
            w_ram_en            <= '1';
            w_ram_add           <= e_size_add;
            -- Event Data Size: The event data size contains the size of the event in bytes excluding the event header
            w_ram_data          <= event_size_cnt;
            event_tagging_state <= bank_set_size;

        when bank_set_size =>
            w_ram_en            <= '1';
            w_ram_add           <= b_size_add;
            -- All Bank Size : Size in bytes of the following data banks including their bank names
            w_ram_data          <= event_size_cnt - 8;
            event_size_cnt      <= (others => '0');
            event_tagging_state <= write_tagging_fifo;

        when write_tagging_fifo =>
            w_fifo_en           <= '1';
            w_fifo_data         <= w_ram_add_reg;
            last_event_add      <= w_ram_add_reg;
            w_ram_add           <= w_ram_add_reg - 1;
            event_tagging_state <= EVENT_IDLE;
            b_length_add        <= (others => '0');
            serial_number       <= serial_number + '1';

        when others =>
            event_tagging_state <= EVENT_IDLE;

        end case;

    end if;
    end process;


    -- dma end of events, count events and write control
    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 = '0' ) then
        o_wen        <= '0';
        o_endofevent        <= '0';
        o_state_out         <= x"0";
        cnt_skip_event_dma  <= (others => '0');
        o_done              <= '0';
        r_fifo_en           <= '0';
        r_ram_add           <= (others => '1');
        event_last_ram_add  <= (others => '0');
        event_counter_state <= waiting;	
        word_counter        <= (others => '0');
        o_dma_cnt_words     <= (others => '0');
        word_counter_endofevent <= (others => '0');
        --
    elsif rising_edge(i_clk_250) then

        o_done          <= '0';
        r_fifo_en       <= '0';
        o_wen    <= '0';
        o_endofevent    <= '0';
        
        if ( i_wen = '0' ) then
            word_counter <= (others => '0');
        end if;
        
        if ( i_wen = '1' and word_counter >= i_get_n_words ) then
            o_done <= '1';
            o_dma_cnt_words <= word_counter_endofevent;
        end if;

        case event_counter_state is
        when waiting =>
                o_state_out             <= x"A";
                if (tag_fifo_empty = '0') then
                    r_fifo_en           <= '1';
                    event_last_ram_add  <= r_fifo_data(11 downto 3);
                    r_ram_add           <= r_ram_add + '1';
                    event_counter_state <= get_data;
                end if;

        when get_data =>
                o_state_out             <= x"B";
                if ( i_dmamemhalffull = '1' or ( i_get_n_words /= (i_get_n_words'range => '0') and word_counter >= i_get_n_words ) ) then
                    event_counter_state <= skip_event;
                    cnt_skip_event_dma  <= cnt_skip_event_dma + '1';
                else
                    o_wen               <= i_wen;
                    word_counter        <= word_counter + '1';
                    event_counter_state <= runing;
                end if;
                r_ram_add       <= r_ram_add + '1';

        when runing =>
                o_state_out             <= x"C";
                o_wen                   <= i_wen;
                word_counter            <= word_counter + '1';
                if(r_ram_add = event_last_ram_add - '1') then
                    o_endofevent        <= '1'; -- end of event
                    event_counter_state <= waiting;
                    if ( word_counter + '1' <= i_get_n_words ) then
                        word_counter_endofevent <= word_counter + '1';
                    end if;
                else
                    r_ram_add <= r_ram_add + '1';
                end if;

        when skip_event =>
                o_state_out <= x"E";
                if(r_ram_add = event_last_ram_add - '1') then
                    event_counter_state	<= waiting;
                else
                    r_ram_add <= r_ram_add + '1';
                end if;

        when others =>
                o_state_out <= x"D";
                event_counter_state	<= waiting;
                
        end case;

    end if;
    end process;

end architecture;
