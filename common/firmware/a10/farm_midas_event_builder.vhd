-------------------------------------------------------
--! farm_midas_event_builder.vhd
--! @brief the @farm_midas_event_builder builds 32a_banks
--! for MIDAS which are stored in the DDR Memory
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity farm_midas_event_builder is
generic(
    g_NLINKS_SWB_TOTL    : positive :=  16;
    N_PIXEL              : positive :=  8;
    N_SCIFI              : positive :=  8;
    RAM_ADDR_W           : positive :=  10;
    RAM_ADDR_R           : positive :=  10--;
);
port(
    i_pixel             : in  std_logic_vector(N_PIXEL * 32 + 1 downto 0);
    i_empty_pixel       : in  std_logic;
    o_ren_pixel         : out std_logic;
    
    i_scifi             : in  std_logic_vector(N_SCIFI * 32 + 1 downto 0);
    i_empty_scifi       : in  std_logic;
    o_ren_scifi         : out std_logic;

    -- DDR
    o_data              : out std_logic_vector(511 downto 0);
    o_wen               : out std_logic;
    o_endofevent        : out std_logic;
    o_event_ts          : out std_logic_vector(47 downto 0);
    i_ddr_ready         : in  std_logic;
    
    -- Link data
    o_pixel             : out std_logic_vector(N_PIXEL * 32 + 1 downto 0);
    o_wen_pixel         : out std_logic;
    
    o_scifi             : out std_logic_vector(N_SCIFI * 32 + 1 downto 0);
    o_wen_scifi         : out std_logic;
    
    --! status counters 
    --! 0: cnt_idle_not_header_pixel
    --! 1: cnt_idle_not_header_scifi
    --! 2: bank_builder_ram_full
    --! 3: bank_builder_tag_fifo_full
    --! 4: # events written to RAM (one subheader of Pixel and Scifi)
    --! 5: # 256b pixel x"FFF"
    --! 6: # 256b scifi x"FFF"
    --! 7: # 256b pixel written to link
    --! 8: # 256b scifi written to link
    o_counters          : out work.util.slv32_array_t(8 downto 0);

    i_reset_n_250       : in  std_logic;
    i_clk_250           : in  std_logic--;
);
end entity;

architecture arch of farm_midas_event_builder is

    -- tagging fifo
    type event_tagging_state_type is ( 
        EVENT_IDLE, event_head, bank_data_pixel_header, bank_data_pixel_one, 
        bank_data_pixel, bank_data_scifi_header, bank_data_scifi_one, 
        bank_data_scifi, align_event_size, bank_set_length_pixel, 
        bank_set_length_scifi, write_tagging_fifo--,
    );
    signal event_tagging_state : event_tagging_state_type;
    signal w_ram_add_reg, w_ram_add, w_fifo_data, r_fifo_data : std_logic_vector(RAM_ADDR_W - 1 downto 0);
    signal w_fifo_en, r_fifo_en, tag_fifo_empty, tag_fifo_full : std_logic;

    -- ram 
    signal w_ram_en : std_logic;
    signal r_ram_add : std_logic_vector(RAM_ADDR_R - 1 downto 0);
    signal header_pixel : std_logic_vector(N_PIXEL * 32 + 1 downto 0);
    signal header_scifi : std_logic_vector(N_SCIFI * 32 + 1 downto 0);
    signal w_ram_data : std_logic_vector(383 downto 0);
    signal r_ram_data : std_logic_vector(255 downto 0);

    -- midas event 
    signal event_id, trigger_mask : std_logic_vector(15 downto 0);
    signal serial_number, time_tmp, type_bank, flags, event_size_cnt : std_logic_vector(31 downto 0);
    
    -- bank bank builder
    signal ts : std_logic_vector(47 downto 0);

    -- error cnt
    signal cnt_idle_not_header_pixel, cnt_idle_not_header_scifi, cnt_idle_pixel_marked, cnt_idle_scifi_marked : std_logic_vector(31 downto 0);
    signal flip, ram_halffull : std_logic;
    signal sub_add : std_logic_vector(31 downto 0);

begin

    --! counter
    o_counters(0) <= cnt_idle_not_header_pixel;
    
    o_counters(1) <= cnt_idle_not_header_scifi;
    
    -- calculate ram halffull
    flip <= '0' when w_ram_add >= r_ram_add else '1';
    sub_add <= w_ram_add - r_ram_add when flip = '0' else r_ram_add - w_ram_add;
    ram_halffull <= sub_add(RAM_ADDR_W-1);
    e_cnt_ram_halffull : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(2), i_ena => ram_halffull, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );
    
    e_cnt_tag_fifo : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(3), i_ena => tag_fifo_full, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );
    
    o_counters(4) <= serial_number;
    
    o_counters(5) <= cnt_idle_pixel_marked;
    
    o_counters(6) <= cnt_idle_scifi_marked;
    
    e_cnt_pixel_link : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(7), i_ena => not i_empty_pixel, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );
    
    e_cnt_scifi_link : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(8), i_ena => not i_empty_scifi, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    --! data out
    o_data <= r_ram_data;

    e_ram_32_256 : entity work.ip_ram
    generic map (
        ADDR_WIDTH_A    => RAM_ADDR_W,
        ADDR_WIDTH_B    => RAM_ADDR_R,
        DATA_WIDTH_A    => 384,
        DATA_WIDTH_B    => 512,
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
        ADDR_WIDTH      => RAM_ADDR_W,
        DATA_WIDTH      => RAM_ADDR_R + 48,
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
    
    pixel_header <= '1' when i_pixel(N_PIXEL * 32 + 1 downto N_PIXEL * 32) = "01" else '0';
    pixel_trailer<= '1' when i_pixel(N_PIXEL * 32 + 1 downto N_PIXEL * 32) = "10" else '0';
    scifi_header <= '1' when i_scifi(N_SCIFI * 32 + 1 downto N_SCIFI * 32) = "01" else '0';
    scifi_trailer<= '1' when i_scifi(N_SCIFI * 32 + 1 downto N_SCIFI * 32) = "10" else '0';

    o_ren_pixel <=
        '1' when ( (event_tagging_state = bank_data_pixel or event_tagging_state = bank_data_pixel_one) and i_empty_pixel = '0' and pixel_header = '0' ) else
        '1' when ( event_tagging_state = event_head and i_empty_pixel = '0' ) else
        '1' when ( event_tagging_state = EVENT_IDLE and i_empty_pixel = '0' and pixel_header = '0' and i_pixel(223 downto 200) = x"FFF" ) else
        '0';
        
    o_ren_scifi <=
        '1' when ( (event_tagging_state = bank_data_scifi or event_tagging_state = bank_data_scifi_one) and i_empty_scifi = '0' and scifi_header = '0' ) else
        '1' when ( event_tagging_state = event_head and i_empty_scifi = '0' ) else
        '1' when ( event_tagging_state = EVENT_IDLE and i_empty_scifi = '0' and scifi_header = '0' and i_scifi(223 downto 200) = x"FFF" ) else
        '0';
        
    -- mark if data is send to DDR3 for the next farm pc
    o_pixel(N_PIXEL * 32 + 1 downto 224) <= i_pixel(N_PIXEL * 32 + 1 downto 224);
    o_pixel(223 downto 200) <= x"000" when ( event_tagging_state = EVENT_IDLE and (pixel_header = '0' or scifi_header = '0' or ram_halffull = '0') ) else x"FFF";
    o_pixel(199 downto 0) <= i_pixel(199 downto 0);
    o_wen_pixel <= not i_empty_pixel;
    cnt_fff <= '0' when ( event_tagging_state = EVENT_IDLE and (pixel_header = '0' or scifi_header = '0' or ram_halffull = '0') ) else '1';
    o_scifi(N_SCIFI * 32 + 1 downto 224) <= i_scifi(N_SCIFI * 32 + 1 downto 224);
    o_scifi(223 downto 200) <= x"000" when ( event_tagging_state = EVENT_IDLE and (pixel_header = '0' or scifi_header = '0' or ram_halffull = '0') ) else x"FFF";
    o_scifi(199 downto 0) <= i_scifi(199 downto 0);
    o_wen_scifi <= not i_empty_scifi;

    -- write link data to event ram
    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 = '0' ) then
        e_size_add          <= (others => '0');
        b_size_add          <= (others => '0');
        b_length_add        <= (others => '0');
        w_ram_add_reg       <= (others => '0');
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
        event_size_cnt      <= (others => '0');
        cnt_idle_not_header_pixel <= (others => '0');
        cnt_idle_not_header_scifi <= (others => '0');
        cnt_idle_pixel_marked <= (others => '0');
        cnt_idle_scifi_marked <= (others => '0');

        -- state machine singals
        event_tagging_state <= EVENT_IDLE;

    --
    elsif ( rising_edge(i_clk_250) ) then
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

        -- ram back-pressure
        if ( ram_halffull = '0' ) then
            case event_tagging_state is
            when EVENT_IDLE =>
            
                if ( i_empty_pixel = '0' and pixel_header = '0' ) then
                    cnt_idle_not_header_pixel <= cnt_idle_not_header_pixel + 1;
                end if;
                
                if ( i_empty_scifi = '0' and scifi_header = '0' ) then
                    cnt_idle_not_header_scifi <= cnt_idle_not_header_scifi + 1;
                end if;
                
                if ( i_empty_pixel = '0' and i_pixel(223 downto 200) = x"FFF" ) then
                    cnt_idle_pixel_marked <= cnt_idle_pixel_marked + 1;
                end if;
                
                if ( i_empty_scifi = '0' and i_scifi(223 downto 200) = x"FFF" ) then
                    cnt_idle_scifi_marked <= cnt_idle_scifi_marked + 1;
                end if;
                
                -- start when both Scifi and Pixel are not empty, both have header and ram_halffull = '0'
                if ( i_empty_pixel = '0' and i_empty_pixel = '0' and pixel_header = '1' and scifi_header = '1' and ram_halffull = '0' and i_pixel(223 downto 200) = x"000" and i_scifi(223 downto 200) = x"000" ) then
                    event_tagging_state <= event_head;
                    header_pixel        <= i_pixel;
                    header_scifi        <= i_scifi;
                end if;

            when event_head =>
                -- TODO: compare TS of headers --> error if not the same?
                -- if ( header_pixel(x downto y) /= header_scifi(x downto y) ) then
                --  error_ts_header <= '1';
                -- end if;
                
                -- TODO: store TS of headers for DDR address
                ts(15 downto  0) <= header_pixel(87 downto 72);
                ts(31 downto 16) <= header_pixel(31 downto 16);
                ts(47 downto 32) <= header_pixel(55 downto 40);
                
                -- event header
                w_ram_pixel_header( 31 downto   0) <= trigger_mask & event_id;
                w_ram_pixel_header( 63 downto  32) <= serial_number;
                w_ram_pixel_header( 95 downto  64) <= time_tmp;
                w_ram_pixel_header(127 downto  96) <= (others => '0'); -- e_size
                -- bank header
                w_ram_pixel_header(159 downto 128) <= (others => '0'); -- b_size
                w_ram_pixel_header(191 downto 160) <= flags;
                -- BANK32A (start with IPIX)
                w_ram_pixel_header(223 downto 192) <= x"58495049"; -- bank header
                w_ram_pixel_header(255 downto 224) <= type_bank;
                w_ram_pixel_header(287 downto 256) <= (others => '0'); -- bank length
                w_ram_pixel_header(319 downto 288) <= (others => '0'); -- bank reserved
                w_ram_pixel_header(351 downto 320) <= header_pixel(127 downto  96); -- overflow (24b) & 7C
                w_ram_pixel_header(383 downto 352) <= header_pixel(159 downto 128); -- overflow (24b) & 7C
                w_ram_data                 <= (others => '0');
                event_size_cnt             <= event_size_cnt + 8*4; -- b_size, flags, bank name, bank type, bank length, bank reserved, overflow, overflow
                header_add                 <= w_ram_add + 1;
                w_ram_add                  <= w_ram_add + 1;
                w_ram_en                   <= '1';
                event_tagging_state        <= bank_data_pixel_header;

            when bank_data_pixel_header =>
                w_ram_data(63 downto 0) <= (others => '0'); -- reserved
                event_tagging_state     <= bank_data_pixel_one;
                
            when bank_data_pixel_one =>
                -- check again if the fifo is empty and no header/trailer
                if ( i_empty_pixel = '0' and pixel_header = '0' and pixel_trailer = '0' ) then
                    -- convert 5 38 bit hits to 5 64 bit hits
                    w_ram_data(383 downto 64) <= convert_to_64_pixel(i_pixel(189 downto 0), 5);
                    i_pixel_reg(63 downto 0)  <= convert_to_64_pixel(i_pixel(227 downto 190), 1);
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_pixel           <= bank_size_pixel + 14*4;
                    w_ram_add                 <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                    event_tagging_state       <= bank_data_pixel;
                elsif ( pixel_header = '1' or pixel_trailer = '1' ) then
                    w_ram_data(383 downto 64) <= (others => '1');
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_pixel           <= bank_size_pixel + 14*4;
                    w_ram_add                 <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                    event_tagging_state       <= bank_data_scifi_header;
                end if;
                
            when bank_data_pixel =>
                -- check again if the fifo is empty and no header/trailer
                if ( i_empty_pixel = '0' and pixel_header = '0' and pixel_trailer = '0' ) then
                    w_ram_data(63 downto 0)   <= i_pixel_reg;
                    w_ram_data(383 downto 64) <= convert_to_64_pixel(i_pixel(189 downto 0), 5);
                    i_pixel_reg(63 downto 0)  <= convert_to_64_pixel(i_pixel(227 downto 190), 1);
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_pixel           <= bank_size_pixel + 12*4;
                    w_ram_add                 <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                elsif ( pixel_header = '1' or pixel_trailer = '1' ) then
                    w_ram_data(63 downto 0)   <= i_pixel_reg;
                    w_ram_data(383 downto 64) <= (others => '1');
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_pixel           <= bank_size_pixel + 12*4;
                    w_ram_add                 <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                    event_tagging_state       <= bank_data_scifi_header;
                end if;

            when bank_data_scifi_header =>
                -- BANK32A ISCI
                w_ram_scifi_data( 31 downto   0) <= x"49435349"; -- bank header
                w_ram_scifi_data( 63 downto  32) <= type_bank;
                w_ram_scifi_data( 95 downto  64) <= (others => '0'); -- bank length
                w_ram_scifi_data(127 downto  96) <= (others => '0'); -- bank reserved
                w_ram_scifi_data(159 downto 128) <= header_scifi(127 downto  96); -- overflow (24b) & 7C
                w_ram_scifi_data(191 downto 160) <= header_scifi(159 downto 128); -- overflow (24b) & 7C
                w_ram_scifi_data(223 downto 192) <= (others => '0'); -- reserved
                w_ram_scifi_data(255 downto 224) <= (others => '0'); -- reserved
                event_tagging_state              <= bank_data_scifi_one;
                
            when bank_data_scifi_one =>
                -- check again if the fifo is empty and no header/trailer
                if ( i_empty_scifi = '0' and scifi_header = '0' and scifi_trailer = '0' ) then
                    -- convert 2 38 bit hits to 2 64 bit hits
                    w_ram_scifi_data(383 downto 256)<= convert_to_64_scifi(i_scifi(75 downto 0), 2);
                    i_scifi_reg(255 downto 0) <= convert_to_64_scifi(i_scifi(227 downto 76), 4);
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_scifi           <= bank_size_scifi + 16*4;
                    w_ram_add                 <= w_ram_add + 1;
                    scifi_header_add          <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                    event_tagging_state       <= bank_data_scifi;
                elsif ( scifi_header = '1' or scifi_trailer = '1' ) then
                    w_ram_scifi_data(383 downto 256)<= (others => '1');
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_scifi           <= bank_size_scifi + 16*4;
                    w_ram_add                 <= w_ram_add + 1;
                    scifi_header_add          <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                    event_tagging_state       <= align_event_size;
                end if;
                
            when bank_data_scifi =>
                -- check again if the fifo is empty and no header/trailer
                if ( i_empty_scifi = '0' and scifi_header = '0' and scifi_trailer = '0' ) then
                    -- convert 2 38 bit hits to 2 64 bit hits
                    w_ram_data(255 downto 0)  <= i_scifi_reg;
                    w_ram_data(383 downto 256)<= convert_to_64_scifi(i_scifi(75 downto 0), 2);
                    i_scifi_reg(255 downto 0) <= convert_to_64_scifi(i_scifi(227 downto 76), 4);
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_scifi           <= bank_size_scifi + 12*4;
                    w_ram_add                 <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                elsif ( scifi_header = '1' or scifi_trailer = '1' ) then
                    w_ram_data(255 downto 0)  <= i_scifi_reg;
                    w_ram_data(383 downto 256)<= (others => '1');
                    event_size_cnt            <= event_size_cnt + 12*4;
                    bank_size_scifi           <= bank_size_scifi + 12*4;
                    w_ram_add                 <= w_ram_add + 1;
                    w_ram_en                  <= '1';
                    event_tagging_state       <= align_event_size;
                end if;
                
            when align_event_size =>
                -- write padding
                w_ram_data(383 downto 0) <= (others => '1');
                event_size_cnt           <= event_size_cnt + 12*4;
                w_ram_add                <= w_ram_add + 1;
                w_ram_add_reg            <= w_ram_add + 1;
                w_ram_en                 <= '1';
                -- check if the size of the event data 
                -- is in 512 bit if not add dummy words
                if ( w_ram_add(1 downto 0) + '1' = "00" ) then
                    event_tagging_state <= bank_set_length_pixel;
                else
                    event_size_cnt <= event_size_cnt + 12*4;
                    bank_size_scifi <= bank_size_scifi + 12*4;
                end if;
                
            when bank_set_length_pixel =>
                w_ram_en  <= '1';
                w_ram_add <= header_add;
                -- write header values
                w_ram_data( 63 downto   0) <= w_ram_pixel_header( 63 downto   0);
                w_ram_data(127 downto  96) <= event_size_cnt; -- event size
                w_ram_data(159 downto 128) <= event_size_cnt - 8; -- all bank size
                w_ram_data(255 downto 160) <= w_ram_pixel_header(255 downto 160);
                w_ram_data(287 downto 256) <= bank_size_pixel; -- bank length pixel
                w_ram_data(383 downto 288) <= w_ram_pixel_header(383 downto 288);
                bank_size_pixel <= (others => '0');
                event_size_cnt  <= (others => '0');
                event_tagging_state <= bank_set_length_scifi;
                
            when bank_set_length_scifi =>
                w_ram_en        <= '1';
                w_ram_add       <= scifi_header_add;
                -- write header values scifi
                w_ram_data( 63 downto   0)  <= w_ram_scifi_data( 63 downto   0);
                w_ram_data( 95 downto  64)  <= bank_size_scifi; -- bank length scifi
                w_ram_data(383 downto  96)  <= w_ram_scifi_data(383 downto   96);
                bank_size_scifi             <= (others => '0');
                event_tagging_state         <= write_tagging_fifo;

            when write_tagging_fifo =>
                w_fifo_en           <= '1';
                w_fifo_data         <= w_ram_add_reg & ts;
                w_ram_add           <= w_ram_add_reg - 1;
                serial_number       <= serial_number + '1';
                event_tagging_state <= EVENT_IDLE;

            when others =>
                event_tagging_state <= EVENT_IDLE;

            end case;
        end if;
    end if;
    end process;
    
    -- readout MIDAS Bank Builder RAM
    process(i_clk_250, i_reset_n_250)
    begin
    if ( i_reset_n_250 = '0' ) then
        r_fifo_en               <= '0';
        o_wen                   <= '0';
        o_endofevent            <= '0';
        o_event_ts              <= (others => '0');
        event_last_ram_add      <= (others => '0');
        r_ram_add               <= (others => '1');
        event_counter_state     <= waiting;
        --
    elsif rising_edge(i_clk_250) then
        
        r_fifo_en    <= '0';
        o_wen        <= '0';
        o_endofevent <= '0';
        
        case event_counter_state is
        when waiting =>
            if ( tag_fifo_empty = '0' ) then
                r_fifo_en           <= '1';
                event_last_ram_add  <= r_fifo_data(RAM_ADDR_W + 48 - 1 downto 48 + 2);
                o_event_ts          <= r_fifo_data(47 downto 0);
                r_ram_add           <= r_ram_add + '1';
                event_counter_state <= get_data;
            end if;

        when get_data =>
            -- if i_ddr_ready = '1' we switch from
            -- one DDR to the other we wait here until 
            -- this happend so that we dont split events over
            -- the two DDR memories
            if ( i_ddr_ready /= '1' ) then
                o_wen <= '1';
                event_counter_state <= runing;
                r_ram_add <= r_ram_add + '1';
            end if;

        when runing =>
            o_wen <= '1';
            if(r_ram_add = event_last_ram_add - '1') then
                -- end of event
                o_endofevent <= '1'; 
                event_counter_state <= waiting;
            else
                r_ram_add <= r_ram_add + '1';
            end if;

        when others =>
            event_counter_state	<= waiting;
                
        end case;

    end if;
    end process;

end architecture;
