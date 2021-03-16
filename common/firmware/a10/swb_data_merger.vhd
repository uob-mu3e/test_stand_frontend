-----------------------------------------------------------------------------
-- Converting merged data to 256 link to farm
--
-- Marius Koeppel, JGU Mainz
-- mkoeppel@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


entity swb_data_merger is
    generic (
        NLINKS      : positive := 8;
        SWB_ID      : std_logic_vector(7 downto 0) := x"01";
        -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
        DATA_TYPE   : std_logic_vector(7 downto 0) := x"01"--;
    );
    port (
        i_reset_n   : in std_logic;
        i_clk       : in std_logic;

        i_data      : in std_logic_vector(NLINKS * 38 - 1 downto 0);
        i_empty     : in std_logic;

        o_ren       : out std_logic;
        o_wen       : out std_logic;
        o_data      : out std_logic_vector(NLINKS * 32 - 1  downto 0);
        o_datak     : out std_logic_vector(NLINKS * 4 - 1  downto 0)--;
);
end entity;

architecture arch of swb_data_merger is
         
    type merge_state_type is (wait_for_pre, get_ts_1, get_ts_2, get_sh, hit, delay, get_tr, error_state);
    signal merge_state : merge_state_type;

    signal o_data_reg : std_logic_vector(71 downto 0);
    signal hit_reg    : std_logic_vector(NLINKS * 38 - 1  downto 0);
    signal hit_reg_cnt : integer;
    signal header_state : std_logic_vector(5 downto 0);
    
begin

    header_state(0) <= '1' when i_data(37 downto 32) = pre_marker and i_data(7 downto 0) = x"BC" else '0';
    header_state(1) <= '1' when i_data(37 downto 32) = ts1_marker else '0';
    header_state(2) <= '1' when i_data(37 downto 32) = ts2_marker else '0';
    header_state(3) <= '1' when i_data(37 downto 32) = sh_marker else '0';
    header_state(4) <= '1' when i_data(37 downto 32) = tr_marker and i_data(7 downto 0) = x"9C" else '0';
    header_state(5) <= '1' when i_data(37 downto 32) = err_marker else '0';

    o_ren <= '1' when or_reduce(header_state) = '1' and i_empty = '0' and merge_state /= hit else 
             '1' when merge_state = hit and hit_reg_cnt /= 6 and or_reduce(header_state) = '0' and i_empty = '0' else '0';

    process(i_clk, i_reset_n)
    begin
        if ( i_reset_n = '0' ) then
            o_wen       <= '0';
            o_data      <= (others => '0');
            o_data_reg  <= (others => '0');
            o_datak     <= (others => '0');
            hit_reg     <= (others => '0');
            merge_state <= wait_for_pre;
            hit_reg_cnt <= 0;
            --
        elsif ( rising_edge(i_clk) ) then

            o_wen       <= '0';
            o_data      <= (others => '0');

            case merge_state is

                when wait_for_pre =>
                    if ( header_state(0) = '1' ) then
                        merge_state             <= get_ts_1;
                        -- reg data
                        o_data_reg(7 downto 0)  <= SWB_ID;
                    end if;
                    FOR I in NLINKS - 1 downto 0 LOOP
                        o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_5;
                        o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                        o_wen                               <= '1';
                    END LOOP;

                when get_ts_1 =>
                    if ( header_state(1) = '1' ) then
                        merge_state             <= get_ts_2;
                        -- reg data
                        o_data_reg(39 downto 8) <= i_data(31 downto 0);
                    end if;
                    FOR I in NLINKS - 1 downto 0 LOOP
                        o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_5;
                        o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                        o_wen                               <= '1';
                    END LOOP;

                when get_ts_2 =>
                    -- send out data if ts2 is there
                    -- every link is getting K.28.3 = 7C for pre
                    o_wen <= '1';
                    if ( header_state(2) = '1' ) then
                        merge_state              <= get_sh;
                        o_data_reg               <= (others => '0');
                        -- 1. link
                        o_data(31 downto 0)      <= o_data_reg(23 downto 0) & x"7C";
                        o_datak(3 downto 0)      <= "0001";
                        -- 2. link
                        o_data(63 downto 40)     <= o_data_reg(47 downto 24);
                        o_data(39 downto 32)     <= x"7C";
                        o_datak(7 downto 4)      <= "0001";
                        -- 3. link
                        o_data(95 downto 72)     <= o_data_reg(71 downto 48);
                        o_data(71 downto 64)     <= x"7C";
                        o_datak(11 downto 8)     <= "0001";
                        -- 4. link
                        o_data(127 downto 104)   <= i_data(23 downto 0);
                        o_data(103 downto 96)    <= x"7C";
                        o_datak(15 downto 12)    <= "0001";
                        -- 5. link
                        o_data(143 downto 136)   <= i_data(31 downto 24);
                        o_data(135 downto 128)   <= x"7C";
                        o_datak(19 downto 16)    <= "0001";
                        -- 6. - 8. link
                        FOR I in NLINKS - 1 downto 5 LOOP
                            o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_3;
                            o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                        END LOOP;
                    else
                        FOR I in NLINKS - 1 downto 0 LOOP
                            o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_5;
                            o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                        END LOOP;
                    end if;

                when get_sh =>
                    -- send out data if sh is there
                    -- every link is getting K.28.2 = 5C for sh
                    o_wen <= '1';
                    if ( header_state(3) = '1' ) then
                        merge_state             <= delay;
                        -- 1. link
                        o_data(31 downto 0)     <= i_data(15 downto 0) & DATA_TYPE & x"5C";
                        -- 2. link
                        o_data(63 downto 32)    <= i_data(31 downto 16) & x"00" & x"5C";
                        -- 3. - 8. link
                        FOR I in NLINKS - 1 downto 2 LOOP
                            o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_2;
                            o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                        END LOOP;
                    else
                        FOR I in NLINKS - 1 downto 0 LOOP
                            o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_5;
                            o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                        END LOOP;
                    end if;

                when delay =>
                    merge_state <= hit;
                    o_wen       <= '1';
                    FOR I in NLINKS - 1 downto 0 LOOP
                        o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_5;
                        o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                    END LOOP;

                -- hits after alignment
                -- 1. hit  =  37 downto   0
                -- 2. hit  =  75 downto  38
                -- 3. hit  = 113 downto  76
                -- 4. hit  = 151 downto 114
                -- 5. hit  = 189 downto 152
                -- 6. hit  = 227 downto 190
                -- 6.5 hit = 246 downto 228
                -- 7. hit  = 265 downto 228
                -- 8. hit  = 303 downto 266
                -- marker => 10 -> 1/2 MSB
                --           01 -> 1/2 LSB
                --           00 -> error
                --           11 -> no 1/2 hits
                when hit =>
                    -- send out hits if fifo is not empty
                    o_wen <= '1';
                    if ( i_empty = '0' ) then
                        hit_reg <= (others => '0');
                        -- marker 
                        o_data(255 downto 254)      <= "11";
                        if ( header_state(3) = '1' ) then
                            merge_state <= get_sh;
                            hit_reg_cnt <= 0;
                            -- writeout last reg data
                            o_data(227 downto 0) <= hit_reg(227 downto 0);
                        elsif ( header_state(4) = '1' ) then
                            merge_state <= get_tr;
                            hit_reg_cnt <= 0;
                            -- writeout last reg data
                            o_data(227 downto 0) <= hit_reg(227 downto 0);
                        elsif ( header_state(5) = '1' ) then
                            merge_state <= error_state;
                            hit_reg_cnt <= 0;
                            -- writeout last reg data
                            o_data(227 downto 0) <= hit_reg(227 downto 0);
                            hit_reg <= i_data;
                        else
                            FOR I in NLINKS - 1 downto 0 LOOP
                                o_datak(I * 4 + 3 downto I * 4)     <= "0000";
                            END LOOP;                         
                            if ( hit_reg_cnt = 6 ) then
                                -- 6 hits from reg
                                o_data(227 downto 0)    <= hit_reg(227 downto 0);
                                -- set cnt
                                hit_reg_cnt <= 0;
                            elsif ( hit_reg_cnt = 4 ) then
                                -- 4 from reg
                                o_data(151 downto 0)    <= hit_reg(151 downto 0);
                                -- 2 hits from input
                                o_data(227 downto 152)  <= i_data(75 downto 0);
                                -- save 6 hits
                                hit_reg(227 downto 0)   <= i_data(303 downto 76);
                                -- set cnt
                                hit_reg_cnt <= 6;
                            elsif ( hit_reg_cnt = 2 ) then
                                -- 2 from reg
                                o_data(75 downto 0)     <= hit_reg(75 downto 0);
                                -- 4 hits from input
                                o_data(227 downto 76)   <= i_data(151 downto 0);
                                -- save 4 hits
                                hit_reg(151 downto 0)   <= i_data(303 downto 152);
                                -- set cnt
                                hit_reg_cnt <= 4;
                            else
                                -- 6 hits from input
                                o_data(227 downto 0)    <= i_data(227 downto 0);
                                -- save 2 hits
                                hit_reg(75 downto 0)    <= i_data(303 downto 228);
                                -- set cnt
                                hit_reg_cnt <= 2;
                            end if;
                        end if;
                    else
                        FOR I in NLINKS - 1 downto 0 LOOP
                            o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_5;
                            o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                        END LOOP;
                    end if;

                when get_tr =>
                    -- send out data if tr is there
                    -- every link is getting K.28.4 = 9C for tr
                    o_wen <= '1';
                    merge_state             <= wait_for_pre;
                    FOR I in NLINKS - 1 downto 0 LOOP
                        o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_4;
                        o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                    END LOOP;

                when error_state =>
                    o_wen               <= '1';
                    o_data(7 downto 0)  <= x"7C";
                    o_data(8)           <= hit_reg(12); -- error_gtime1
                    o_data(9)           <= hit_reg(13); -- error_gtime2
                    o_data(10)          <= hit_reg(14); -- error_shtime
                    o_data(11)          <= hit_reg(15); -- error_merger
                    o_data(31 downto 12)<= x"0000" & "0000"; -- free bits
                    o_datak(3 downto 0) <= "0001";
                    FOR I in NLINKS - 2 downto 0 LOOP
                        o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_6;
                        o_datak(I * 4 + 3 downto I * 4)     <= "0001";
                    END LOOP;
                    merge_state <= get_tr;

                when others =>
                    merge_state <= wait_for_pre;
                    o_data_reg  <= (others => '0');
                    hit_reg     <= (others => '0');
                    hit_reg_cnt <= 0;

            end case;

        end if;
    end process;
    
end architecture;
