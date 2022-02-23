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
    o_data      : out std_logic_vector(NLINKS * 32 - 1  downto 0);
    o_data_valid: out std_logic_vector(NLINKS * 2 - 1  downto 0)--;
);
end entity;

architecture arch of swb_data_merger is

    type merge_state_type is (wait_for_pre, get_ts_1, get_ts_2, get_sh, hit, delay, get_tr, error_state);
    signal merge_state : merge_state_type;

    signal hit_reg    : std_logic_vector(NLINKS * 38 - 1  downto 0);
    signal hit_reg_cnt : integer;
    signal header_state : std_logic_vector(5 downto 0);
    signal TS : std_logic_vector(47 downto 0);

begin

    header_state(0) <= '1' when i_data(37 downto 32) = pre_marker and i_data(7 downto 0) = x"BC" else '0';
    header_state(1) <= '1' when i_data(37 downto 32) = ts1_marker else '0';
    header_state(2) <= '1' when i_data(37 downto 32) = ts2_marker else '0';
    header_state(3) <= '1' when i_data(37 downto 32) = sh_marker else '0';
    header_state(4) <= '1' when i_data(37 downto 32) = tr_marker and i_data(7 downto 0) = x"9C" else '0';
    header_state(5) <= '1' when i_data(37 downto 32) = err_marker else '0';

    o_ren <= '1' when or_reduce(header_state) = '1' and i_empty = '0' and merge_state /= hit else
             '0' when merge_state = hit and header_state(3) = '1' else
             '1' when merge_state = hit and hit_reg_cnt /= 6 and or_reduce(header_state) = '0' and i_empty = '0' else '0';

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        o_data      <= (others => '1');
        o_data_valid<= (others => '0');
        hit_reg     <= (others => '1');
        TS          <= (others => '0');
        merge_state <= wait_for_pre;
        hit_reg_cnt <= 0;
        --
    elsif ( rising_edge(i_clk) ) then

            o_data      <= (others => '1');
            o_data_valid<= (others => '0'); -- idle

            case merge_state is

                when wait_for_pre =>
                    if ( header_state(0) = '1' ) then
                        merge_state             <= get_ts_1;
                    end if;

                when get_ts_1 =>
                    if ( header_state(1) = '1' ) then
                        merge_state             <= get_ts_2;
                        -- reg data
                        -- TS (47:16)
                        TS(47 downto 16) <= i_data(31 downto 0);
                    end if;

                when get_ts_2 =>
                    -- send out data if ts2 is there
                    -- every link is getting K.28.3 = 7C for pre
                    if ( header_state(2) = '1' ) then
                        merge_state <= get_sh;
                        -- reg data
                        -- TS (15:0)
                        TS(15 downto 0) <= i_data(31 downto 16);
                    end if;

                when get_sh =>
                    -- send out header if sh is there
                    if ( header_state(3) = '1' ) then
                        merge_state              <= delay;
                        -- 1. link
                        -- x"7C"
                        -- SWB ID
                        -- TS(31:16)
                        o_data(15 downto 0)      <= SWB_ID & K28_3;
                        o_data(31 downto 16)     <= TS(31 downto 16);
                        o_data_valid(1 downto 0) <= "10"; -- header
                        -- 2. link
                        -- x"7C"
                        -- TS(47:32)
                        -- DT
                        o_data(39 downto 32)     <= K28_3;
                        o_data(55 downto 40)     <= TS(47 downto 32);
                        o_data(63 downto 56)     <= DATA_TYPE;
                        o_data_valid(3 downto 2) <= "10"; -- header
                        -- 3. link
                        -- x"7C"
                        -- TS(15:0)
                        o_data(71 downto 64)     <= K28_3;
                        -- lower 4 bit of hit data are zero
                        o_data(75 downto 72)     <= "0000";
                        -- TODO: add tiles
                        if ( DATA_TYPE = x"01" ) then
                            -- get TS(10:4)
                            o_data(82 downto 76)     <= i_data(22 downto 16);
                            -- set TS(15:11)
                            o_data(87 downto 83)     <= TS(15 downto 11);
                        elsif ( DATA_TYPE = x"02" ) then
                            -- get TS(13:4)
                            o_data(85 downto 76)     <= i_data(25 downto 16);
                            -- set TS(15:14)
                            o_data(87 downto 86)     <= i_data(15 downto 14);
                        end if;
                        o_data(95 downto 88)     <= (others => '0');
                        o_data_valid(5 downto 4) <= "10"; -- header
                        -- 4. link
                        -- set rest of the overflow to zero
                        o_data(127 downto 120)   <= (others => '0');
                        -- or'd overflow from all subheaders
                        o_data(119 downto 104)   <= i_data(15 downto 0);
                        o_data(103 downto 96)    <= K28_3;
                        o_data_valid(7 downto 6) <= "10"; -- header
                        -- 5. link
                        -- set rest of the overflow to zero
                        o_data(143 downto 136)   <= (others => '0');
                        o_data(135 downto 128)   <= K28_3;
                        o_data_valid(9 downto 8) <= "10"; -- header
                        -- 6. - 8. link
                        FOR I in NLINKS - 1 downto 5 LOOP
                            o_data(I * 32 + 31 downto I * 32)    <= x"000000" & K28_3;
                            o_data_valid(I * 2 + 1 downto I * 2) <= "10"; -- header
                        END LOOP;
                    end if;

                when delay =>
                    merge_state <= hit;

                -- hits after alignment
                -- 1. hit  =  37 downto   0
                -- 2. hit  =  75 downto  38
                -- 3. hit  = 113 downto  76
                -- 4. hit  = 151 downto 114
                -- 5. hit  = 189 downto 152
                -- 6. hit  = 227 downto 190
                when hit =>
                    -- send out hits if fifo is not empty
                    if ( i_empty = '0' ) then
                        o_data_valid<= (others => '1'); -- hits
                        hit_reg     <= (others => '1');
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
                    end if;

                when get_tr =>
                    -- send out data if tr is there
                    -- every link is getting K.28.4 = 9C for tr
                    merge_state             <= wait_for_pre;
                    FOR I in NLINKS - 1 downto 0 LOOP
                        o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_4;
                        o_data_valid(I * 2 + 1 downto I * 2)     <= "01"; -- trailer
                    END LOOP;

                when error_state =>
                    o_data(7 downto 0)  <= x"7C";
                    o_data(8)           <= hit_reg(12); -- error_gtime1
                    o_data(9)           <= hit_reg(13); -- error_gtime2
                    o_data(10)          <= hit_reg(14); -- error_shtime
                    o_data(11)          <= hit_reg(15); -- error_merger
                    o_data(31 downto 12)<= x"0000" & "0000"; -- free bits
                    o_data_valid(1 downto 0) <= "01"; -- trailer
                    FOR I in NLINKS - 2 downto 0 LOOP
                        o_data(I * 32 + 31 downto I * 32)   <= x"000000" & K28_6;
                        o_data_valid(I * 2 + 1 downto I * 2)     <= "01"; -- trailer
                    END LOOP;
                    merge_state <= get_tr;

                when others =>
                    merge_state <= wait_for_pre;
                    hit_reg     <= (others => '0');
                    hit_reg_cnt <= 0;

            end case;

    end if;
    end process;

end architecture;
