-------------------------------------------------------------
--
-- This is the Data unpacker for MuPix10 in the online repo 
-- THIS IS NOT IDENTICAL TO MUPIX8_DAQ Data_unpacker !!!
-- THIS IS NOT THE DETECTORFPGA Data_unpacker !!!
--
-- Martin Mueller, Oktober 2020
-- muellem@uni-mainz.de
-- 
-- derived from mupix8_daq data_unpacker
-- Sebastian Dittmeier, September 2017
-- Ann-Kathrin Perrevoort, April 2017
--------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

use work.mupix_registers.all;
use work.mupix.all;
use work.mudaq.all;

entity data_unpacker is 
    generic (
        COARSECOUNTERSIZE   : integer   := 32;
        LVDS_ID             : integer   := 0--;
    );
    port (
        reset_n             : in  std_logic;
        clk                 : in  std_logic;
        datain              : in  std_logic_vector(7 downto 0);
        kin                 : in  std_logic;
        readyin             : in  std_logic;
        i_mp_readout_mode   : in  std_logic_vector(31 downto 0);
        o_ts                : out std_logic_vector(10 downto 0);
        o_chip_ID           : out std_logic_vector(5 downto 0);
        o_row               : out std_logic_vector(7 downto 0);
        o_col               : out std_logic_vector(7 downto 0);
        o_tot               : out std_logic_vector(5 downto 0);
        o_hit_ena           : out std_logic;
        errorcounter        : out std_logic_vector(31 downto 0)
    );
end data_unpacker;

architecture RTL of data_unpacker is

    signal coarsecounter        : std_logic_vector(COARSECOUNTERSIZE-1 downto 0); -- Gray Counter[7:0] & Binary Counter [23:0]
    signal coarsecounter_ena    : std_logic;
    signal link_flag            : std_logic;

    type state_type is (IDLE, ERROR, COUNTER, LINK, DATA);
    signal NS                   : state_type;

    signal data_i               : std_logic_vector(31 downto 0) := (others => '0');

    signal errorcounter_reg     : std_logic_vector(31 downto 0);

    signal link_i               : std_logic_vector(7 downto 0);
    signal link_reg             : std_logic_vector(7 downto 0);
    signal link_toggle          : std_logic := '0';
    signal data_mode            : std_logic;

    signal cnt4                 : std_logic_vector(1 downto 0);

    signal counter_seen         : std_logic;
    signal coarse_reg           : std_logic;
    signal hit_reg              : std_logic;
    signal link_flag_reg        : std_logic;

    signal ts                   : std_logic_vector(10 downto 0);
    signal ts2                  : std_logic_vector(4 downto 0);
    signal ts_buf               : std_logic_vector(10 downto 0);
    signal ts2_buf              : std_logic_vector(4 downto 0);
    signal row                  : std_logic_vector(7 downto 0);
    signal col                  : std_logic_vector(7 downto 0);
    signal tot                  : std_logic_vector(5 downto 0);
    signal hit_ena              : std_logic;

    signal chip_ID_mode         : std_logic_vector(1 downto 0);
    signal tot_mode             : std_logic_vector(2 downto 0);
    signal invert_TS            : std_logic;
    signal invert_TS2           : std_logic;
    signal gray_TS              : std_logic;
    signal gray_TS2             : std_logic;

    function convert_lvds_to_chip_id (
        lvds_ID       : integer;
        chip_ID_mode  : std_logic_vector(1 downto 0)--;
    ) return std_logic_vector is 
        variable chip_id : std_logic_vector(5 downto 0);
    begin
        -- TODO: correct numbering for different feb positions (chip_id_mode's)
        chip_id := std_logic_vector(to_unsigned(lvds_ID, 6));
        return chip_id;
    end;

    function convert_row (
        i_row       : std_logic_vector(8 downto 0)--;
    ) return std_logic_vector is 
        variable row : std_logic_vector(7 downto 0);
        variable tmp : std_logic_vector(8 downto 0);
    begin
        if (unsigned(i_row)>380) then
            tmp     := std_logic_vector(499-unsigned(i_row));
            if (i_row(0)='0') then
                row := std_logic_vector(unsigned(tmp(8 downto 1)) + 60);
            else 
                row := tmp(8 downto 1);
            end if;
        elsif (i_row(8)='1') then
            tmp     := std_logic_vector(380-unsigned(i_row));
            if (i_row(0)='1') then
                row := std_logic_vector(unsigned(tmp(8 downto 1)) + 63);
            else 
                row := tmp(8 downto 1);
            end if;
        elsif (unsigned(i_row)>124) then
            tmp     := std_logic_vector(255-unsigned(i_row));
            if (i_row(0)='0') then
                row := std_logic_vector(unsigned(tmp(8 downto 1)) + 119 + 66);
            else 
                row := std_logic_vector(unsigned(tmp(8 downto 1)) + 119);
            end if;
        else
            tmp     := std_logic_vector(124-unsigned(i_row));
            if (i_row(0)='1') then
                row := std_logic_vector(unsigned(tmp(8 downto 1)) + 125 + 63);
            else
                row := std_logic_vector(unsigned(tmp(8 downto 1)) + 125);
            end if;
        end if;
        return row;
    end;

    function convert_col (
        i_col       : std_logic_vector(6 downto 0);
        i_row       : std_logic_vector(8 downto 0)--;
    ) return std_logic_vector is 
        variable col : std_logic_vector(7 downto 0);
    begin
        if (unsigned(i_row)> 380 or (i_row(8) = '0' and unsigned(i_row)>124)) then
            col := i_col & '1';
        else
            col := i_col & '0';
        end if;
        return col;
    end;

    function calc_tot (
        ts2       : std_logic_vector( 4 downto 0);
        ts        : std_logic_vector(10 downto 0);
        tot_mode  : std_logic_vector( 2 downto 0)--;
    ) return std_logic_vector is 
        variable tot : std_logic_vector(5 downto 0);
    begin
        -- TODO: calc. something here
        tot := '0' & ts2;
        return tot;
    end;

begin

    o_ts    <= ts_buf;
    o_tot   <= calc_tot(ts2_buf,ts_buf,tot_mode);

    chip_ID_mode    <= i_mp_readout_mode(CHIP_ID_MODE_RANGE);
    tot_mode        <= i_mp_readout_mode(TOT_MODE_RANGE);
    invert_TS       <= i_mp_readout_mode(INVERT_TS_BIT);
    invert_TS2      <= i_mp_readout_mode(INVERT_TS2_BIT);
    gray_TS         <= i_mp_readout_mode(GRAY_TS_BIT);
    gray_TS2        <= i_mp_readout_mode(GRAY_TS2_BIT);

    errorcounter     <= errorcounter_reg;

    fsmProc: process (clk, reset_n)
    begin
        if reset_n = '0' then
            NS <= IDLE;

            hit_ena             <= '0';
            coarsecounter       <= (others => '0');
            coarsecounter_ena   <= '0';
            link_flag           <= '0';

            link_flag_reg       <= '0';
            coarse_reg          <= '0';
            hit_reg             <= '0';

            link_i              <= (others =>'0');
            link_reg            <= (others =>'0');
            link_toggle         <= '0';

            errorcounter_reg    <= (others => '0');

            cnt4                <= "00";
            data_mode           <= '0'; -- indicates if all counter mode or actual hit data
            counter_seen        <= '0';
        elsif rising_edge(clk) then

            link_flag_reg       <= '0';
            coarse_reg          <= '0';
            hit_reg             <= '0';

            hit_ena             <= hit_reg;
            coarsecounter_ena   <= coarse_reg;
            link_flag           <= link_flag_reg;

            coarsecounter       <= data_i(7 downto 0) & data_i(31 downto 8); -- gray counter & binary counter

            if(hit_reg = '1')then
                -- The data arrives from MuPix10: TS2[4:0] TS1[10:0] Col[6:0] Row[8:0]
                    -- TS2 = data_i(31 downto 27)
                    -- TS1 = data_i(26 downto 16)
                    -- Col = data_i(15 downto 9)
                    -- Row = data_i(8 downto 0)
                -- We present to the outside:
                ts              <= data_i(26 downto 16);
                o_chip_ID       <= convert_lvds_to_chip_id(LVDS_ID,chip_ID_mode);
                row             <= convert_row(data_i(8 downto 0));
                col             <= convert_col(data_i(15 downto 9),data_i(8 downto 0));
                ts2             <= data_i(31 downto 27);
            end if;

            if(readyin = '0')then
                link_reg        <= (others => '0');
                link_toggle     <= '0';
                NS              <= IDLE;
            else
                data_i          <= data_i(23 downto 0) & datain;

                    case NS is
                    
                        when IDLE =>
                            cnt4            <= "00";
                            link_toggle     <= '0';
                            if kin = '0'then                                   -- counter mode
                                if(data_mode = '1' and counter_seen = '1')then -- we expect valid hit data here after LINK ID and counter was seen
                                    NS      <= DATA;
                                    cnt4    <= "01";
                                else
                                    NS      <= COUNTER;
                                    cnt4    <= "01";
                                end if;
                            elsif kin = '1' and datain = K28_0 then -- data mode
                                NS <= LINK;
                            elsif kin = '1' and datain = K28_5 then
                                NS <= IDLE; 
                            else
                                NS <= ERROR;
                            end if;


                        when COUNTER =>
                            if kin = '0' then
                                cnt4                <= cnt4 + '1';
                                if cnt4 = "11" then
                                    coarse_reg      <= '1';
                                    counter_seen    <= '1';
                                    NS              <= IDLE;
                                end if;
                            elsif kin = '1' and datain = K28_5 then    --and counter_int = 3 then
                                NS <= IDLE;
                            else
                                NS <= ERROR;
                            end if;

                        when LINK =>
                            if kin = '0' then
                                if link_toggle = '0' then
                                    link_reg            <= datain;
                                elsif link_reg = datain then
                                    link_i              <= link_reg;
                                    link_flag_reg       <= '1';
                                    link_toggle         <= '0';
                                    NS                  <= IDLE;
                                    data_mode           <= '1';
                                    counter_seen        <= '0';
                                else
                                    NS                <= ERROR;
                                end if;
                            elsif kin = '1' and datain = K28_0 then
                                link_toggle <= '1';
                            else
                                NS <= ERROR;
                            end if;

                        when DATA =>
                        
                            if kin = '0' then
                                cnt4        <= cnt4 + '1';
                                if cnt4 = "11" then
                                    hit_reg <= '1';
                                    NS      <= IDLE;
                                end if;
                            elsif kin = '1' and datain = K28_5 then
                                NS          <= IDLE;
                            else
                                NS          <= ERROR;
                            end if;

                        when ERROR =>
                            errorcounter_reg <= errorcounter_reg + '1';
                            if ( kin = '1' and datain = K28_5 ) then
                                NS <= IDLE;
                            else
                                NS <= ERROR;
                            end if;
                        
                        when others =>
                            NS <= IDLE;
                    end case; -- NS
            end if; -- readyin

        end if;

    end process;

    e_mp_sc_rm: mp_sc_removal
    port map(
        i_reset_n               => reset_n,
        i_clk                   => clk, 
        i_sc_active             => '1',
        i_new_block             => link_flag,
        i_hit                   => hit_out_i,
        i_hit_ena               => hit_ena_i,
        i_coarsecounters_ena    => coarsecounter_ena,
        o_hit                   => hit_out_o,
        o_hit_ena               => hit_ena_o--,
    );


    degray_single : work.hit_ts_conversion
    port map(
        reset_n     => reset_n,
        clk         => clk,
        invert_TS   => invert_TS,
        invert_TS2  => invert_TS2,
        gray_TS     => gray_TS,
        gray_TS2    => gray_TS2,
        
        o_ts        => ts_buf,
        o_row       => o_row,
        o_col       => o_col,
        o_ts2       => ts2_buf,
        o_hit_ena   => o_hit_ena,
        
        i_ts        => ts,
        i_row       => row,
        i_col       => col,
        i_ts2       => ts2,
        i_hit_ena   => hit_ena--,
    );

end RTL;
