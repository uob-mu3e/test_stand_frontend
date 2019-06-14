library ieee;
use ieee.std_logic_1164.all;

entity rx_align is
    generic (
        Nb : positive := 4;
        K : std_logic_vector(7 downto 0) := X"BC"--;
    );
    port (
        data    :   out std_logic_vector(8*Nb-1 downto 0);
        datak   :   out std_logic_vector(Nb-1 downto 0);

        lock    :   out std_logic;

        datain  :   in  std_logic_vector(8*Nb-1 downto 0);
        datakin :   in  std_logic_vector(Nb-1 downto 0);

        syncstatus      :   in  std_logic_vector(Nb-1 downto 0);
        patterndetect   :   in  std_logic_vector(Nb-1 downto 0);
        enapatternalign :   out std_logic;

        errdetect   :   in  std_logic_vector(Nb-1 downto 0);
        disperr     :   in  std_logic_vector(Nb-1 downto 0);

        rst_n   :   in  std_logic;
        clk     :   in  std_logic--;
    );
end entity;

architecture arch of rx_align is

    signal data_i : std_logic_vector(63 downto 0);
    signal datak_i : std_logic_vector(7 downto 0);

    signal lock_i : std_logic;
    signal pattern_i : std_logic_vector(3 downto 0);

    signal enapatternalign_rst_n : std_logic;

    -- quality counter
    -- - increment if good pattern
    -- - decrement if error
    signal quality : integer range 0 to 7;

begin

    lock <= lock_i;

    process(clk, rst_n)
    begin
    if ( rst_n = '0' ) then
        data_i <= (others => '0');
        datak_i <= (others => '0');
        --
    elsif rising_edge(clk) then
        data_i <= (others => '0');
        datak_i <= (others => '0');
        -- assume link is LSB first
        data_i(31 downto 0) <= data_i(63 downto 32);
        datak_i(3 downto 0) <= datak_i(7 downto 4);
        data_i(8*Nb-1 + 32 downto 32) <= datain;
        datak_i(Nb-1 + 4 downto 4) <= datakin;
        --
    end if;
    end process;

    i_enapatternalign_rst_n : entity work.watchdog
    port map (
        d(0) => '0',
        rstout_n => enapatternalign_rst_n,
        rst_n => rst_n and not lock_i, -- reset if locked
        clk => clk--,
    );

    process(clk, rst_n)
        variable error_v : boolean;
        --
    begin
    if ( rst_n = '0' ) then
        lock_i <= '0';
        pattern_i <= "0000";
        quality <= 0;
        enapatternalign <= '0';
        --
    elsif rising_edge(clk) then
        error_v := false;

        -- generate rising edge if not locked
        -- set to '0' for one clock cycle if not locked for long time
        enapatternalign <= not lock_i and enapatternalign_rst_n;

        if ( patterndetect = "0000" ) then
            -- idle
        elsif ( patterndetect = "0001" or patterndetect = "0010" or patterndetect = "0100" or patterndetect = "1000" ) then
            if ( pattern_i /= "0000" and pattern_i /= patterndetect) then
                -- unexpected pattern
                error_v := true;
            end if;

            -- require one control symbol
            if ( patterndetect /= datakin ) then
                error_v := true;
            end if;

            -- check control symbol
            if ( patterndetect = "0001" and datain(7 downto 0) /= K ) then
                error_v := true;
            end if;
            if ( patterndetect = "0010" and datain(15 downto 8) /= K ) then
                error_v := true;
            end if;
            if ( patterndetect = "0100" and datain(23 downto 16) /= K ) then
                error_v := true;
            end if;
            if ( patterndetect = "1000" and datain(31 downto 24) /= K ) then
                error_v := true;
            end if;
        else
            -- invalid pattern
            error_v := true;
        end if;

        if ( error_v
            or syncstatus /= (syncstatus'range => '1')
            or errdetect /= (errdetect'range => '0')
            or disperr /= (disperr'range => '0')
        ) then
            if ( quality = 0 ) then
                -- not locked
                lock_i <= '0';
                pattern_i <= "0000";
            else
                quality <= quality - 1;
            end if;
        elsif ( patterndetect /= "0000" ) then
            -- good pattern
            if ( quality = 7 ) then
                -- locked
                lock_i <= '1';
                pattern_i <= patterndetect;
            else
                quality <= quality + 1;
            end if;
        end if;

        data <= (others => '-');
        datak <= (others => '-');

        -- align such that LSB is K
        case pattern_i is
        when "0001" =>
            data <= data_i(8*Nb-1 + 0 downto 0);
            datak <= datak_i(Nb-1 + 0 downto 0);
        when "0010" =>
            data <= data_i(8*Nb-1 + 8 downto 8);
            datak <= datak_i(Nb-1 + 1 downto 1);
        when "0100" =>
            data <= data_i(8*Nb-1 + 16 downto 16);
            datak <= datak_i(Nb-1 + 2 downto 2);
        when "1000" =>
            data <= data_i(8*Nb-1 + 24 downto 24);
            datak <= datak_i(Nb-1 + 3 downto 3);
        when others =>
            null;
        end case;

        --
    end if; -- rising_edge
    end process;

end architecture;
