library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity tb_fifo_reg is
end entity;

architecture arch of tb_fifo_reg is

    constant CLK_MHZ : real := 1000.0; -- MHz
    signal clk, clk_fast, reset_n : std_logic := '0';

    signal data, cnt : std_logic_vector(7 downto 0);
    signal q, q_reg, o_data, q_reg_reg : std_logic_vector(15 downto 0);
    signal a, b : std_logic_vector(7 downto 0);
    signal rdreq, reg_rdreq, reg_reg_rdreq, reg_full, wrreq, rdempty, wrfull, reg_empty, reg_reg_empty, reg_reg_full  : std_logic;

begin

    clk     <= not clk after (0.5 us / CLK_MHZ);
    clk_fast<= not clk_fast after (0.1 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);
    
    write : process(clk, reset_n)
    begin
        if ( reset_n = '0' ) then
            data <= (others => '0');
            cnt <= (others => '0');
            wrreq<= '0';
        elsif ( rising_edge(clk) ) then
            if ( wrfull = '0' ) then
                wrreq <= '1';
                cnt <= cnt + '1';
                data <= cnt + '1';
            end if;
        end if;
    end process;
    
    e_link_fifo : entity work.ip_scfifo
    generic map(
        RAM_OUT_REG    => "ON"--,
    )
    port map (
        sclr    => not reset_n,
        data    => data,
        clock   => clk,
        rdreq   => rdreq,
        wrreq   => wrreq,
        q       => open,
        empty => rdempty,
        full  => wrfull--,
    );
    
    rdreq <= '1' when rdempty = '0' and reg_full = '0' else '0';
    reg_rdreq <= '1' when reg_empty = '0' and reg_reg_full = '0' else '0';

    process(clk, reset_n)
    begin
    if ( reset_n = '0' ) then
        reg_empty <= '1';
        reg_full  <= '0';
        q_reg     <= (others => '0');
        reg_reg_empty <= '1';
        reg_reg_full  <= '0';
        q_reg_reg <= (others => '0');
        --
    elsif ( rising_edge(clk) ) then

        if ( rdreq = '1' ) then
            q_reg       <= q;
            reg_full    <= '1';
            reg_empty   <= '0';
        end if;

        if ( reg_rdreq = '1' ) then
            q_reg_reg <= q_reg;
            reg_full  <= '0';
            reg_empty <= '1';

            reg_reg_full    <= '1';
            reg_reg_empty   <= '0';
        end if;

        if ( reg_reg_rdreq = '1' ) then
            reg_reg_full <= '0';
            reg_reg_empty <= '1';
        end if;

    end if;
    end process;

    reg_reg_rdreq <= '1' when reg_reg_empty = '0' else '0';
    
    a <= q_reg_reg(7 downto 0);
    b <= q_reg_reg(15 downto 8);

end architecture;
