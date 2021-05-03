library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity tb_fifo_reg is
end entity;

architecture arch of tb_fifo_reg is

    constant CLK_MHZ : real := 1000.0; -- MHz
    signal clk, clk_fast, reset_n : std_logic := '0';

    signal data, cnt : std_logic_vector(31 downto 0);
    signal q, q_reg, o_data : std_logic_vector(63 downto 0);
    signal rdreq, rdreq_reg, wrfull_reg, wrreq, rdempty, wrfull, rdempty_reg : std_logic;

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
            wrreq <= '1';
            cnt <= cnt + '1';
            data <= cnt;
        end if;
    end process;
    
    e_link_fifo : entity work.ip_dcfifo_mixed_widths
    generic map(
        ADDR_WIDTH_w    => 8,
        DATA_WIDTH_w    => 32,
        ADDR_WIDTH_r    => 8,
        DATA_WIDTH_r    => 64,
        DEVICE          => "Arria 10"--,
    )
    port map (
        aclr    => not reset_n,
        data    => data,
        rdclk   => clk,
        rdreq   => rdreq,
        wrclk   => clk,
        wrreq   => wrreq,
        q       => q,
        rdempty => rdempty,
        wrfull  => wrfull--,
    );
    
    reg : process(clk, reset_n)
    begin
        if ( reset_n = '0' ) then
            q_reg       <= (others => '0');
            o_data      <= (others => '0');
            rdreq       <= '0';
            wrfull_reg  <= '0';
        elsif ( rising_edge(clk) ) then
            rdreq <= '0';
            if ( rdempty = '0' and (wrfull_reg = '0' or rdreq_reg = '1') ) then
                rdreq       <= '1';
                q_reg       <= q;
                wrfull_reg  <= '1';
            end if;
            
            if ( rdreq_reg = '1' ) then
                o_data <= q_reg;
                wrfull_reg  <= '0';
            end if;
            
            if ( rdempty = '0' and rdreq_reg = '1' ) then
                rdempty_reg <= '0';
            elsif ( rdempty = '0' and wrfull_reg = '0' ) then
                rdempty_reg <= '0';
            elsif ( rdreq_reg = '1' ) then
                rdempty_reg <= '1';
            end if;
            
        end if;
    end process;
    
    read : process(clk, reset_n)
    begin
        if ( reset_n = '0' ) then
            rdreq_reg <= '0';
        elsif ( rising_edge(clk) ) then
            if ( rdempty_reg = '0' ) then
                rdreq_reg <= '1';
            else
                rdreq_reg <= '0';
            end if;
        end if;
    end process;

end architecture;
