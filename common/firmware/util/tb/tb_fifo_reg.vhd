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
    signal q, q_reg, o_data : std_logic_vector(15 downto 0);
    signal a, b : std_logic_vector(7 downto 0);
    signal rdreq, rdreq_reg, wrfull_reg, wrreq, rdempty, wrfull, rdempty_reg, last_rdreq_reg : std_logic;

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
            data <= cnt + '1';
        end if;
    end process;
    
    e_link_fifo : entity work.ip_dcfifo_mixed_widths
    generic map(
        ADDR_WIDTH_w    => 8,
        DATA_WIDTH_w    => 8,
        ADDR_WIDTH_r    => 4,
        DATA_WIDTH_r    => 16,
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
    
    rdreq <= '1' when rdempty = '0' and wrfull_reg = '0' else '0';
    
    e_reg_fifo : entity work.reg_fifo
        generic map (
            g_WIDTH    => 16,
            g_DEPTH    => 8,
            g_AF_LEVEL => 6,
            g_AE_LEVEL => 2--,
    )
    port map (
        i_rst_sync => not reset_n,
        i_clk      => clk,
    
        -- FIFO Write Interface
        i_wr_en    => rdreq,
        i_wr_data  => q,
        o_af       => wrfull_reg,
        o_full     => open,
    
        -- FIFO Read Interface
        i_rd_en    => rdreq_reg,
        o_rd_data  => q_reg,
        o_ae       => rdempty_reg,
        o_empty    => open--,
    );
 
    rdreq_reg <= '1' when rdempty_reg = '0' else '0';
    
    a <= q_reg(7 downto 0);
    b <= q_reg(15 downto 8);

end architecture;
