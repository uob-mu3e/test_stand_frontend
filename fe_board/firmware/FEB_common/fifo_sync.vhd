library ieee;
use ieee.std_logic_1164.all;

entity fifo_sync is
generic (
    DATA_WIDTH_g : positive := 8;
    FIFO_ADDR_WIDTH_g : positive := 3--;
);
port (
    o_rdata     : out   std_logic_vector(DATA_WIDTH_g-1 downto 0);
    i_rclk      : in    std_logic;

    i_wdata     : in    std_logic_vector(DATA_WIDTH_g-1 downto 0);
    i_wclk      : in    std_logic;

    i_fifo_aclr : in    std_logic--;
);
end entity;

architecture rtl of fifo_sync is

    signal rempty, wfull : std_logic;
    signal rdata : std_logic_vector(DATA_WIDTH_g-1 downto 0);

begin

    e_fifo : entity work.ip_dcfifo
    generic map (
        ADDR_WIDTH => FIFO_ADDR_WIDTH_g,
        DATA_WIDTH => DATA_WIDTH_g--,
    )
    port map (
        rdempty     => rempty,
        rdreq       => not rempty,
        q           => rdata,
        rdusedw     => open,
        rdclk       => i_rclk,

        wrfull      => wfull,
        wrreq       => not wfull,
        data        => i_wdata,
        wrusedw     => open,
        wrclk       => i_wclk,

        aclr        => i_fifo_aclr--,
    );

    process(i_rclk, i_fifo_aclr, i_wdata)
    begin
    if ( i_fifo_aclr = '1' ) then
        o_rdata <= i_wdata;
        --
    elsif rising_edge(i_rclk) then
        if ( rempty = '0' ) then
            o_rdata <= rdata;
        end if;
        --
    end if;
    end process;

end architecture;
