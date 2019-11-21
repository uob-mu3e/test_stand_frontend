library ieee;
use ieee.std_logic_1164.all;

entity fifo_sync is
generic (
    RDATA_RESET_g : std_logic_vector;
    FIFO_ADDR_WIDTH_g : positive := 3--;
);
port (
    -- read domain
    o_rdata     : out   std_logic_vector(RDATA_RESET_g'range);
    i_rreset_n  : in    std_logic;
    i_rclk      : in    std_logic;

    -- write domain
    i_wdata     : in    std_logic_vector(RDATA_RESET_g'range);
    i_wreset_n  : in    std_logic;
    i_wclk      : in    std_logic--;
);
end entity;

architecture rtl of fifo_sync is

    signal rempty, wfull, we : std_logic;
    signal rdata, wdata : std_logic_vector(RDATA_RESET_g'range);

begin

    e_fifo : entity work.ip_dcfifo
    generic map (
        ADDR_WIDTH => FIFO_ADDR_WIDTH_g,
        DATA_WIDTH => RDATA_RESET_g'length--,
    )
    port map (
        rdempty     => rempty,
        rdreq       => not rempty,
        q           => rdata,
        rdusedw     => open,
        rdclk       => i_rclk,

        wrfull      => wfull,
        wrreq       => we,
        data        => wdata,
        wrusedw     => open,
        wrclk       => i_wclk,

        aclr        => not i_rreset_n--,
    );

    process(i_rclk, i_rreset_n)
    begin
    if ( i_rreset_n = '0' ) then
        o_rdata <= RDATA_RESET_g;
        --
    elsif rising_edge(i_rclk) then
        if ( rempty = '0' ) then
            o_rdata <= rdata;
        end if;
        --
    end if;
    end process;

    process(i_wclk, i_wreset_n)
    begin
    if ( i_wreset_n = '0' ) then
        we <= '0';
        --
    elsif rising_edge(i_wclk) then
        wdata <= i_wdata;
        we <= not wfull;
        --
    end if;
    end process;

end architecture;
