--

library ieee;
use ieee.std_logic_1164.all;

entity ip_dcfifo_mixed_widths is
generic (
    ADDR_WIDTH_w    : positive := 8;
    DATA_WIDTH_w    : positive := 8;
    ADDR_WIDTH_r    : positive := 8;
    DATA_WIDTH_r    : positive := 8;
    SHOWAHEAD       : string := "ON";
    DEVICE          : string := "Arria 10"--;
);
port (
    aclr        : in    std_logic := '0';
    data        : in    std_logic_vector(DATA_WIDTH_w-1 downto 0);
    rdclk       : in    std_logic;
    rdreq       : in    std_logic;
    wrclk       : in    std_logic;
    wrreq       : in    std_logic;
    q           : out   std_logic_vector(DATA_WIDTH_r-1 downto 0);
    rdempty     : out   std_logic;
    rdusedw     : out   std_logic_vector(ADDR_WIDTH_r-1 downto 0);
    wrfull      : out   std_logic;
    wrusedw     : out   std_logic_vector(ADDR_WIDTH_w-1 downto 0)
);
end entity;

architecture arch of ip_dcfifo_mixed_widths is
begin

    e_ip_dcfifo_v2 : entity work.ip_dcfifo_v2
    generic map (
        g_WADDR_WIDTH => ADDR_WIDTH_w,
        g_WDATA_WIDTH => DATA_WIDTH_w,
        g_RADDR_WIDTH => ADDR_WIDTH_r,
        g_RDATA_WIDTH => DATA_WIDTH_r,
        g_SHOWAHEAD => SHOWAHEAD,
        g_WREG_N => 0,
        g_RREG_N => 0,
        g_DEVICE_FAMILY => DEVICE--,
    )
    port map (
        i_wdata     => data,
        i_we        => wrreq,
        o_wfull     => wrfull,
        o_wusedw    => wrusedw,
        i_wclk      => wrclk,

        o_rdata     => q,
        i_rack      => rdreq,
        o_rempty    => rdempty,
        o_rusedw    => rdusedw,
        i_rclk      => rdclk,

        i_reset_n   => not aclr--,
    );

end architecture;
