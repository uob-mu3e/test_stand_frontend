--

library ieee;
use ieee.std_logic_1164.all;

entity ip_dcfifo IS
generic (
    ADDR_WIDTH  : positive := 8;
    DATA_WIDTH  : positive := 8;
    SHOWAHEAD   : string := "ON";
    OVERFLOW    : string := "ON";
    DEVICE      : string := "Arria 10"--;
);
port (
    aclr        : in    std_logic := '0';
    data        : in    std_logic_vector(DATA_WIDTH-1 downto 0);
    rdclk       : in    std_logic;
    rdreq       : in    std_logic;
    wrclk       : in    std_logic;
    wrreq       : in    std_logic;
    q           : out   std_logic_vector(DATA_WIDTH-1 downto 0);
    rdempty     : out   std_logic;
    rdusedw     : out   std_logic_vector(ADDR_WIDTH-1 downto 0);
    wrfull      : out   std_logic;
    wrusedw     : out   std_logic_vector(ADDR_WIDTH-1 downto 0)--;
);
end entity;

architecture arch of ip_dcfifo is
begin

    e_ip_dcfifo_v2 : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => ADDR_WIDTH,
        g_DATA_WIDTH => DATA_WIDTH,
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
