-----------------------------------------------------------------------------
-- PCIe to register application, pcie writeable registers
--
-- Niklaus Berger, Heidelberg University
-- nberger@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;

entity pcie_writeable_registers is
port (
    o_writeregs_B   : out   reg32array_pcie;
    o_regwritten_B  : out   std_logic_vector(63 downto 0);
    i_clk_B         : in    std_logic := '0';

    o_writeregs_C   : out   reg32array_pcie;
    o_regwritten_C  : out   std_logic_vector(63 downto 0);
    i_clk_C         : in    std_logic := '0';

    local_rstn      : in    std_logic;
    refclk          : in    std_logic;

    -- from IF
    i_rx_st_data0   : in    std_logic_vector(255 downto 0);
    i_rx_st_eop0    : in    std_logic;
    i_rx_st_sop0    : in    std_logic;
    o_rx_st_ready0  : out   std_logic;
    i_rx_st_valid0  : in    std_logic;
    i_rx_bar        : in    std_logic;

    -- registers
    writeregs       : out   reg32array_pcie;
    regwritten      : out   std_logic_vector(63 downto 0);

    -- to response engine
    readaddr        : out   std_logic_vector(5 downto 0);
    readlength      : out   std_logic_vector(9 downto 0);
    header2         : out   std_logic_vector(31 downto 0);
    readen          : out   std_logic;
    inaddr32_w      : out   STD_LOGIC_VECTOR (31 DOWNTO 0)
);
end entity;

architecture RTL of pcie_writeable_registers is

    signal rx_st_data0 : std_logic_vector(i_rx_st_data0'range);
    signal rx_st_sop0 : std_logic;
    signal rx_bar : std_logic;

    type receiver_state_type is (reset, waiting);
    signal state : receiver_state_type;

    signal inaddr32 : std_logic_vector(31 downto 0);
    signal regaddr  : std_logic_vector(5 downto 0);

    -- Decoding PCIe TLP headers
    signal fmt : std_logic_vector(1 downto 0);
    signal ptype : std_logic_vector(4 downto 0);
    signal tc : std_logic_vector(2 downto 0);
    signal td : std_logic;
    signal ep : std_logic;
    signal attr : std_logic_vector(1 downto 0);
    signal plength : std_logic_vector(9 downto 0);
    signal plength_reg : std_logic_vector(9 downto 0);
    signal fdw_be : std_logic_vector(3 downto 0);
    signal ldw_be : std_logic_vector(3 downto 0);
    signal ldw_be_reg : std_logic_vector(3 downto 0);

    signal word3 : std_logic_vector(31 downto 0);
    signal word4 : std_logic_vector(31 downto 0);

    signal be3 : std_logic;
    signal be4 : std_logic;

    signal be3_prev : std_logic;
    signal be4_prev : std_logic;

    signal addr3: std_logic_vector(5 downto 0);
    signal addr4: std_logic_vector(5 downto 0);

    -- registers
    signal writeregs_r : reg32array_pcie;
    signal regwritten_r : std_logic_vector(63 downto 0);

    signal writeregs_B : reg32array_pcie;
    signal regwritten_B : std_logic_vector(63 downto 0);
    signal writeregs_B_reset_n : std_logic;
    signal writeregs_B_fifo_wdata, writeregs_B_fifo_rdata : std_logic_vector(37 downto 0);
    signal writeregs_B_fifo_rempty : std_logic;

    signal writeregs_C : reg32array_pcie;
    signal regwritten_C : std_logic_vector(63 downto 0);
    signal writeregs_C_reset_n : std_logic;
    signal writeregs_C_fifo_wdata, writeregs_C_fifo_rdata : std_logic_vector(37 downto 0);
    signal writeregs_C_fifo_rempty : std_logic;

begin

    process(refclk, local_rstn)
    begin
    if ( local_rstn = '0' ) then
        o_rx_st_ready0 <= '0';
        rx_st_data0 <= (others => '0');
        rx_st_sop0 <= '0';
        rx_bar <= '0';
    elsif rising_edge(refclk) then
        if ( state = reset ) then
            o_rx_st_ready0 <= '0';
        else
            o_rx_st_ready0 <= '1';
        end if;
        rx_st_data0 <= i_rx_st_data0;
        rx_st_sop0 <= i_rx_st_sop0;
        rx_bar <= i_rx_bar;
    end if;
    end process;

    -- Endian chasing for addresses
    inaddr32 <= rx_st_data0(95 downto 66) & "00";
    --inaddr32 <= rx_st_data0(95 downto 74) & "1" & rx_st_data0(72 downto 66) & "00";

    regaddr <= inaddr32(7 downto 2);

    -- decode TLP
    fmt <= rx_st_data0(30 downto 29);
    ptype <= rx_st_data0(28 downto 24);
    tc <= rx_st_data0(22 downto 20);
    td <= rx_st_data0(15);
    ep <= rx_st_data0(14);
    attr <= rx_st_data0(13 downto 12);
    plength <= rx_st_data0(9 downto 8) & rx_st_data0(7 downto 0);
    fdw_be <= rx_st_data0(35 downto 32);
    ldw_be <= rx_st_data0(39 downto 36);

    writeregs <= writeregs_r;

    process(local_rstn, refclk)
    begin
    if(local_rstn = '0') then
        state <= reset;
        writeregs_r <= (others => (others => '0'));
        regwritten <= (others => '0');
        --
    elsif (refclk'event and refclk = '1') then
        regwritten <= regwritten_r;

        readen <= '0';
        word3 <= rx_st_data0(127 downto 96);
        word4 <= rx_st_data0(159 downto 128);
        regwritten_r <= (others => '0');

        -- do the actual writing
        if ( be3 = '1' ) then
            writeregs_r(TO_INTEGER(UNSIGNED(addr3))) <= word3;
            regwritten_r(TO_INTEGER(UNSIGNED(addr3))) <= '1';
        end if;
        if ( be4 = '1' ) then
            writeregs_r(TO_INTEGER(UNSIGNED(addr4))) <= word4;
            regwritten_r(TO_INTEGER(UNSIGNED(addr4))) <= '1';
        end if;

        case state is
        when reset =>
            be3 <= '0';
            be4 <= '0';

            state <= waiting;
            --
        when waiting =>
            be3 <= '0';
            be4 <= '0';

            if(rx_st_sop0 = '1' and rx_bar = '1') then --  and inaddr32 = x"fb480040"
                if(fmt = "10" and ptype = "00000") then -- 32 bit memory write request
                    if(inaddr32(2) = '1') then -- Unaligned write, first data word at word3
                        addr3 <= regaddr;
                        if(fdw_be /= "0000") then
                            be3 <= '1';
                        else
                            be3 <= '0';
                        end if;
                        state <= waiting;
                    else -- aligned write, first data word at word4
                        addr4 <= regaddr;
                        if(fdw_be /= "0000") then
                            be4 <= '1';
                        else
                            be4 <= '0';
                        end if;
                        state <= waiting;
                    end if; -- if aligned
                elsif(fmt = "00" and ptype = "00000") then -- 32 bit memory read request
                    inaddr32_w <= rx_st_data0(95 downto 66) & "00";
                    readaddr <= regaddr;
                    readlength <= plength;
                    header2 <= rx_st_data0(63 downto 32);
                    readen <= '1';
                    state <= waiting;
                end if; -- 32 bit write/read request
            end if; -- if Start of Packet

            state <= waiting;
            --
        end case;
        --
    end if;
    end process;



    o_writeregs_B <= writeregs_B;
    o_regwritten_B <= regwritten_B;

    process(refclk)
    begin
    if(rising_edge(refclk)) then
        be3_prev <= be3;
        be4_prev <= be4;
    end if;
    end process;

    writeregs_B_fifo_wdata <=
        ( addr3 & word3 ) when ( be3_prev = '1' ) else
        ( addr4 & word4 ) when ( be4_prev = '1' ) else
        (others => '0');

    -- sync writeregs writes to i_clk_B clock domain
    e_writeregs_B_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => 4,
        g_DATA_WIDTH => writeregs_B_fifo_wdata'length,
        g_RREG_N => 1--,
    )
    port map (
        i_we        => be3_prev or be4_prev,
        i_wdata     => writeregs_B_fifo_wdata,
        o_wfull     => open,
        i_wclk      => refclk,

        i_rack      => not writeregs_B_fifo_rempty,
        o_rdata     => writeregs_B_fifo_rdata,
        o_rempty    => writeregs_B_fifo_rempty,
        i_rclk      => i_clk_B,

        i_reset_n   => local_rstn--,
    );

    -- writeregs_B_reset_n is several clock cycles longer than local_rstn,
    e_writeregs_B_reset_n : entity work.reset_sync
    port map ( o_reset_n => writeregs_B_reset_n, i_reset_n => local_rstn, i_clk => i_clk_B );

    process(i_clk_B, writeregs_B_reset_n)
    begin
    if ( writeregs_B_reset_n = '0' ) then
        -- note that e_writeregs_B_fifo is driven by local_rstn,
        -- so during writeregs_B_reset_n the write requests are buffered
        -- and the writes are delayed (but not lost)
        writeregs_B <= (others => (others => '0'));
        --
    elsif rising_edge(i_clk_B) then
        if ( writeregs_B_fifo_rempty = '0' ) then
            writeregs_B(to_integer(unsigned(writeregs_B_fifo_rdata(37 downto 32)))) <= writeregs_B_fifo_rdata(31 downto 0);
            regwritten_B(to_integer(unsigned(writeregs_B_fifo_rdata(37 downto 32)))) <= '1';
        end if;
        --
    end if;
    end process;

    o_writeregs_C <= writeregs_C;
    o_regwritten_C <= regwritten_C;

    writeregs_C_fifo_wdata <=
        ( addr3 & word3 ) when ( be3_prev = '1' ) else
        ( addr4 & word4 ) when ( be4_prev = '1' ) else
        (others => '0');

    -- sync writeregs writes to i_clk_C clock domain
    e_writeregs_C_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => 4,
        g_DATA_WIDTH => writeregs_C_fifo_wdata'length,
        g_RREG_N => 1--,
    )
    port map (
        i_wdata     => writeregs_C_fifo_wdata,
        i_we        => be3_prev or be4_prev,
        o_wfull     => open,
        i_wclk      => refclk,

        o_rdata     => writeregs_C_fifo_rdata,
        i_rack      => not writeregs_C_fifo_rempty,
        o_rempty    => writeregs_C_fifo_rempty,
        i_rclk      => i_clk_C,

        i_reset_n   => local_rstn--,
    );

    -- writeregs_C_reset_n is several clock cycles longer than local_rstn,
    e_writeregs_C_reset_n : entity work.reset_sync
    port map ( o_reset_n => writeregs_C_reset_n, i_reset_n => local_rstn, i_clk => i_clk_C );

    process(i_clk_C, writeregs_C_reset_n)
    begin
    if ( writeregs_C_reset_n = '0' ) then
        -- note that e_writeregs_C_fifo is driven by local_rstn,
        -- so during writeregs_C_reset_n the write requests are buffered
        -- and the writes are delayed (but not lost)
        writeregs_C <= (others => (others => '0'));
        --
    elsif rising_edge(i_clk_C) then
        if ( writeregs_C_fifo_rempty = '0' ) then
            writeregs_C(to_integer(unsigned(writeregs_C_fifo_rdata(37 downto 32)))) <= writeregs_C_fifo_rdata(31 downto 0);
            regwritten_C(to_integer(unsigned(writeregs_C_fifo_rdata(37 downto 32)))) <= '1';
        end if;
        --
    end if;
    end process;

end architecture;
