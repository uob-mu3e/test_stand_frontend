-- M.Mueller, May 2021

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.scifi_registers.all;

entity scifi_reg_mapping is
generic (
    N_MODULES   : positive;
    N_ASICS     : positive;
    N_LINKS     : positive := 1--;
);
port (
    i_clk                       : in  std_logic;
    i_reset_n                   : in  std_logic;

    i_receivers_usrclk          : in  std_logic;

    i_reg_add                   : in  std_logic_vector(15 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    -- inputs
    i_cntreg_num                : in  std_logic_vector(31 downto 0); -- on receivers_usrclk domain
    i_cntreg_denom_b            : in  std_logic_vector(63 downto 0); -- on receivers_usrclk domain
    i_rx_pll_lock               : in  std_logic;
    i_frame_desync              : in  std_logic_vector(1 downto 0);
    i_rx_dpa_lock_reg           : in  std_logic_vector(N_MODULES*N_ASICS - 1 downto 0); -- on receivers_usrclk domain
    i_rx_ready                  : in  std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);

    -- outputs
    o_cntreg_ctrl               : out std_logic_vector(31 downto 0);
    o_dummyctrl_reg             : out std_logic_vector(31 downto 0);
    o_dpctrl_reg                : out std_logic_vector(31 downto 0);
    o_subdet_reset_reg          : out std_logic_vector(31 downto 0);
    o_subdet_resetdly_reg_written : out std_logic;
    o_subdet_resetdly_reg       : out std_logic_vector(31 downto 0)--;

);
end entity;

architecture rtl of scifi_reg_mapping is

    signal cntreg_ctrl          : std_logic_vector(31 downto 0);
    signal dummyctrl_reg        : std_logic_vector(31 downto 0);
    signal dpctrl_reg           : std_logic_vector(31 downto 0);
    signal subdet_reset_reg     : std_logic_vector(31 downto 0);
    signal subdet_resetdly_reg  : std_logic_vector(31 downto 0);

    signal sync_cntreg_num      : std_logic_vector(31 downto 0);
    signal sync_cntreg_denom_b  : std_logic_vector(63 downto 0);
    signal sync_rx_dpa_lock_reg : std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);

    signal q_sync, data_sync    : std_logic_vector(32 + 64 + N_MODULES*N_ASICS - 1 downto 0);
    signal empty                : std_logic;

    signal q_sync_out, data_sync_out : std_logic_vector(63 downto 0);
    signal empty_out            : std_logic;

begin

    o_dpctrl_reg            <= dpctrl_reg;
    o_subdet_reset_reg      <= subdet_reset_reg;
    o_subdet_resetdly_reg   <= subdet_resetdly_reg;


    --! input sync
    data_sync <= i_cntreg_num & i_cntreg_denom_b & i_rx_dpa_lock_reg;
    e_sync_in : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4, DATA_WIDTH  => 32 + 64 + N_MODULES*N_ASICS--,
    ) port map ( data => data_sync, wrreq => '1',
             rdreq => not empty, wrclk => i_receivers_usrclk, rdclk => i_clk,
             q => q_sync, rdempty => empty, aclr => '0'--,
    );
    sync_cntreg_num         <= q_sync(32 + 64 + N_MODULES*N_ASICS - 1 downto 64 + N_MODULES*N_ASICS);
    sync_cntreg_denom_b     <= q_sync(     64 + N_MODULES*N_ASICS - 1 downto      N_MODULES*N_ASICS);
    sync_rx_dpa_lock_reg    <= q_sync(          N_MODULES*N_ASICS - 1 downto                      0);


    --! output sync
    data_sync_out <= cntreg_ctrl & dummyctrl_reg;
    e_sync_out : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4, DATA_WIDTH  => 64--,
    ) port map ( data => data_sync_out, wrreq => '1',
             rdreq => not empty_out, wrclk => i_clk, rdclk => i_receivers_usrclk,
             q => q_sync_out, rdempty => empty_out, aclr => '0'--,
    );
    o_cntreg_ctrl   <= q_sync_out(63 downto 32);
    o_dummyctrl_reg <= q_sync_out(31 downto  0);


    process(i_clk, i_reset_n)
        variable regaddr : integer;
    begin

    if ( i_reset_n = '0' ) then
        dummyctrl_reg           <= (others=>'0');
        dpctrl_reg              <= (others=>'0');
        subdet_reset_reg        <= (others=>'0');
        subdet_resetdly_reg     <= (others=>'0');
    elsif ( rising_edge(i_clk) ) then
        o_reg_rdata         <= X"CCCCCCCC";
        regaddr             := to_integer(unsigned(i_reg_add));
        o_subdet_resetdly_reg_written <= '0';

        if ( i_reg_re = '1' and regaddr = SCIFI_CNT_CTRL_REGISTER_W ) then
            o_reg_rdata <= cntreg_ctrl;
        end if;
        if ( i_reg_we = '1' and regaddr = SCIFI_CNT_CTRL_REGISTER_W ) then
            cntreg_ctrl <= i_reg_wdata;
        end if;

        if ( i_reg_re = '1' and regaddr = SCIFI_CNT_NOM_REGISTER_REGISTER_R ) then
            o_reg_rdata <= sync_cntreg_num;
        end if;

        if ( i_reg_re = '1' and regaddr = SCIFI_CNT_DENOM_LOWER_REGISTER_R ) then
            o_reg_rdata <= sync_cntreg_denom_b(31 downto 0);
        end if;
        if ( i_reg_re = '1' and regaddr = SCIFI_CNT_DENOM_UPPER_REGISTER_R ) then
            o_reg_rdata <= sync_cntreg_denom_b(63 downto 32);
        end if;

        if ( i_reg_re = '1' and regaddr = SCIFI_MON_STATUS_REGISTER_R ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(0) <= i_rx_pll_lock;
            o_reg_rdata(5 downto 4) <= i_frame_desync;
            o_reg_rdata(9 downto 8) <= "00";
        end if;

        if ( i_reg_re = '1' and regaddr = SCIFI_MON_RX_DPA_LOCK_REGISTER_R ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(N_MODULES*N_ASICS - 1 downto 0) <= sync_rx_dpa_lock_reg;
        end if;

        if ( i_reg_re = '1' and regaddr = SCIFI_MON_RX_READY_REGISTER_R ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(N_MODULES*N_ASICS - 1 downto 0) <= i_rx_ready;
        end if;

        if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_DUMMY_REGISTER_W ) then
            dummyctrl_reg <= i_reg_wdata;
        end if;
        if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_DUMMY_REGISTER_W ) then
            o_reg_rdata <= dummyctrl_reg;
        end if;

        if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_DP_REGISTER_W ) then
            dpctrl_reg <= i_reg_wdata;
        end if;
        if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_DP_REGISTER_W ) then
            o_reg_rdata <= dpctrl_reg;
        end if;

        if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_RESET_REGISTER_W ) then
            subdet_reset_reg <= i_reg_wdata;
        end if;
        if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_RESET_REGISTER_W ) then
            o_reg_rdata <= subdet_reset_reg;
        end if;

        if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_RESETDELAY_REGISTER_W ) then
            subdet_resetdly_reg <= i_reg_wdata;
            o_subdet_resetdly_reg_written <= '1';
        end if;
        if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_RESETDELAY_REGISTER_W ) then
            o_reg_rdata <= subdet_resetdly_reg;
        end if;

    end if;
    end process;

end architecture;
