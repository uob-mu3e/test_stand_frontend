-----------------------------------------------------------------------------
-- Handling of the DDR3 (eventually DDR4) buffer for the farm pcs
--
-- Niklaus Berger, JGU Mainz
-- niberger@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.a10_pcie_registers.all;

entity ddr3_block is
port (
    -- Control and status registers
    ddr3control         : in    std_logic_vector(31 downto 0);
    ddr3status          : out   std_logic_vector(31 downto 0);

    -- A interface
    A_ddr3clk           : out   std_logic;
    A_ddr3calibrated    : out   std_logic;
    A_ddr3ready         : out   std_logic;
    A_ddr3addr          : in    std_logic_vector(25 downto 0);
    A_ddr3datain        : in    std_logic_vector(511 downto 0);
    A_ddr3dataout       : out   std_logic_vector(511 downto 0);
    A_ddr3_write        : in    std_logic;
    A_ddr3_read         : in    std_logic;
    A_ddr3_read_valid   : out   std_logic;

    -- B interface
    B_ddr3clk           : out   std_logic;
    B_ddr3calibrated    : out   std_logic;
    B_ddr3ready         : out   std_logic;
    B_ddr3addr          : in    std_logic_vector(25 downto 0);
    B_ddr3datain        : in    std_logic_vector(511 downto 0);
    B_ddr3dataout       : out   std_logic_vector(511 downto 0);
    B_ddr3_write        : in    std_logic;
    B_ddr3_read         : in    std_logic;
    B_ddr3_read_valid   : out   std_logic;

    -- Error counters
    errout              : out   std_logic_vector(31 downto 0);

    -- Interface to memory bank A
    A_mem_ck            : out   std_logic_vector(0 downto 0);
    A_mem_ck_n          : out   std_logic_vector(0 downto 0);
    A_mem_a             : out   std_logic_vector(15 downto 0);
    A_mem_ba            : out   std_logic_vector(2 downto 0);
    A_mem_cke           : out   std_logic_vector(0 downto 0);
    A_mem_cs_n          : out   std_logic_vector(0 downto 0);
    A_mem_odt           : out   std_logic_vector(0 downto 0);
    A_mem_reset_n       : out   std_logic_vector(0 downto 0);
    A_mem_we_n          : out   std_logic_vector(0 downto 0);
    A_mem_ras_n         : out   std_logic_vector(0 downto 0);
    A_mem_cas_n         : out   std_logic_vector(0 downto 0);
    A_mem_dqs           : inout std_logic_vector(7 downto 0)
    A_mem_dqs_n         : inout std_logic_vector(7 downto 0)
    A_mem_dq            : inout std_logic_vector(63 downto 0)
    A_mem_dm            : out   std_logic_vector(7 downto 0);
    A_oct_rzqin         : in    std_logic;
    A_pll_ref_clk       : in    std_logic;

    -- Interface to memory bank B
    B_mem_ck            : out   std_logic_vector(0 downto 0);
    B_mem_ck_n          : out   std_logic_vector(0 downto 0);
    B_mem_a             : out   std_logic_vector(15 downto 0);
    B_mem_ba            : out   std_logic_vector(2 downto 0);
    B_mem_cke           : out   std_logic_vector(0 downto 0);
    B_mem_cs_n          : out   std_logic_vector(0 downto 0);
    B_mem_odt           : out   std_logic_vector(0 downto 0);
    B_mem_reset_n       : out   std_logic_vector(0 downto 0);
    B_mem_we_n          : out   std_logic_vector(0 downto 0);
    B_mem_ras_n         : out   std_logic_vector(0 downto 0);
    B_mem_cas_n         : out   std_logic_vector(0 downto 0);
    B_mem_dqs           : inout std_logic_vector(7 downto 0)
    B_mem_dqs_n         : inout std_logic_vector(7 downto 0)
    B_mem_dq            : inout std_logic_vector(63 downto 0)
    B_mem_dm            : out   std_logic_vector(7 downto 0);
    B_oct_rzqin         : in    std_logic;
    B_pll_ref_clk       : in    std_logic;

    reset_n             : in    std_logic--;
);
end entity;

architecture RTL of ddr3_block is

    signal A_cal_success : std_logic;
    signal A_cal_fail : std_logic;

    signal A_clk : std_logic;
    signal A_reset_n : std_logic;

    signal A_ready : std_logic;
    signal A_read : std_logic;
    signal A_write : std_logic;
    signal A_address : std_logic_vector(25 downto 0);
    signal A_readdata : std_logic_vector(511 downto 0);
    signal A_writedata : std_logic_vector(511 downto 0);
    signal A_burstcount : std_logic_vector(6 downto 0);
    signal A_readdatavalid : std_logic;


    signal B_cal_success : std_logic;
    signal B_cal_fail : std_logic;

    signal B_clk : std_logic;
    signal B_reset_n : std_logic;

    signal B_ready : std_logic;
    signal B_read : std_logic;
    signal B_write : std_logic;
    signal B_address : std_logic_vector(25 downto 0);
    signal B_readdata : std_logic_vector(511 downto 0);
    signal B_writedata : std_logic_vector(511 downto 0);
    signal B_burstcount : std_logic_vector(6 downto 0);
    signal B_readdatavalid : std_logic;

    signal A_errout : std_logic_vector(31 downto 0);
    signal B_errout : std_logic_vector(31 downto 0);

begin

    ddr3_A : component work.cmp.ddr3_if
    port map (
        amm_ready_0         => A_ready,
        amm_read_0          => A_read,
        amm_write_0         => A_write,
        amm_address_0       => A_address,
        amm_readdata_0      => A_readdata,
        amm_writedata_0     => A_writedata,
        amm_burstcount_0    => A_burstcount,
        amm_byteenable_0    =>  (others => '1'),
        amm_readdatavalid_0 => A_readdatavalid,
        emif_usr_clk        => A_clk,
        emif_usr_reset_n    => A_reset_n,
        global_reset_n      => reset_n,
        mem_ck              => A_mem_ck,
        mem_ck_n            => A_mem_ck_n,
        mem_a               => A_mem_a,
        mem_ba              => A_mem_ba,
        mem_cke             => A_mem_cke,
        mem_cs_n            => A_mem_cs_n,
        mem_odt             => A_mem_odt,
        mem_reset_n         => A_mem_reset_n,
        mem_we_n            => A_mem_we_n,
        mem_ras_n           => A_mem_ras_n,
        mem_cas_n           => A_mem_cas_n,
        mem_dqs             => A_mem_dqs,
        mem_dqs_n           => A_mem_dqs_n,
        mem_dq              => A_mem_dq,
        mem_dm              => A_mem_dm,
        oct_rzqin           => A_oct_rzqin,
        pll_ref_clk         => A_pll_ref_clk,
        local_cal_success   => A_cal_success,
        local_cal_fail      => A_cal_fail
    );

    ddr3_B : component work.cmp.ddr3_if
    port map (
        amm_ready_0         => B_ready,
        amm_read_0          => B_read,
        amm_write_0         => B_write,
        amm_address_0       => B_address,
        amm_readdata_0      => B_readdata,
        amm_writedata_0     => B_writedata,
        amm_burstcount_0    => B_burstcount,
        amm_byteenable_0    =>  (others => '1'),
        amm_readdatavalid_0 => B_readdatavalid,
        emif_usr_clk        => B_clk,
        emif_usr_reset_n    => B_reset_n,
        global_reset_n      => reset_n,
        mem_ck              => B_mem_ck,
        mem_ck_n            => B_mem_ck_n,
        mem_a               => B_mem_a,
        mem_ba              => B_mem_ba,
        mem_cke             => B_mem_cke,
        mem_cs_n            => B_mem_cs_n,
        mem_odt             => B_mem_odt,
        mem_reset_n         => B_mem_reset_n,
        mem_we_n            => B_mem_we_n,
        mem_ras_n           => B_mem_ras_n,
        mem_cas_n           => B_mem_cas_n,
        mem_dqs             => B_mem_dqs,
        mem_dqs_n           => B_mem_dqs_n,
        mem_dq              => B_mem_dq,
        mem_dm              => B_mem_dm,
        oct_rzqin           => B_oct_rzqin,
        pll_ref_clk         => B_pll_ref_clk,
        local_cal_success   => B_cal_success,
        local_cal_fail      => B_cal_fail
    );


    errout <= B_errout when ddr3control(1+16) = '1' else A_errout;

    memctlA : entity work.ddr3_memory_controller
    port map (
        reset_n         => reset_n,

        -- Control and status registers
        ddr3control     => ddr3control(15 downto 0),
        ddr3status      => ddr3status(15 downto 0),

        ddr3clk         => A_ddr3clk,
        ddr3_calibrated => A_ddr3calibrated,
        ddr3_ready      => A_ddr3ready,

        ddr3addr        => A_ddr3addr,
        ddr3datain      => A_ddr3datain,
        ddr3dataout     => A_ddr3dataout,
        ddr3_write      => A_ddr3_write,
        ddr3_read       => A_ddr3_read,
        ddr3_read_valid => A_ddr3_read_valid,

        -- Error counters
        errout          => A_errout,

        -- IF to DDR3
        M_cal_success   => A_cal_success,
        M_cal_fail      => A_cal_fail,
        M_clk           => A_clk,
        M_reset_n       => A_reset_n,
        M_ready         => A_ready,
        M_read          => A_read,
        M_write         => A_write,
        M_address       => A_address,
        M_readdata      => A_readdata,
        M_writedata     => A_writedata,
        M_burstcount    => A_burstcount,
        M_readdatavalid => A_readdatavalid
    );

    memctlB : entity work.ddr3_memory_controller
    port map (
        reset_n         => reset_n,

        -- Control and status registers
        ddr3control     => ddr3control(31 downto 16),
        ddr3status      => ddr3status(31 downto 16),

        ddr3clk         => B_ddr3clk,
        ddr3_calibrated => B_ddr3calibrated,
        ddr3_ready      => B_ddr3ready,

        ddr3addr        => B_ddr3addr,
        ddr3datain      => B_ddr3datain,
        ddr3dataout     => B_ddr3dataout,
        ddr3_write      => B_ddr3_write,
        ddr3_read       => B_ddr3_read,
        ddr3_read_valid => B_ddr3_read_valid,

        -- Error counters
        errout          => B_errout,

        -- IF to DDR3
        M_cal_success   => B_cal_success,
        M_cal_fail      => B_cal_fail,
        M_clk           => B_clk,
        M_reset_n       => B_reset_n,
        M_ready         => B_ready,
        M_read          => B_read,
        M_write         => B_write,
        M_address       => B_address,
        M_readdata      => B_readdata,
        M_writedata     => B_writedata,
        M_burstcount    => B_burstcount,
        M_readdatavalid => B_readdatavalid
    );

end architecture;
