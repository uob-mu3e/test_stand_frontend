-- Copyright (C) 2018  Intel Corporation. All rights reserved.
-- Your use of Intel Corporation's design tools, logic functions 
-- and other software and tools, and its AMPP partner logic 
-- functions, and any output files from any of the foregoing 
-- (including device programming or simulation files), and any 
-- associated documentation or information are expressly subject 
-- to the terms and conditions of the Intel Program License 
-- Subscription Agreement, the Intel Quartus Prime License Agreement,
-- the Intel FPGA IP License Agreement, or other applicable license
-- agreement, including, without limitation, that your use is for
-- the sole purpose of programming logic devices manufactured by
-- Intel and sold by Intel or its authorized distributors.  Please
-- refer to the applicable agreement for further details.

-- ***************************************************************************
-- This file contains a Vhdl test bench template that is freely editable to   
-- suit user's needs .Comments are provided in each section to help the user  
-- fill out necessary details.                                                
-- ***************************************************************************
-- Generated on "03/09/2021 11:19:57"
                                                            
-- Vhdl Test Bench template for design  :  histogram_generic
-- 
-- Simulation tool : ModelSim-Altera (VHDL)
-- 

LIBRARY ieee;                                               
USE ieee.std_logic_1164.all;                                
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

ENTITY histogram_generic_vhd_tst IS
END histogram_generic_vhd_tst;
ARCHITECTURE histogram_generic_arch OF histogram_generic_vhd_tst IS
-- constants                                                 
-- signals                                                   
SIGNAL busy_n : STD_LOGIC;
SIGNAL can_overflow : STD_LOGIC;
SIGNAL data_in : STD_LOGIC_VECTOR(5 DOWNTO 0);
SIGNAL ena : STD_LOGIC;
SIGNAL q_out : STD_LOGIC_VECTOR(7 DOWNTO 0);
SIGNAL raddr_in : STD_LOGIC_VECTOR(5 DOWNTO 0);
SIGNAL rclk : STD_LOGIC;
SIGNAL rst_n : STD_LOGIC;
SIGNAL valid_in : STD_LOGIC;
SIGNAL wclk : STD_LOGIC;
SIGNAL zeromem : STD_LOGIC;
signal injections : std_logic_vector(31 downto 0);
signal counter    : std_logic_vector(31 downto 0);
COMPONENT histogram_generic
    PORT (
    busy_n : OUT STD_LOGIC;
    can_overflow : IN STD_LOGIC;
    data_in : IN STD_LOGIC_VECTOR(5 DOWNTO 0);
    ena : IN STD_LOGIC;
    q_out : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    raddr_in : IN STD_LOGIC_VECTOR(5 DOWNTO 0);
    rclk : IN STD_LOGIC;
    rst_n : IN STD_LOGIC;
    valid_in : IN STD_LOGIC;
    wclk : IN STD_LOGIC;
    zeromem : IN STD_LOGIC
    );
END COMPONENT;
BEGIN
    i1 : histogram_generic
    PORT MAP (
-- list connections between master ports and signals
    busy_n => busy_n,
    can_overflow => can_overflow,
    data_in => data_in,
    ena => ena,
    q_out => q_out,
    raddr_in => raddr_in,
    rclk => rclk,
    rst_n => rst_n,
    valid_in => valid_in,
    wclk => wclk,
    zeromem => zeromem
    );
init : PROCESS                                               
-- variable declarations                                     
BEGIN                                                        
        -- code that executes only once                      
    rst_n           <= '0';
    ena             <= '0';
    can_overflow    <= '0';
    zeromem         <= '0';
    raddr_in        <= (others => '0');
    injections      <= x"0000FFFF";
    wait for 20 ns;
    rst_n             <= '1';
    wait for 20 ns;
 --   zeromem         <= '1';
    wait for 20 ns;
    ena             <= '1';
    wait for 10 us;
    ena             <= '0';
    wait for 100 ns;
    raddr_in        <= raddr_in+1;
    wait for 10 ns;
    raddr_in        <= raddr_in+1;
    wait for 10 ns;
    raddr_in        <= raddr_in+1;
    wait for 10 ns;
    raddr_in        <= raddr_in+1;
    wait for 10 ns;
    zeromem         <= '1';
    
WAIT;                                                       
END PROCESS init;                                           

wclk_prc : Process
begin
    wclk <= '0';
    wait for 2 ns;
    wclk <= '1';
    wait for 2 ns;
end process wclk_prc;

rclk_prc : Process
begin
    rclk <= '0';
    wait for 2.5 ns;
    rclk <= '1';
    wait for 2.5 ns;
end process rclk_prc;

always : PROCESS(wclk)                                              
BEGIN                                                         
    if(rising_edge(wclk))then
        if(rst_n = '0')then
            valid_in <= '0';
            data_in <= (others => '0');
            counter <= (others => '0');
        else
            if(ena = '1' and counter < injections)then
                valid_in <= '1';
                data_in <= data_in + 31;
                counter <= counter + 1;
                if(to_integer(unsigned(counter)) mod 29 = 0)then
                    valid_in <= '0';
                end if;
            else
                valid_in <= '0';
            end if;
        end if;
    end if;
END PROCESS always;
END histogram_generic_arch;
