library IEEE;
use IEEE.STD_LOGIC_1164.all;
use ieee.numeric_std.all;

package ipbus_decode_mu3e_address is

  constant IPBUS_SEL_WIDTH: positive := 5; -- Should be enough for now?
  subtype ipbus_sel_t is std_logic_vector(IPBUS_SEL_WIDTH - 1 downto 0);
  function ipbus_sel_mu3e_address(addr : in std_logic_vector(31 downto 0)) return ipbus_sel_t;

  constant N_SLV_FIFO_REG_OUT: integer := 0;
  constant N_SLV_FIFO_REG_OUT_CHARISK: integer := 1;
  constant N_SLV_FIFO_REG_IN: integer := 2;
  constant N_SLV_CTRL_REG: integer := 3;
  constant N_SLV_I2C: integer := 4;
  constant N_SLV_I2C_FAST: integer := 5;
  constant N_SLV_I2C_MEM: integer := 6;
  constant N_SLAVES: integer := 7;


end ipbus_decode_mu3e_address;

package body ipbus_decode_mu3e_address is

  function ipbus_sel_mu3e_address(addr : in std_logic_vector(31 downto 0)) return ipbus_sel_t is
    variable sel: ipbus_sel_t;
  begin

    if    std_match(addr, "---------------------------0000-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_FIFO_REG_OUT, IPBUS_SEL_WIDTH)); -- fifo_reg_out / base 0x00000000 / mask 0x0000000e
    elsif std_match(addr, "---------------------------0001-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_FIFO_REG_OUT_CHARISK, IPBUS_SEL_WIDTH)); -- fifo_reg_out_charisk / base 0x00000002 / mask 0x0000000e
    elsif std_match(addr, "---------------------------0010-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_FIFO_REG_IN, IPBUS_SEL_WIDTH)); -- fifo_reg_in / base 0x00000004 / mask 0x0000000e
    elsif std_match(addr, "---------------------------0011-") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_CTRL_REG, IPBUS_SEL_WIDTH)); -- ctrl_reg / base 0x00000006 / mask 0x0000000e
    elsif std_match(addr, "---------------------------01---") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_I2C, IPBUS_SEL_WIDTH)); -- i2c / base 0x00000008 / mask 0x00000008
    elsif std_match(addr, "---------------------------10---") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_I2C_FAST, IPBUS_SEL_WIDTH)); -- i2c / base 0x00000010 / mask 0x00000008 
    elsif std_match(addr, "---------------------------11---") then
      sel := ipbus_sel_t(to_unsigned(N_SLV_I2C_MEM, IPBUS_SEL_WIDTH)); -- i2c / base 0x00000018 / mask 0x00000008      

    else
        sel := ipbus_sel_t(to_unsigned(N_SLAVES, IPBUS_SEL_WIDTH));
    end if;

    return sel;

  end function ipbus_sel_mu3e_address;

end ipbus_decode_mu3e_address;
