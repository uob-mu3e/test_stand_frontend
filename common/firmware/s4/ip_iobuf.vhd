-- megafunction wizard: %ALTIOBUF%
-- GENERATION: STANDARD
-- VERSION: WM1.0
-- MODULE: altiobuf_bidir 

-- ============================================================
-- File Name: ip_iobuf.vhd
-- Megafunction Name(s):
-- 			altiobuf_bidir
--
-- Simulation Library Files(s):
-- 			arriav
-- ============================================================
-- ************************************************************
-- THIS IS A WIZARD-GENERATED FILE. DO NOT EDIT THIS FILE!
--
-- 18.0.0 Build 614 04/24/2018 SJ Standard Edition
-- ************************************************************


--Copyright (C) 2018  Intel Corporation. All rights reserved.
--Your use of Intel Corporation's design tools, logic functions 
--and other software and tools, and its AMPP partner logic 
--functions, and any output files from any of the foregoing 
--(including device programming or simulation files), and any 
--associated documentation or information are expressly subject 
--to the terms and conditions of the Intel Program License 
--Subscription Agreement, the Intel Quartus Prime License Agreement,
--the Intel FPGA IP License Agreement, or other applicable license
--agreement, including, without limitation, that your use is for
--the sole purpose of programming logic devices manufactured by
--Intel and sold by Intel or its authorized distributors.  Please
--refer to the applicable agreement for further details.


--altiobuf_bidir CBX_AUTO_BLACKBOX="ALL" DEVICE_FAMILY="Arria V" ENABLE_BUS_HOLD="FALSE" NUMBER_OF_CHANNELS=1 OPEN_DRAIN_OUTPUT="FALSE" USE_DIFFERENTIAL_MODE="FALSE" USE_DYNAMIC_TERMINATION_CONTROL="FALSE" USE_TERMINATION_CONTROL="FALSE" datain dataio dataout oe
--VERSION_BEGIN 18.0 cbx_altiobuf_bidir 2018:04:18:06:50:44:SJ cbx_mgl 2018:04:18:07:37:08:SJ cbx_stratixiii 2018:04:18:06:50:44:SJ cbx_stratixv 2018:04:18:06:50:44:SJ  VERSION_END

 LIBRARY arriav;
 USE arriav.all;

--synthesis_resources = arriav_io_ibuf 1 arriav_io_obuf 1 
 LIBRARY ieee;
 USE ieee.std_logic_1164.all;

 ENTITY  ip_iobuf_iobuf_bidir_nho IS 
	 PORT 
	 ( 
		 datain	:	IN  STD_LOGIC_VECTOR (0 DOWNTO 0);
		 dataio	:	INOUT  STD_LOGIC_VECTOR (0 DOWNTO 0);
		 dataout	:	OUT  STD_LOGIC_VECTOR (0 DOWNTO 0);
		 oe	:	IN  STD_LOGIC_VECTOR (0 DOWNTO 0)
	 ); 
 END ip_iobuf_iobuf_bidir_nho;

 ARCHITECTURE RTL OF ip_iobuf_iobuf_bidir_nho IS

	 SIGNAL  wire_ibufa_o	:	STD_LOGIC;
	 SIGNAL  wire_obufa_o	:	STD_LOGIC;
	 COMPONENT  arriav_io_ibuf
	 GENERIC 
	 (
		bus_hold	:	STRING := "false";
		differential_mode	:	STRING := "false";
		simulate_z_as	:	STRING := "z";
		lpm_type	:	STRING := "arriav_io_ibuf"
	 );
	 PORT
	 ( 
		dynamicterminationcontrol	:	IN STD_LOGIC := '0';
		i	:	IN STD_LOGIC := '0';
		ibar	:	IN STD_LOGIC := '0';
		o	:	OUT STD_LOGIC
	 ); 
	 END COMPONENT;
	 COMPONENT  arriav_io_obuf
	 GENERIC 
	 (
		bus_hold	:	STRING := "false";
		open_drain_output	:	STRING := "false";
		shift_series_termination_control	:	STRING := "false";
		lpm_type	:	STRING := "arriav_io_obuf"
	 );
	 PORT
	 ( 
		dynamicterminationcontrol	:	IN STD_LOGIC := '0';
		i	:	IN STD_LOGIC := '0';
		o	:	OUT STD_LOGIC;
		obar	:	OUT STD_LOGIC;
		oe	:	IN STD_LOGIC := '1';
		parallelterminationcontrol	:	IN STD_LOGIC_VECTOR(15 DOWNTO 0) := (OTHERS => '0');
		seriesterminationcontrol	:	IN STD_LOGIC_VECTOR(15 DOWNTO 0) := (OTHERS => '0')
	 ); 
	 END COMPONENT;
 BEGIN

	dataio(0) <= wire_obufa_o;
	dataout(0) <= wire_ibufa_o;
	ibufa :  arriav_io_ibuf
	  GENERIC MAP (
		bus_hold => "false",
		differential_mode => "false"
	  )
	  PORT MAP ( 
		i => dataio(0),
		o => wire_ibufa_o
	  );
	obufa :  arriav_io_obuf
	  GENERIC MAP (
		bus_hold => "false",
		open_drain_output => "false"
	  )
	  PORT MAP ( 
		i => datain(0),
		o => wire_obufa_o,
		oe => oe(0)
	  );

 END RTL; --ip_iobuf_iobuf_bidir_nho
--VALID FILE


LIBRARY ieee;
USE ieee.std_logic_1164.all;

ENTITY ip_iobuf IS
	PORT
	(
		datain		: IN STD_LOGIC_VECTOR (0 DOWNTO 0);
		oe		: IN STD_LOGIC_VECTOR (0 DOWNTO 0);
		dataio		: INOUT STD_LOGIC_VECTOR (0 DOWNTO 0);
		dataout		: OUT STD_LOGIC_VECTOR (0 DOWNTO 0)
	);
END ip_iobuf;


ARCHITECTURE RTL OF ip_iobuf IS

	SIGNAL sub_wire0	: STD_LOGIC_VECTOR (0 DOWNTO 0);



	COMPONENT ip_iobuf_iobuf_bidir_nho
	PORT (
			datain	: IN STD_LOGIC_VECTOR (0 DOWNTO 0);
			oe	: IN STD_LOGIC_VECTOR (0 DOWNTO 0);
			dataout	: OUT STD_LOGIC_VECTOR (0 DOWNTO 0);
			dataio	: INOUT STD_LOGIC_VECTOR (0 DOWNTO 0)
	);
	END COMPONENT;

BEGIN
	dataout    <= sub_wire0(0 DOWNTO 0);

	ip_iobuf_iobuf_bidir_nho_component : ip_iobuf_iobuf_bidir_nho
	PORT MAP (
		datain => datain,
		oe => oe,
		dataout => sub_wire0,
		dataio => dataio
	);



END RTL;

-- ============================================================
-- CNX file retrieval info
-- ============================================================
-- Retrieval info: PRIVATE: INTENDED_DEVICE_FAMILY STRING "Arria V"
-- Retrieval info: PRIVATE: SYNTH_WRAPPER_GEN_POSTFIX STRING "0"
-- Retrieval info: LIBRARY: altera_mf altera_mf.altera_mf_components.all
-- Retrieval info: CONSTANT: INTENDED_DEVICE_FAMILY STRING "Arria V"
-- Retrieval info: CONSTANT: enable_bus_hold STRING "FALSE"
-- Retrieval info: CONSTANT: left_shift_series_termination_control STRING "FALSE"
-- Retrieval info: CONSTANT: number_of_channels NUMERIC "1"
-- Retrieval info: CONSTANT: open_drain_output STRING "FALSE"
-- Retrieval info: CONSTANT: use_differential_mode STRING "FALSE"
-- Retrieval info: CONSTANT: use_dynamic_termination_control STRING "FALSE"
-- Retrieval info: CONSTANT: use_termination_control STRING "FALSE"
-- Retrieval info: USED_PORT: datain 0 0 1 0 INPUT NODEFVAL "datain[0..0]"
-- Retrieval info: USED_PORT: dataio 0 0 1 0 BIDIR NODEFVAL "dataio[0..0]"
-- Retrieval info: USED_PORT: dataout 0 0 1 0 OUTPUT NODEFVAL "dataout[0..0]"
-- Retrieval info: USED_PORT: oe 0 0 1 0 INPUT NODEFVAL "oe[0..0]"
-- Retrieval info: CONNECT: @datain 0 0 1 0 datain 0 0 1 0
-- Retrieval info: CONNECT: @oe 0 0 1 0 oe 0 0 1 0
-- Retrieval info: CONNECT: dataio 0 0 1 0 @dataio 0 0 1 0
-- Retrieval info: CONNECT: dataout 0 0 1 0 @dataout 0 0 1 0
-- Retrieval info: GEN_FILE: TYPE_NORMAL ip_iobuf.vhd TRUE
-- Retrieval info: GEN_FILE: TYPE_NORMAL ip_iobuf.inc FALSE
-- Retrieval info: GEN_FILE: TYPE_NORMAL ip_iobuf.cmp FALSE
-- Retrieval info: GEN_FILE: TYPE_NORMAL ip_iobuf.bsf FALSE
-- Retrieval info: GEN_FILE: TYPE_NORMAL ip_iobuf_inst.vhd FALSE
-- Retrieval info: LIB_FILE: arriav