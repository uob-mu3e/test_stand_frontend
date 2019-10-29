/********************************************************************\

Name:			MALIBU.h
Created by:		Tiancheng Zhong 
Date:			2019.10.14
Contents:       Definition of fumctions in class MALIBU
to provide an abstraction layer to the (slow control) functions on the FE-FPGA

\********************************************************************/

#ifndef MALIBU_H
#define MALIBU_H

#include "midas.h"
#include "mudaq_device_scifi.h"

namespace mudaq { 
	class MALIBU {
		private:
			mudaq::MudaqDevice& m_mu;
			static MALIBU* m_instance; //signleton instance pointer
			MALIBU(const MALIBU&)=delete;
			MALIBU(mudaq::MudaqDevice& mu):m_mu(mu){};

			// bits for MALIBU   all of them are chip related
			const alt_u8 A_bit =  0x01;	//dis-/enable 1.8A
			const alt_u8 D_bit =  0x02;	//dis-/enable 1.8D
			const alt_u8 CS_bit = 0x04;	//dis-/enable SC for chip
		public:
			static const uint8_t FPGA_broadcast_ID;

			static MALIBU* Create(mudaq::MudaqDevice& mu){printf("MALIBU::Create()");if(!m_instance) m_instance=new MALIBU(mu); return m_instance;};
			static MALIBU* Instance(){return m_instance;};

			//Basic functions{{{
			int		WriteTo_MALIBU(i2c_reg_t* regs, int n);//n:number of the partern; TODO:need the information from MIDAS
			bool	CheckWrite(i2c_reg_t* regs,int n);
			alt_u8	ReadFrom_MALIBU(i2c_reg_t regs);//n:number of the partern; TODO:need the information from MIDAS
			alt_u8	SetPattern(alt_u8 bit_data, alt_u8 bit_mask,bool enable); // prepare the bit parttern to enable or disable the specific bit
			//chip related basic functions
			int		Power18A(int asic_ID,bool enable);
			int		Power18D(int asic_ID,bool enable);
			int		EnableSPI(int asic_ID,bool enable);// enable ASIC configuration or not
			int		PowerAUX(bool enable, int i);//i: (0:AUX0, 1:AUX1 )
			//clk related basic functions
			int		sel_sysclk(int opt);	// opt:(0:CK_SI0, 1: CK_FPGA0)			//choose sysclk line
			int		sel_pllclk(int opt);	//opt:(0: MCRF connector, 1: CK_SI1)	//choose clk line for the pll reference clk
			int		sel_pllTest(int opt);	//opt (0: on Board, 1: MCRF connecters) //choose input signal for pll test
			int		Enable_pllTest(bool enable=false);	//enable pll test or not
			//}}}

			
			//higher level functions{{{
			int		configure_chip(int asic_ID, alt_u8* bitpattern);
			int		ASIC_Power(int asic_ID, bool enable, const alt_u8* chip_config_pattern); //Enable 1.8D and 1.8A and configure the chip to ALLOFF
			int		MALIBU_PowerUp();
			int		MALIBU_PowerDown();
			int		SetPLLtest(bool enable,bool useCoaxConnectors=false);
			
			//}}}
			struct	i2c_reg_t {                                                                                                                              
				alt_u8 slave;                                                                                                                               
				alt_u8 addr;                                                                                                                                
				alt_u8 data;                                                                                                                                
			}; 

			i2c_reg_t malibu_init_regs[18] = {
				{0x38,0x01,0x0C^0x20},//select all the clks
				{0x38,0x03,0x00},
				{0x38,0x01,0x0D^0x20},//enable AUX1
				{0xff,0x00,0x00},
				{0x38,0x01,0x0F^0x20},//enable AUX2
				{0xff,0x00,0x00},
				{0x39,0x01,0x3C},//keep all the 1.8 A D off
				{0x39,0x03,0x00},
				{0x3a,0x01,0x3C},
				{0x3a,0x03,0x00},
				{0x3b,0x01,0x3C},
				{0x3b,0x03,0x00},
				{0x3c,0x01,0x3C},
				{0x3c,0x03,0x00},
				{0x3d,0x01,0x3C},
				{0x3d,0x03,0x00},
				{0x3f,0x01,0x3C},
				{0x3f,0x03,0x00}
			};


			i2c_reg_t malibu_powerdown_regs[17] = {
				{0x3f,0x01,0x3C},    //Powerdown 1.8 A D for every single ASIC
				{0xff,0x00,0x00},    
				{0x3e,0x01,0x3C},    
				{0xff,0x00,0x00},    
				{0x3d,0x01,0x3C},    
				{0xff,0x00,0x00},    
				{0x3c,0x01,0x3C},    
				{0xff,0x00,0x00},    
				{0x3b,0x01,0x3C},    
				{0xff,0x00,0x00},    
				{0x3a,0x01,0x3C},    
				{0xff,0x00,0x00},    
				{0x39,0x01,0x3C},    
				{0xff,0x00,0x00},    
				{0x38,0x01,0x0D},    //Power down both AUX (3.3V)
				{0xff,0x00,0x00},    
				{0x38,0x01,0x0C}     
			}; 


	};//class MALIBU 
}//namespace mudaq 

#endif // MALIBU_H
