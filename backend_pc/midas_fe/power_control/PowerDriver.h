//****************************************************************************************
//
//	Base Driver for the LV power supplies. Use derived class fro TDK or HAMEG or .. supply
//
//	F.Wauters - Nov. 2020
//	

#ifndef POWERDRIVER_H
#define POWERDRIVER_H

#include <iostream>
#include "midas.h"
#include "odbxx.h"
#include "TCPClient.h"



class PowerDriver{	

	public:
	
		PowerDriver();
		PowerDriver(std::string, EQUIPMENT_INFO*);
		
		virtual INT ConnectODB(){return FE_ERR_DRIVER;};
		virtual INT Connect(){return FE_ERR_DRIVER;};
		virtual INT Init(){return FE_ERR_DRIVER;};
		virtual INT ReadAll(){return FE_ERR_DRIVER;}
		virtual std::vector<bool> GetState() { return {}; }
		virtual std::vector<float> GetVoltage() { return {}; }
		virtual std::vector<float> GetCurrent() { return {}; }
		virtual	void Print(){};
		
	protected:
		EQUIPMENT_INFO* info;
		std::string name;
		midas::odb settings;
		midas::odb variables;
};

#endif
