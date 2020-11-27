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
		
		virtual INT ConnectODB();
		INT Connect();
		virtual INT Init(){return FE_ERR_DRIVER;};
		virtual INT ReadAll(){return FE_ERR_DRIVER;}
		virtual std::vector<bool> GetState() { return {}; }
		virtual std::vector<float> GetVoltage() { return {}; }
		virtual std::vector<float> GetCurrent() { return {}; }
		void Print();
		
		bool Initialized() { return initialized; }
		bool Enabled();
		void SetInitialized() { initialized = true; }
		std::string ReadIDCode(int,INT&);
		
		virtual bool AskPermissionToTurnOn(int) { std::cout << "Ask permissions in derived class!" << std::endl; return false;};
		
		bool ReadState(int,INT&);
		float ReadVoltage(int,INT&);
		float ReadCurrent(int,INT&);
		int ReadESR(int,INT&);
		

		

	protected:
	
		//read
		float Read(std::string,INT&);
		float ReadSetVoltage(int,INT&);
		float ReadCurrentLimit(int,INT&);
		
		//set
		bool SelectChannel(int);
		bool OPC();
		bool Set(std::string,INT&);
		void SetCurrentLimit(int,float,INT&);
		
		//watch
		void CurrentLimitChanged();
		void SetStateChanged();
		void SetState(int,bool,INT&);
		void SetVoltage(int,float,INT&);
		void DemandVoltageChanged();
		
		EQUIPMENT_INFO* info;
		std::string name;
		midas::odb settings;
		midas::odb variables;
		
		TCPClient* client;
		
		float relevantchange;

		
		//local copies of hardware state
		std::vector<bool> state;
		std::vector<float> voltage;
		std::vector<float> demandvoltage;
		std::vector<float> current;
		std::vector<float> currentlimit;
		
		std::vector<int> instrumentID;
		
		
		
		
	private:
		bool initialized;
		int min_reply_length;
		

};

#endif
