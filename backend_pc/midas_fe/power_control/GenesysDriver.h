//****************************************************************************************
//
//	Driver for the Genesys LV power supplies. Not compatible with EQ_SLOW, use EQ_PERIODIC
//
//	F.Wauters - Sep. 2020
//	

#ifndef GENESYSDRIVER_H
#define GENESYSDRIVER_H

#include "PowerDriver.h"

/*SCPI Protocol for TDK
Recommended time delay between commands: 5mSec minimum. Some commands might require
longer time. In such cases, refer to NOTE following command description.
*/

//void my_settings_changed(midas::odb);


class GenesysDriver : public PowerDriver
{
	public:
	
		GenesysDriver(std::string n, EQUIPMENT_INFO* inf);
		~GenesysDriver();
		
		INT ConnectODB();
		INT Connect();
		INT Init();
		void Print();
		bool ReadState(int,INT&);
		float ReadVoltage(int,INT&);
		float ReadSetVoltage(int,INT&);
		float ReadCurrent(int,INT&);
		float ReadCurrentLimit(int,INT&);
		std::string ReadIDCode(int,INT&);
		INT ReadAll();
		
		std::vector<bool> GetState() { return state; }
		std::vector<float> GetVoltage() { return voltage; }
		std::vector<float> GetCurrent() { return current; }


	
	private:
	
		//utility/communications functions
		bool OPC();
		bool SetActiveChannel(int);
		void SetState(int,bool,INT&);
		void SetVoltage(int,float,INT&);
		void SetCurrentLimit(int,float,INT&);
		void SetBlink(int,bool,INT&);
		bool AskPermissionToTurnOn(int);
		bool SelectChannel(int);
		void SetStateChanged();
		void DemandVoltageChanged();
		void CurrentLimitChanged();
		void BlinkChanged();
		void InitODBArray();
		float Read(std::string,INT&);
		bool Set(std::string,INT&);
	

		TCPClient* client;
		
		//local copy of hardware settings
		std::vector<int> supplyID;
		std::vector<bool> state;
		std::vector<float> voltage;
		std::vector<float> demandvoltage;
		std::vector<float> current;
		std::vector<float> currentlimit;
		std::vector<std::string> idCode;
		
		float relevantchange;
		
		

};


#endif
