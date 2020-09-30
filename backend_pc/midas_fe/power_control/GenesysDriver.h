//****************************************************************************************
//
//	Driver for the Genesys LV power supplies. Not compatible with EQ_SLOW, use EQ_PERIODIC
//
//	F.Wauters - Sep. 2020
//	

#ifndef GENESYSDRIVER_H
#define GENESYSDRIVER_H

#include <iostream>
#include "midas.h"
#include "odbxx.h"
#include "TCPClient.h"

/*SCPI Protocol
Recommended time delay between commands: 5mSec minimum. Some commands might require
longer time. In such cases, refer to NOTE following command description.
*/

//void my_settings_changed(midas::odb);


class GenesysDriver
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


	
	private:
	
		//utility/communications functions
		bool OPC();
		bool SetActiveChannel(int);
		void SetState(int,bool,INT&);
		void SetVoltage(int,float,INT&);
		void SetCurrentLimit(int,float,INT&);
		bool AskPermissionToTurnOn(int);
		bool SelectChannel(int);
		void SetStateChanged();
		void DemandVoltageChanged();
		void CurrentLimitChanged();
		void InitODBArray();
		float Read(std::string,INT&);
		bool Set(std::string,INT&);

	
		EQUIPMENT_INFO* info;
		std::string name;				
		midas::odb settings;
		midas::odb variables;
		TCPClient* client;
		
		//local copy of hardware settings
		std::vector<int> supplyID;
		std::vector<bool> state;
		std::vector<float> voltage;
		std::vector<float> demandvoltage;
		std::vector<float> current;
		std::vector<float> currentlimit;
		
		float relevantchange;
		
		

};


#endif
