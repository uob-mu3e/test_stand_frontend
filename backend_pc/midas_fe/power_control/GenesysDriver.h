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
		//bool ReadVoltage(int)
		bool SelectChannel(int);
		void SetStateChanged();

	
	private:
	
		bool OPC();
		bool SetActiveChannel(int);
		void SetState(int,bool,INT&);
		bool AskPermissionToTurnOn(int);
	
		EQUIPMENT_INFO* info;
		std::string name;				
		midas::odb settings;
		midas::odb variables;
		TCPClient* client;
		
		//channel settings
		std::vector<int> supplyID;
		std::vector<bool> state;
		std::vector<bool> demandstate;

};


#endif
