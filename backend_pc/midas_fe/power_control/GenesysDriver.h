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
		INT Init();
		INT ReadAll();
		
		std::vector<bool> GetState() { return state; }
		std::vector<float> GetVoltage() { return voltage; }
		std::vector<float> GetCurrent() { return current; }


	
	private:
	
		//utility/communications functions
		bool SetActiveChannel(int);
		void SetState(int,bool,INT&);
		void SetVoltage(int,float,INT&);
		void SetCurrentLimit(int,float,INT&);
		void SetBlink(int,bool,INT&);
		void SetInterlock(int,bool,INT&);
		bool AskPermissionToTurnOn(int);
		void SetStateChanged();
		void DemandVoltageChanged();
		void CurrentLimitChanged();
		void BlinkChanged();
		void InitODBArray();

	
		int reply_time_out;

		//local copy of hardware settings
		std::vector<int> supplyID;
		std::vector<std::string> idCode;
		std::vector<bool> interlock_enabled;
		

};


#endif
