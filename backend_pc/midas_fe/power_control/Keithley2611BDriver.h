//****************************************************************************************
//
//	Base Driver for the LV power supplies. Use derived class fro TDK or HAMEG or .. supply
//
//	F.Wauters - Nov. 2020
//	

#ifndef Keithley2611BDriver_H
#define Keithley2611BDriver_H

#include "PowerDriver.h"

class Keithley2611BDriver : public PowerDriver {

	public:
	
        Keithley2611BDriver();
        Keithley2611BDriver(std::string n, EQUIPMENT_INFO* inf);
        ~Keithley2611BDriver();
		
		INT ConnectODB();
		INT Init();
		INT ReadAll();
	
        std::string GenerateCommand(COMMAND_TYPE cmdt, float val) override;

    private:
	
		void InitODBArray();
		bool AskPermissionToTurnOn(int);
		std::string idCode;
		std::string ip;

		//watch
		void ReadESRChanged();

	
};

#endif
