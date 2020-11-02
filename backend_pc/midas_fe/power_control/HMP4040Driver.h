//****************************************************************************************
//
//	Base Driver for the LV power supplies. Use derived class fro TDK or HAMEG or .. supply
//
//	F.Wauters - Nov. 2020
//	

#ifndef HMP4040DRIVER_H
#define HMP4040DRIVER_H

#include "PowerDriver.h"

class HMP4040Driver : public PowerDriver {

	public:
	
		HMP4040Driver();
		~HMP4040Driver();
	
	private:
	
};

#endif
