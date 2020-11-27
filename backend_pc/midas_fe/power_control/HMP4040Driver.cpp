#include "HMP4040Driver.h"

HMP4040Driver::HMP4040Driver()
{
}


HMP4040Driver::~HMP4040Driver()
{
}


HMP4040Driver::HMP4040Driver(std::string n, EQUIPMENT_INFO* inf) : PowerDriver(n,inf)
{
	std::cout << " HMP4040 HAMEG driver instantiated " << std::endl;
	
	//device specific constants
	nChannels=4;
}


INT HMP4040Driver::ConnectODB()
{
	InitODBArray();
	INT status = PowerDriver::ConnectODB();
	settings["port"](5025);
	settings["reply timout"](300);
	settings["min reply"](2); //minimum reply , 2 chars , not 3 (not fully figured out why)
	if(false) return FE_ERR_ODB;
}


void HMP4040Driver::InitODBArray()
{
	midas::odb settings_array = { {"Names",std::array<std::string,4>()} };
	settings_array.connect("/Equipment/"+name+"/Settings");
}


INT HMP4040Driver::Init()
{
	ip = settings["IP"];
	std::cout << "Call init on " << ip << std::endl;
	std::string cmd = "";
	std::string reply = "";
	INT err;
	
	//longer wait time for the HMP supplies
	client->SetDefaultWaitTime(50);
	
	//global reset if requested
	if( settings["Global Reset On FE Start"] )
	{
		cmd = "*RST\n";
		if( !client->Write(cmd) ) cm_msg(MERROR, "Init HAMEG supply ... ", "could not global reset %s", ip.c_str());
		else cm_msg(MINFO,"power_fe","Init global reset of %s",ip.c_str());
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	
	//beep
	cmd="SYST:BEEP\n";
	if( !client->Write(cmd) ) cm_msg(MERROR, "Init HAMEG supply ... ", "could not beep %s", ip.c_str());
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	
	//clear error an status registers
	cmd = "*CLS\n";
	if( !client->Write(cmd) ) cm_msg(MERROR, "Init HAMEG supply ... ", "could perform global clear %s", ip.c_str());
	else cm_msg(MINFO,"power_fe","Global CLS of %s",ip.c_str());
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	
	//HAMEG has fixed 4 channels
	settings["NChannels"] = nChannels;
	
	voltage.resize(nChannels);
	demandvoltage.resize(nChannels);
	current.resize(nChannels);
	currentlimit.resize(nChannels);
	state.resize(nChannels);
	instrumentID = {1,2,3,4}; // The HMP4040 supply has 4 channel numbered 1,2,3, and 4.
	
	idCode=ReadIDCode(-1,err); 	//channel selection not relevant for HAMEG supply to read ID
								// "-1" is a trick not to select a channel before the query
								
	std::cout << "ID code: " << idCode << std::endl;
								
	//client->FlushQueu();
		
	//read channels
	for(int i = 0; i<nChannels; i++ ) 
	{ 	
		state[i]=ReadState(instrumentID[i],err);
		
		voltage[i]=ReadVoltage(instrumentID[i],err);
		demandvoltage[i]=ReadSetVoltage(instrumentID[i],err);

		current[i]=ReadCurrent(instrumentID[i],err);
		currentlimit[i]=ReadCurrentLimit(instrumentID[i],err);
  	
	 	if(err!=FE_SUCCESS) return err;  	
	}
	
	settings["Identification Code"]=idCode;
	
	variables["State"]=state; //push to odb
	variables["Set State"]=state; //the init function can not change the on/off state of the supply
  
 	variables["Voltage"]=voltage;
 	variables["Demand Voltage"]=demandvoltage;
 	
 	variables["Current"]=current;
 	variables["Current Limit"]=currentlimit;
 	
 	//watch functions
 	variables["Current Limit"].watch(  [&](midas::odb &arg) { this->CurrentLimitChanged(); }  );
 	variables["Set State"].watch(  [&](midas::odb &arg) { this->SetStateChanged(); }  );
	variables["Demand Voltage"].watch(  [&](midas::odb &arg) { this->DemandVoltageChanged(); }  );
 	
	return FE_SUCCESS;
}



bool HMP4040Driver::AskPermissionToTurnOn(int channel) //extra check whether it is safe to tunr on supply;
{
	return true;
}


INT HMP4040Driver::ReadAll()
{
	int nChannels = instrumentID.size();
	INT err;
	
	//update local book keeping
	for(int i=0; i<nChannels; i++)
	{
		bool bvalue = ReadState(instrumentID[i],err);
		if(state[i]!=bvalue) //only update odb if there is a change
		{
			state[i]=bvalue;
			variables["State"][i]=bvalue;
		}
 	
 		float fvalue = ReadVoltage(instrumentID[i],err);
		if( fabs(voltage[i]-fvalue) > fabs(relevantchange*voltage[i]) )
		{
			voltage[i]=fvalue;
			variables["Voltage"][i]=fvalue;	  	
		}
  	
		fvalue = ReadCurrent(instrumentID[i],err);
		if( fabs(current[i]-fvalue) > fabs(relevantchange*current[i]) )
		{
			current[i]=fvalue;
			variables["Current"][i]=fvalue;	  	
		}
  	
	 	if(err!=FE_SUCCESS) return err;		
	}	
	return FE_SUCCESS;
}

//************************************************************************************
//** the STATE and SELECT OUTPUT on of is a bit confusion: from the manual

/*
 
OUTPut:SELect {OFF | ON | 0 | 1}
Activates or deactivates the previous selected channel. If the channel is activated the channel
LED lights up green in CV (constant voltage) mode or red in CC (constant current) mode.
Parameters:		
ON | 1 Channel will be activated
			 OFF | 0 Channel will be deactivated
			 *RST: OFF | 0
OUTPut[:STATe] {OFF | ON | 0 | 1}
Activates or deactivates the previous selected channel and turning on the output. The selected
channel LED lights up green. If the output will be turned of with OUTP OFF only the previous
selected channel will be deactivated. After sending OUTP OFF command the output button is
still activated.
Parameters:		
ON | 1 Channel and output will be activated
			 OFF | 0 Channel will be deactivated
			 *RST: OFF | 0
Example:
INST OUT1
OUTP ON (= channel 1 and output will be activated; channel and output LED will light up)
OUTPut[:STATe]?
Queries the output state.
Return values:		
1
			 0
ON - output is activated
OFF - output is deactivated
26SCPI Commands HMP series
Remote Control
Command Reference
OUTPut:GENeral {OFF | ON | 0 | 1}
Turning on / off all previous selcted channels simultaneously.
Parameters:		
ON | 1 Channels and output will be activated
			 OFF | 0 Channels will be deactivated
Example:
INST OUT1
Volt 12
Curr 0.1
OUTP:SEL ON		 CH1 LED lights up green
INST OUT2
Volt 12
Curr 0.2
OUTP:SEL ON		 CH2 LED lights up green
OUTP:GEN ON		 Channels will be activated simultaneously

*/




