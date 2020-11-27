#include "GenesysDriver.h"
#include <cstring>
#include <chrono>
#include <thread>

//
// MIDAS Driver for our TDK Genesys supplies
// FW, Sep. 2020
// 
// communicate with SCPI commands
// Setup for 1 supply connected with LAN, the other supplies daisy chained with the RS485 bus
// supply IDs are setup as 0,1,2,...
// 
// From manual:
// * Recommended time delay between commands: 5mSec minimum. Some commands might require longer time. In such cases, refer to NOTE following command description.
// 
GenesysDriver::GenesysDriver(std::string n, EQUIPMENT_INFO* inf) : PowerDriver(n,inf)
{
	std::cout << " GenesysDriver instantiated " << std::endl;

	//device specific constants
}


GenesysDriver::~GenesysDriver()
{

}


INT GenesysDriver::ConnectODB()
{
	InitODBArray();
	INT status = PowerDriver::ConnectODB();
	settings["port"](8003);
	settings["reply timout"](50);
	settings["min reply"](3); //minimum reply , a char + "\n"
	if(false) return FE_ERR_ODB;  
	return FE_SUCCESS;  
}


//didn't find a cleaner way to init the array entries, have a prior initialized 'variables', and not overwrite Set values
void GenesysDriver::InitODBArray()
{
	midas::odb settings_array = { {"Names",std::array<std::string,16>()} , {"Blink",std::array<bool,16>()} };
	settings_array.connect("/Equipment/"+name+"/Settings");
}



INT GenesysDriver::Init()
{	
	std::string ip = settings["IP"];
	std::cout << "Call init on " << ip << std::endl;
	std::string cmd = "";
	std::string reply = "";
	INT err;
	
	//global reset if requested
  if( settings["Global Reset On FE Start"] )
  {
  	cmd = "GLOB:*RST\n";
    if( !client->Write(cmd) ) cm_msg(MERROR, "Init genesys supply ... ", "could not global reset %s", ip.c_str());
    else cm_msg(MINFO,"power_fe","Init global reset of %s",ip.c_str());
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  
  //clear error an status registers
  cmd = "GLOB:*CLS\n";
  if( !client->Write(cmd) ) cm_msg(MERROR, "Init genesys supply ... ", "could perform global clear %s", ip.c_str());
  else cm_msg(MINFO,"power_fe","Global CLS of %s",ip.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  
  //figure out the number of supplies connected
  for(int i = 0; i < 12; i++)
  {
  	cmd = "INST:NSEL " + std::to_string(i) + "\n";
  	client->Write(cmd);
  	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
		bool success = OPC();
		if(!success) cm_msg(MINFO,"power_fe","Supply nr %d NOT detected",i);
		else
		{
			cm_msg(MINFO,"power_fe","Supply nr %d seen",i); 
			instrumentID.push_back(i);
		}		
  }
 
  cm_msg(MINFO,"power_fe"," %d supplies registered in daisy chain",int(instrumentID.size()));
  int nChannels = instrumentID.size();
  if(nChannels != settings["NChannels"]) cm_msg(MINFO,"power_fe"," Different as the set ODB value of %d, updating",int(settings["NChannels"]));
  settings["NChannels"] = instrumentID.size();

	// ***** channel by channel settings ***** //
  
	settings["Address"] = instrumentID;
	state.resize(nChannels);
	voltage.resize(nChannels);
	demandvoltage.resize(nChannels);
	current.resize(nChannels);
	currentlimit.resize(nChannels);
	idCode.resize(nChannels);
	interlock_enabled.resize(nChannels);
  
	for(int i = 0; i<nChannels; i++ ) 
	{
		idCode[i]=ReadIDCode(instrumentID[i],err);	
  	
		state[i]=ReadState(instrumentID[i],err);
 	
		voltage[i]=ReadVoltage(instrumentID[i],err);
		demandvoltage[i]=ReadSetVoltage(instrumentID[i],err);

		current[i]=ReadCurrent(instrumentID[i],err);
		currentlimit[i]=ReadCurrentLimit(instrumentID[i],err);

		interlock_enabled[i]=true;
		SetInterlock(instrumentID[i],interlock_enabled[i],err);
  	
		if(err!=FE_SUCCESS) return err;  	
	}
  
	settings["Identification Code"]=idCode;
  
	variables["State"]=state; //push to odb
	variables["Set State"]=state; //the init function can not change the on/off state of the supply
  
 	variables["Voltage"]=voltage;
 	variables["Demand Voltage"]=demandvoltage;
 	
 	variables["Current"]=current;
 	variables["Current Limit"]=currentlimit;
 	
 	// user arrays.
	settings["Names"].resize(nChannels);
 	settings["Blink"].resize(nChannels);
 	
  
	// ***** set up watch ***** //
	variables["Set State"].watch(  [&](midas::odb &arg) { this->SetStateChanged(); }  );
	variables["Demand Voltage"].watch(  [&](midas::odb &arg) { this->DemandVoltageChanged(); }  );
	variables["Current Limit"].watch(  [&](midas::odb &arg) { this->CurrentLimitChanged(); }  );
	
	settings["Blink"].watch(  [&](midas::odb &arg) { this->BlinkChanged(); }  );
 



	return FE_SUCCESS;
}





bool GenesysDriver::AskPermissionToTurnOn(int channel) //extra check whether it is safe to tunr on supply;
{
	return true;
}



// ************ watch functions ************* //


void GenesysDriver::BlinkChanged()
{
	INT err;
	int nChannelsChanged = 0;
	
	for(unsigned int i=0; i< instrumentID.size(); i++)
	{
		bool value = settings["Blink"][i];		
		SetBlink(i,value,err);
		if(err!=FE_SUCCESS ) cm_msg(MERROR, "Genesys supply ... ", "changing flashing of channel %d to %d failed, error %d", i,value,err);
	}
	
}




// **************  Set Functions ************** //

void GenesysDriver::SetInterlock(int channel,bool value, INT& error)
{
	std::string cmd;
	bool success;
	error = FE_SUCCESS;

	if( SelectChannel(instrumentID[channel]) )
	{
		//OUTPut:ILC[:STATe] <Bool>
 		if(value==true) { cmd="OUTP:ILC 1\n"; }
		else { cmd = "OUTP:ILC 0\n"; }
		client->Write(cmd);
		std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  		success = OPC();
  		if(!success) error=FE_ERR_DRIVER;
		else cm_msg(MINFO, "Genesys supply ... ", "Interlock enabled[1]/disabled[0]: ",value );
	}	
}



void GenesysDriver::SetBlink(int channel, bool value,INT& error)
{
	std::string cmd;
	bool success;
  error = FE_SUCCESS;
  
  if( SelectChannel(instrumentID[channel]) )
  {
 		if(value==true) { cmd="DISP:WIND:FLAS 1\n"; }
		else { cmd = "DISP:WIND:FLAS 0\n"; }
		client->Write(cmd);
		std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  	success = OPC();
  	if(!success) error=FE_ERR_DRIVER;
  }	
}



// **************  Read Functions ************** //



INT GenesysDriver::ReadAll()
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


