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
GenesysDriver::GenesysDriver(std::string n, EQUIPMENT_INFO* inf)
{
	std::cout << " GenesysDriver instantiated " << std::endl;
	info = inf;
	name = n;
}



GenesysDriver::~GenesysDriver()
{

}



INT GenesysDriver::ConnectODB()
{
  //general settings
  settings.connect("/Equipment/"+name+"/Settings");
  settings["IP"]("10.10.10.10");
  settings["port"](8003);
  settings["NChannels"](2);
  settings["Global Reset On FE Start"](true);
  
  //variables
  variables.connect("/Equipment/"+name+"/Variables");
  
  relevantchange=0.005; //only take action when values change more than this value

  InitODBArray();
   
  if(false) return FE_ERR_ODB;
  
  return FE_SUCCESS;  
}

//didn't find a cleaner way to init the array entries, have a prior initialized 'variables', and not overwrite Set values
void GenesysDriver::InitODBArray()
{
	midas::odb settings_array = { {"Names",std::array<std::string,16>()} , {"Blink",std::array<bool,16>() } };
  settings_array.connect("/Equipment/"+name+"/Settings");
}



INT GenesysDriver::Connect()
{
	client = new TCPClient(settings["IP"],settings["port"]);
	ss_sleep(100);
	std::string ip = settings["IP"];
	
	if(!client->Connect())
	{
		cm_msg(MERROR, "Connect to genesys supply ... ", "could not connect to %s", ip.c_str()); 
		return FE_ERR_HW;
	}		
	else cm_msg(MINFO,"power_fe","Init Connection to %s alive",ip.c_str());
	
  return FE_SUCCESS;
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
  cmd = "GLOB:*RST\n";
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
			supplyID.push_back(i);
		}		
  }
 
  cm_msg(MINFO,"power_fe"," %d supplies registered in daisy chain",int(supplyID.size()));
  int nChannels = supplyID.size();
  if(nChannels != settings["NChannels"]) cm_msg(MINFO,"power_fe"," Different as the set ODB value of %d, updating",int(settings["NChannels"]));
  settings["NChannels"] = supplyID.size();

  // ***** channel by channel settings ***** //
  
  settings["Address"] = supplyID;
  state.resize(nChannels);
  voltage.resize(nChannels);
  demandvoltage.resize(nChannels);
  current.resize(nChannels);
  currentlimit.resize(nChannels);
  idCode.resize(nChannels);
  
  for(int i = 0; i<nChannels; i++ ) 
  {
  	idCode[i]=ReadIDCode(i,err);	
  	
    state[i]=ReadState(i,err);
 	
  	voltage[i]=ReadVoltage(i,err);
  	demandvoltage[i]=ReadSetVoltage(i,err);

  	current[i]=ReadCurrent(i,err);
  	currentlimit[i]=ReadCurrentLimit(i,err);
  	
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
	
	variables["Blink"].watch(  [&](midas::odb &arg) { this->BlinkChanged(); }  );
 



	return FE_SUCCESS;
}



void GenesysDriver::Print()
{
	std::cout << "ODB settings: " << std::endl << settings.print() << std::endl;
	std::cout << "ODB variables: " << std::endl << variables.print() << std::endl;
}



bool GenesysDriver::OPC()
{
	client->Write("*OPC?");
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	std::string reply;
	bool status = client->ReadReply(&reply,3,50);
	return status;
}



bool GenesysDriver::AskPermissionToTurnOn(int channel) //extra check whether it is safe to tunr on supply;
{
	return true;
}



bool GenesysDriver::SelectChannel(int ch)
{
	std::string cmd;
  bool success;
  std::string reply;
  
	cmd = "INST:NSEL " + std::to_string(supplyID[ch])+ "\n";
	client->Write(cmd);
  std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  success = OPC();
  if(!success)
  {
		cm_msg(MERROR,"power_fe","Not able to select Supply %d ",supplyID[ch]);
  	return false;
  } 
  return true;
}


// ************ watch functions ************* //

void GenesysDriver::SetStateChanged()
{
	INT err;
	int nChannelsChanged = 0;
	
	for(unsigned int i=0; i<state.size(); i++)
	{
		bool value = variables["Set State"][i];
		if(value!=state[i]) //compare to local book keeping
		{
			SetState(i,value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Genesys supply ... ", "changing state of channel %d to %d failed, error %d", i,value,err);
			else{ cm_msg(MINFO, "Genesys supply ... ", "changing state of channel %d to %d", i,value);	nChannelsChanged++;	}
		}			
	}
	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Genesys supply ... ", "changing state request failed");
	else // read changes back from device 
	{	
		int nChannels = supplyID.size();
		for(int i = 0; i<nChannels; i++ ) 
		{
			bool value=ReadState(i,err);
	    if(err==FE_SUCCESS) state[i]=value;
	  }  
	  variables["State"]=state; //push to odb
	}
}



void GenesysDriver::DemandVoltageChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<voltage.size(); i++)
	{
		float value = variables["Demand Voltage"][i];
		if( fabs(value-voltage[i]) > fabs(relevantchange*voltage[i]) ) //compare to local book keeping, look for significant change
		{
			SetVoltage(i,value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Genesys supply ... ", "changing voltage of channel %d to %f failed, error %d", i,value,err);
			else
			{
				cm_msg(MINFO, "Genesys supply ... ", "changing voltage of channel %d to %f", i,value);
				nChannelsChanged++;
				demandvoltage[i]=value;
			}
		}			
	}	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Genesys supply ... ", "changing voltage request rejected");
}



void GenesysDriver::CurrentLimitChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<currentlimit.size(); i++)
	{
		float value = variables["Current Limit"][i];
		if( fabs(value-currentlimit[i]) > fabs(relevantchange*currentlimit[i]) ) //compare to local book keeping, look for significant change
		{
			SetCurrentLimit(i,value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Genesys supply ... ", "changing current limit of channel %d to %f failed, error %d", i,value,err);
			else
			{
				cm_msg(MINFO, "Genesys supply ... ", "changing current limit of channel %d to %f", i,value);
				nChannelsChanged++;
				currentlimit[i]=value;
			}
		}			
	}	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Genesys supply ... ", "changing current limit request rejected");
}


void GenesysDriver::BlinkChanged()
{
}




// **************  Set Functions ************** //

bool GenesysDriver::Set(std::string cmd, INT& error)
{
	bool success;
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  success = OPC();
  if(!success) { error=FE_ERR_DRIVER; cm_msg(MERROR, "Genesys supply ... ", "command %s not succesful", cmd.c_str() ); }
  return success;
}

void GenesysDriver::SetVoltage(int channel, float value,INT& error)
{
  error = FE_SUCCESS;
  if(value<-0.1 || value > 25.) //check valid range 
  {
  	cm_msg(MERROR, "Genesys supply ... ", "voltage of %f not allowed",value );
  	variables["Demand Voltage"][channel]=demandvoltage[channel]; //disable request
  	error=FE_ERR_DRIVER;
  	return;  	
  }
  
  if( SelectChannel(channel) )
  {
	  bool success = Set("VOLT "+std::to_string(value)+"\n",error);
  	if(!success) error=FE_ERR_DRIVER;
  	else // read changes
  	{
	  	voltage[channel]=ReadVoltage(channel,error);
	  	variables["Voltage"][channel]=voltage[channel];
	  	current[channel]=ReadCurrent(channel,error);
	  	variables["Current"][channel]=current[channel];
  	}		
  }
  else error=FE_ERR_DRIVER;
}

void GenesysDriver::SetCurrentLimit(int channel, float value,INT& error)
{
  error = FE_SUCCESS;
  if(value<-0.1 || value > 90.0) //check valid range 
  {
  	cm_msg(MERROR, "Genesys supply ... ", "current limit of %f not allowed",value );
  	variables["Current Limit"][channel]=currentlimit[channel]; //disable request
  	error=FE_ERR_DRIVER;
  	return;  	
  }
  
  if( SelectChannel(channel) )
  {
	  bool success = Set("CURR "+std::to_string(value)+"\n",error);
  	if(!success) error=FE_ERR_DRIVER;
  	else // read changes
  	{
	  	voltage[channel]=ReadVoltage(channel,error);
	  	variables["Voltage"][channel]=voltage[channel];
  	}		
  }
  else error=FE_ERR_DRIVER;
}

void GenesysDriver::SetState(int channel, bool value,INT& error)
{
	std::string cmd;
	bool success;
  error = FE_SUCCESS;
  std::cout << " **** Request to set channel " << channel << " to : " << value << std::endl;   
  if(value==true)
  {
  	if(!AskPermissionToTurnOn(channel))
  	{
  		cm_msg(MERROR, "Genesys supply ... ", "changing of channel %d not allowed",channel );
  		variables["Set State"][channel]=false; //disable request
  		error=FE_ERR_DISABLED;
  		return;
  	}
  }
  
  if( SelectChannel(channel) )
  {
		if(value==true) { cmd="OUTP:STAT 1\n"; }
		else { cmd = "OUTP:STAT 0\n"; }
  	client->Write(cmd);
		std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  	success = OPC();
  	if(!success) error=FE_ERR_DRIVER;  	  		
  }
  else error=FE_ERR_DRIVER;
}



// **************  Read Functions ************** //

float GenesysDriver::Read(std::string cmd, INT& error)
{
  error = FE_SUCCESS;
  bool success;
  std::string reply;
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  success = client->ReadReply(&reply,3,50);
  if(!success)
  {
		cm_msg(MERROR, "Genesys supply read ... ", "could not read after command %s", cmd.c_str());
		error = FE_ERR_DRIVER;		
	}
	float value = std::stof(reply);
	return value;
}


float GenesysDriver::ReadSetVoltage(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("VOLT?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float GenesysDriver::ReadVoltage(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("MEAS:VOLT?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float GenesysDriver::ReadCurrent(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("MEAS:CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float GenesysDriver::ReadCurrentLimit(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}



bool GenesysDriver::ReadState(int channel,INT& error)
{
  std::string cmd;
  bool success;
  std::string reply;
  error=FE_SUCCESS;
  bool value;
  
  SelectChannel(channel);
  
  cmd = "OUTP:STAT?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
  success = client->ReadReply(&reply,3,50);
  if(!success)
  {
		cm_msg(MERROR, "Genesys supply read ... ", "could not read state supply with address %d", int(supplyID[channel]));
		error = FE_ERR_DRIVER;
	}

  if(reply=="0") value=false;
  else if(reply=="1") value=true;
  else
  { 
  	cm_msg(MERROR, "Genesys supply read ... ", "could not read state of supply with address %d", int(supplyID[channel]));
	  std::cout << "reply on state request = "<< reply << "." <<std::endl; 
		error = FE_ERR_DRIVER;
	}
  return value; 
}

std::string GenesysDriver::ReadIDCode(int channel, INT& error)
{
  std::string cmd;
  bool success;
  std::string reply="";
  error=FE_SUCCESS;

  SelectChannel(channel);

  cmd = "*IDN?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,3,50);
  if(!success)
  {
		cm_msg(MERROR, "Genesys supply read ... ", "could not read id supply with address %d", int(supplyID[channel]));
		error = FE_ERR_DRIVER;
	}
	
	return reply;
	
}

INT GenesysDriver::ReadAll()
{
  int nChannels = supplyID.size();
  INT err;
  //update local book keeping
	for(unsigned int i=0; i<nChannels; i++)
	{
		bool bvalue = ReadState(i,err);
	  if(state[i]!=bvalue) //only update odb if there is a change
	  {
	  	state[i]=bvalue;
	   	variables["State"][i]=bvalue;
	  }
 	
 		float fvalue = ReadVoltage(i,err);
  	if( fabs(voltage[i]-fvalue) > fabs(relevantchange*voltage[i]) )
  	{
  		voltage[i]=fvalue;
  		variables["Voltage"][i]=fvalue;	  	
  	}
  	
  	fvalue = ReadCurrent(i,err);
  	if( fabs(current[i]-fvalue) > fabs(relevantchange*current[i]) )
  	{
  		current[i]=fvalue;
  		variables["Current"][i]=fvalue;	  	
  	}
  	
	 	if(err!=FE_SUCCESS) return err;		
	}	
	return FE_SUCCESS;
}

