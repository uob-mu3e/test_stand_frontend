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
  state={false,false};
  variables["State"]=state;
  variables["Set State"]=state;
   
  if(false) return FE_ERR_ODB;
  
  return FE_SUCCESS;  
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
  
  //query the number of supplies connected
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
  state.resize(supplyID.size());
  demandstate.resize(supplyID.size());
  
  for(int i = 0; i<nChannels; i++ ) 
  {
    // read current state
    bool value=ReadState(i,err);
    state[i]=value;
  	demandstate[i]=value;
  	if(err!=FE_SUCCESS) return err;
  }
  
  variables["State"]=state; //push to odb
  variables["Set State"]=state;
  
	// ***** set up watch ***** //
	variables["Set State"].watch(  [&](midas::odb &arg) { this->SetStateChanged(); }  );
 

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
  	cm_msg(MERROR, "Genesys supply read ... ", "could not read state supply with address %d", int(supplyID[channel]));
	  std::cout << "reply on state request = "<< reply << "." <<std::endl; 
		error = FE_ERR_DRIVER;
	}  
  return value; 
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



void GenesysDriver::SetStateChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<demandstate.size(); i++)
	{
		bool value = variables["Set State"][i];
		if(value!=state[i]) //compare to local book keeping
		{
			SetState(i,value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Genesys supply ... ", "changing state of channel %d to %d failed, error", i,value,err);
			else{ cm_msg(MINFO, "Genesys supply ... ", "changing state of channel %d to %d", i,value);	nChannelsChanged++;	}
		}			
	}
	if(nChannelsChanged < 1) cm_msg(MINFO, "Genesys supply ... ", "changing state request failed");
	
	//read back state
	int nChannels = supplyID.size();
	for(int i = 0; i<nChannels; i++ ) 
  {
    bool value=ReadState(i,err);
    if(err==FE_SUCCESS) state[i]=value;
  }  
  variables["State"]=state; //push to odb

  
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
  		cm_msg(MINFO, "Genesys supply ... ", "changing of channel %d not allowed",channel );
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



bool GenesysDriver::AskPermissionToTurnOn(int channel) //extra check whether it is safe to tunr on supply;
{
	return true;
}

