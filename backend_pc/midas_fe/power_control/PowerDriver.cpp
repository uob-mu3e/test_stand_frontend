#include "PowerDriver.h"

PowerDriver::PowerDriver()
{
	std::cout << "Warning: empty base class instantiated" << std::endl;
}

PowerDriver::PowerDriver(std::string n, EQUIPMENT_INFO* inf)
{
	name=n;
	info=inf;
	initialized=false;
}


INT PowerDriver::ConnectODB()
{
	//general settings
	settings.connect("/Equipment/"+name+"/Settings");
	settings["IP"]("10.10.10.10");
	settings["NChannels"](2);
	settings["Global Reset On FE Start"](true);
	settings["Enable"](false);
  
	//variables
	variables.connect("/Equipment/"+name+"/Variables");
  
	relevantchange=0.005; //only take action when values change more than this value
	return FE_SUCCESS;
}


INT PowerDriver::Connect()
{
	client = new TCPClient(settings["IP"],settings["port"]);
	ss_sleep(100);
	std::string ip = settings["IP"];
	
	if(!client->Connect())
	{
		cm_msg(MERROR, "Connect to power supply ... ", "could not connect to %s", ip.c_str()); 
		return FE_ERR_HW;
	}		
	else cm_msg(MINFO,"power_fe","Init Connection to %s alive",ip.c_str());
	
  return FE_SUCCESS;
}


bool PowerDriver::Enabled()
{
	midas::odb common("/Equipment/"+name+"/Common");
	bool value = common["Enabled"];
	return value;
}


bool PowerDriver::SelectChannel(int ch)
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
		cm_msg(MERROR,"power_fe","Not able to select channel %d ",ch);
		return false;
	}
	return true;
}
