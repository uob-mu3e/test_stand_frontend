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
	nChannels=4;
}


INT HMP4040Driver::ConnectODB()
{
	InitODBArray();
	INT status = PowerDriver::ConnectODB();
	settings["port"](5025);
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
	
	//clear error an status registers
	cmd = "*CLS\n";
	if( !client->Write(cmd) ) cm_msg(MERROR, "Init genesys supply ... ", "could perform global clear %s", ip.c_str());
	else cm_msg(MINFO,"power_fe","Global CLS of %s",ip.c_str());
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	
	//HAMEG has fixed 4 channels
	settings["NChannels"] = nChannels;
	
	state.resize(nChannels);
	voltage.resize(nChannels);
	demandvoltage.resize(nChannels);
	current.resize(nChannels);
	currentlimit.resize(nChannels);
	
	idCode=ReadIDCode(err);	
	//read channels
	/*for(int i = 0; i<nChannels; i++ ) 
	{

  	
		state[i]=ReadState(i,err);
 	
		voltage[i]=ReadVoltage(i,err);
		demandvoltage[i]=ReadSetVoltage(i,err);

		current[i]=ReadCurrent(i,err);
		currentlimit[i]=ReadCurrentLimit(i,err);
  	
	 	if(err!=FE_SUCCESS) return err;  	
	}*/
	
	return FE_SUCCESS;
}


std::string HMP4040Driver::ReadIDCode(INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

	cmd = "*IDN?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,3,50);
	if(!success)
	{
		cm_msg(MERROR, "Genesys supply read ... ", "could not read id supply with ip %s", ip);
		error = FE_ERR_DRIVER;
	}
	
	return reply;	
}


bool HMP4040Driver::OPC()
{
	client->Write("*OPC?");
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	std::string reply;
	bool status = client->ReadReply(&reply,3,50);
	return status;
}

