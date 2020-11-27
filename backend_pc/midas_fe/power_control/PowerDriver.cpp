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
  
	//variables
	variables.connect("/Equipment/"+name+"/Variables");
  
	relevantchange=0.005; //only take action when values change more than this value
	return FE_SUCCESS;
}


INT PowerDriver::Connect()
{
	client = new TCPClient(settings["IP"],settings["port"],settings["reply timout"]);
	ss_sleep(100);
	std::string ip = settings["IP"];
	min_reply_length = settings["min reply"];
	
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
  
	cmd = "INST:NSEL " + std::to_string(ch)+ "\n";
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


bool PowerDriver::OPC()
{
	client->Write("*OPC?\n");
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	std::string reply;
	bool status = client->ReadReply(&reply,min_reply_length);
	return status;
}



void PowerDriver::Print()
{
	std::cout << "ODB settings: " << std::endl << settings.print() << std::endl;
	std::cout << "ODB variables: " << std::endl << variables.print() << std::endl;
}



// *****************   Read functions *************** //



float PowerDriver::Read(std::string cmd, INT& error)
{
	error = FE_SUCCESS;
	bool success;
	std::string reply;
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read after command %s", cmd.c_str());
		error = FE_ERR_DRIVER;		
	}
	float value = std::stof(reply);
	return value;
}



std::string PowerDriver::ReadIDCode(int channel, INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

	if(channel>=0) SelectChannel(channel);

	cmd = "*IDN?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read id supply with address %d", channel);
		error = FE_ERR_DRIVER;
	}
	
	return reply;
	
}


int PowerDriver::ReadESR(int channel, INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

	if(channel>=0) SelectChannel(channel);

	cmd = "*ESR?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read ESR supply with address %d", channel);
		error = FE_ERR_DRIVER;
	}
	int value = std::stoi(reply);
	return value;
}


bool PowerDriver::ReadState(int channel,INT& error)
{
	std::string cmd;
	bool success;
	std::string reply;
	error=FE_SUCCESS;
	bool value;
  
	if(channel>=0) SelectChannel(channel);
  
	cmd = "OUTP:STAT?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = client->ReadReply(&reply,min_reply_length);
	//std::cout << "here ********* , reply: " << reply << std::endl;
	
	if(!success)
	{
		cm_msg(MERROR, "power supply read ... ", "could not read %s state supply/channel: %d of", name.c_str(),channel);
		error = FE_ERR_DRIVER;
	}

	if(reply=="0") value=false;
	else if(reply=="1") value=true;
	else
	{ 
		cm_msg(MERROR, "power supply read ... ", "could not read %s valid state of supply/channel: %d", name.c_str(),channel);
		std::cout << "reply on state request = "<< reply << "." <<std::endl; 
		error = FE_ERR_DRIVER;
	}
	//std::cout << "here ********* , reply: " << reply << std::endl;
	return value; 
}


float PowerDriver::ReadVoltage(int channel,INT& error)
{
	error = FE_SUCCESS;
	float value = 0.0;
	if( SelectChannel(channel) )  {	  value = Read("MEAS:VOLT?\n",error);	}
		else error = FE_ERR_DRIVER;
	return value; 
}


float PowerDriver::ReadSetVoltage(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("VOLT?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float PowerDriver::ReadCurrent(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("MEAS:CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float PowerDriver::ReadCurrentLimit(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}




// ****************** Set functions ********************* //


bool PowerDriver::Set(std::string cmd, INT& error)
{
	bool success;
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = OPC();
	if(!success) { error=FE_ERR_DRIVER; cm_msg(MERROR, "Power supply ... ", "command %s not succesful for %s supply", cmd.c_str(),name.c_str() ); }
	return success;
}



void PowerDriver::SetCurrentLimit(int channel, float value,INT& error)
{
  error = FE_SUCCESS;
  
  if(value<-0.1 || value > 90.0) //check valid range 
  {
  	cm_msg(MERROR, "Power supply ... ", "current limit of %f not allowed",value );
  	variables["Current Limit"][channel]=currentlimit[channel]; //disable request
  	error=FE_ERR_DRIVER;
  	return;  	
  }
  
  if( SelectChannel(channel) ) //'channel' is already instrument channel
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



void PowerDriver::SetState(int channel, bool value,INT& error)
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



void PowerDriver::SetVoltage(int channel, float value,INT& error)
{
	error = FE_SUCCESS;
	if(value<-0.1 || value > 25.) //check valid range 
	{
		cm_msg(MERROR, "Power supply ... ", "voltage of %f not allowed",value );
		variables["Demand Voltage"][channel]=demandvoltage[channel]; //disable request
		error=FE_ERR_DRIVER;
		return;  	
	}
  
	if( SelectChannel(channel) ) // module address in the daisy chain to select channel, or 1/2/3/4 for the HAMEG
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



// ******************* Watch functions ******************** //

void PowerDriver::CurrentLimitChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<currentlimit.size(); i++)
	{
		float value = variables["Current Limit"][i];
		if( fabs(value-currentlimit[i]) > fabs(relevantchange*currentlimit[i]) ) //compare to local book keeping, look for significant change
		{
			SetCurrentLimit(instrumentID[i],value,err);   // 
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Power ... ", "changing %s current limit of channel %d to %f failed, error %d",name.c_str(), instrumentID[i],value,err);
			else
			{
				cm_msg(MINFO, "Power ... ", "changing %s current limit of channel %d to %f", name.c_str(),i,value);
				nChannelsChanged++;
				currentlimit[i]=value;
			}
		}	
	}	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Power ... ", "changing current limit request of %s rejected",name.c_str());
}



void PowerDriver::SetStateChanged()
{
	INT err;
	int nChannelsChanged = 0;
	
	for(unsigned int i=0; i<state.size(); i++)
	{
		bool value = variables["Set State"][i];
		if(value!=state[i]) //compare to local book keeping
		{
			SetState(instrumentID[i],value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Power ... ", "changing %s state of channel %d to %d failed, error %d", name.c_str(), i,value,err);
			else{ cm_msg(MINFO, "Power ... ", "changing %s state of channel %d to %d", name.c_str(),i,value);	nChannelsChanged++;	}
		}			
	}
	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Power ... ", "changing %s state request failed",name.c_str());
	else // read changes back from device 
	{
		int nChannels = instrumentID.size();
		for(int i = 0; i<nChannels; i++ ) 
		{
			bool value=ReadState(instrumentID[i],err);
			if(err==FE_SUCCESS) state[i]=value;
		} 
		variables["State"]=state; //push to odb
	}
}



void PowerDriver::DemandVoltageChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<voltage.size(); i++)
	{
		float value = variables["Demand Voltage"][i];
		if( fabs(value-voltage[i]) > fabs(relevantchange*voltage[i]) ) //compare to local book keeping, look for significant change
		{
			SetVoltage(instrumentID[i],value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Power ... ", "changing %s voltage of channel %d to %f failed, error %d", name.c_str(), i,value,err);
			else
			{
				cm_msg(MINFO, "Power ... ", "changing %s voltage of channel %d to %f", name.c_str(), i,value);
				nChannelsChanged++;
				demandvoltage[i]=value;
			}
		}			
	}	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Genesys supply ... ", "changing voltage request rejected");
}

