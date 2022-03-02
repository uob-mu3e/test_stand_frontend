#include "Keithley2450Driver.h"
#include <thread>


Keithley2450Driver::Keithley2450Driver()
{

}


Keithley2450Driver::~Keithley2450Driver()
{
}


Keithley2450Driver::Keithley2450Driver(std::string n, EQUIPMENT_INFO* inf) : PowerDriver(n,inf)
{
    std::cout << " Keithley2611B driver with " << instrumentID.size() << " channels instantiated " << std::endl;
}


INT Keithley2450Driver::ConnectODB()
{
	InitODBArray();
    PowerDriver::ConnectODB();
	settings["port"](5025);
	settings["reply timout"](300);
	settings["min reply"](2); //minimum reply , 2 chars , not 3 (not fully figured out why)
	settings["ESR"](0);
	settings["Max Voltage"](2);
	if(false) return FE_ERR_ODB;
    return FE_SUCCESS;
}


void Keithley2450Driver::InitODBArray()
{
	midas::odb settings_array = { {"Channel Names",std::array<std::string,4>()} };
	settings_array.connect("/Equipment/"+name+"/Settings");
}


INT Keithley2450Driver::Init()
{
	ip = settings["IP"];
    std::cout << "Call init on " << ip << std::endl;
	std::string cmd = "";
	std::string reply = "";
	INT err;
	
    //longer wait time for the HMP supplies //TODO What abut Keithly?
    client->SetDefaultWaitTime(200);
	
	//global reset if requested
    if(settings["Global Reset On FE Start"] == true)
	{
		cmd = "*RST\n";
        if( !client->Write(cmd) ) cm_msg(MERROR, "Init KEITH supply ... ", "could not global reset %s", ip.c_str());
		else cm_msg(MINFO,"power_fe","Init global reset of %s",ip.c_str());
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	

    cmd=GenerateCommand(COMMAND_TYPE::Beep, 0);
    if( !client->Write(cmd) ) cm_msg(MERROR, "Init KEITH supply ... ", "could not beep %s", ip.c_str());
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	
    std::vector<std::string> error_queue = ReadErrorQueue(-1,err);
	for(auto& s : error_queue)
	{
        if(s.find("No error") == std::string::npos)		{	cm_msg(MERROR,"power_fe"," Error from KEITH supply : %s",s.c_str());		}
    } //TOFIX: what's wrong???
	
	
    //KEITH has 1 channel
    instrumentID = {1};
	int nChannels = instrumentID.size();	
	settings["NChannels"] = nChannels;
	
	voltage.resize(nChannels);
	demandvoltage.resize(nChannels);
	current.resize(nChannels);
	currentlimit.resize(nChannels);
	state.resize(nChannels);
	OVPlevel.resize(nChannels);
    //instrumentID = {1,2,3,4}; // The HMP4040 supply has 4 channel numbered 1,2,3, and 4.
	
	idCode=ReadIDCode(-1,err); 	//channel selection not relevant for HAMEG supply to read ID
								// "-1" is a trick not to select a channel before the query
								
	std::cout << "ID code: " << idCode << std::endl;
								
	//client->FlushQueu();
		
	//read channels
	for(int i = 0; i<nChannels; i++ ) 
	{ 	
		state[i]=ReadState(i,err);
		voltage[i]=ReadVoltage(i,err);
        demandvoltage[i]=ReadSetVoltage(i,err);//T

		current[i]=ReadCurrent(i,err);
		currentlimit[i]=ReadCurrentLimit(i,err);
		
        //OVPlevel[i]=ReadOVPLevel(i,err);
        OVPlevel[i]=20;
  	
	 	if(err!=FE_SUCCESS) return err;  	
	}
	
	settings["Identification Code"]=idCode;
	settings["ESR"]=ReadESR(-1,err);
	settings["Read ESR"]=false;
	
	variables["State"]=state; //push to odb
	variables["Set State"]=state; //the init function can not change the on/off state of the supply
  
 	variables["Voltage"]=voltage;
 	variables["Demand Voltage"]=demandvoltage;
 	
 	variables["Current"]=current;
 	variables["Current Limit"]=currentlimit;
 	
 	variables["OVP Level"]=OVPlevel;
 	variables["Demand OVP Level"]=OVPlevel;
 	
 	//watch functions
    variables["Current Limit"].watch(  [&](midas::odb &arg [[maybe_unused]]) { this->CurrentLimitChanged(); }  );
    variables["Set State"].watch(  [&](midas::odb &arg  [[maybe_unused]]) { this->SetStateChanged(); }  );
    variables["Demand Voltage"].watch(  [&](midas::odb &arg  [[maybe_unused]]) { this->DemandVoltageChanged(); }  );
    variables["Demand OVP Level"].watch(  [&](midas::odb &arg  [[maybe_unused]]) { this->DemandOVPLevelChanged(); }  );


    settings["Read ESR"].watch(  [&](midas::odb &arg  [[maybe_unused]]) { this->ReadESRChanged(); }  );
 	
	return FE_SUCCESS;
}


INT Keithley2450Driver::ReadAll()
{
	INT err;
	INT err_accumulated;
	int nChannels = instrumentID.size();
	//update local book keeping
	for(int i=0; i<nChannels; i++)
	{
		bool bvalue = ReadState(i,err);
		err_accumulated = err;
		if(state[i]!=bvalue) //only update odb if there is a change
		{
			state[i]=bvalue;
			variables["State"][i]=bvalue;
		}
 	
 		float fvalue = ReadVoltage(i,err);
		err_accumulated = err_accumulated | err;
		if( fabs(voltage[i]-fvalue) > fabs(relevantchange*voltage[i]) )
		{
			voltage[i]=fvalue;
			variables["Voltage"][i]=fvalue;	  	
		}
  	
		fvalue = ReadCurrent(i,err);
		err_accumulated = err_accumulated | err;
		if( fabs(current[i]-fvalue) > fabs(relevantchange*current[i]) )
		{
			current[i]=fvalue;
			variables["Current"][i]=fvalue;	  	
		}
		
		fvalue = ReadOVPLevel(i,err);
		err_accumulated = err_accumulated | err;
		if( fabs(OVPlevel[i]-fvalue) > fabs(relevantchange*OVPlevel[i]) )
		{
			OVPlevel[i]=fvalue;
			variables["OVP Level"][i]=fvalue;	  	
		}
  	
  	
	 	if(err_accumulated!=FE_SUCCESS) return err_accumulated & 0xFFFE; //remove the success bit if there is any		
	}
	
    std::vector<std::string> error_queue = ReadErrorQueue(-1,err);
	for(auto& s : error_queue)
	{
        if(s.find("No error") == std::string::npos)		{	cm_msg(MERROR,"power_fe"," Error from KEITH supply : %s",s.c_str());		}
    } //TOFIX: what's wrong???
	
	return FE_SUCCESS;
}


void Keithley2450Driver::ReadESRChanged()
{
	INT err;
	bool value = settings["Read ESR"];
	if(value)
	{
		settings["ESR"] = ReadESR(-1,err);		
		settings["Read ESR"]=false;
	}
}


bool Keithley2450Driver::AskPermissionToTurnOn(int channel) //extra check whether it is safe to tunr on supply;
{
    return true;
}

std::string Keithley2450Driver::GenerateCommand(COMMAND_TYPE cmdt, float val)
{
    if (cmdt == COMMAND_TYPE::SetCurrent) {
        return "smu.source.ilimit.level="+std::to_string(val)+"\n";
    } else if (cmdt == COMMAND_TYPE::ReadCurrent){
        return "smu.measure.read(defbuffer1);printbuffer(defbuffer1.n, defbuffer1.n, defbuffer1)\n";
    } else if (cmdt == COMMAND_TYPE::ReadState) {
        return "print(smu.source.output)\n";
    } else if (cmdt == COMMAND_TYPE::ReadVoltage){
        return "print(smu.source.level)\n";
    } else if (cmdt == COMMAND_TYPE::ReadSetVoltage){
        return "print(smu.source.level)\n";
    } else if (cmdt == COMMAND_TYPE::ReadCurrentLimit){
        return "print(smu.source.ilimit.level)\n";
    } else if (cmdt == COMMAND_TYPE::SetVoltage){
        return "smu.source.level="+std::to_string(val)+"\n";
    } else if (cmdt == COMMAND_TYPE::Beep){
        return "beeper.beep(0.5, 4400);\n";
    } else if (cmdt == COMMAND_TYPE::CLearStatus){
        return "*CLS\n";
    } else if (cmdt == COMMAND_TYPE::OPC){
        return "*OPC?\n";
    } else if (cmdt == COMMAND_TYPE::ReadESR){
        return "*ESR?\n";
    } else if (cmdt == COMMAND_TYPE::Reset){
        return "*RST\n";
    } else if (cmdt == COMMAND_TYPE::SetState){
        int ch = (int)val;
        if (ch == 1) {
            return "smu.source.output=smu.ON\n";
        }
        else if (ch == 0) {
            return "smu.source.output=smu.OFF\n";
        }
        else {
            std::cout << "Error: set state can be onlz 1 or 0\n";//TODO: message in midas
            return "\n";
        }
    } else if (cmdt == COMMAND_TYPE::ReadErrorQueue){
        return "print(errorqueue.next())\n";
    } else if (cmdt == COMMAND_TYPE::ReadOVPLevel){
        return "print(smu.source.protect.level)\n";
    } else if (cmdt == COMMAND_TYPE::SetOVPLevel) {
        return "smu.source.protect.level=smu.PROTECT_"+std::to_string((int)(val))+"V\n";
    }
    return "";
}
