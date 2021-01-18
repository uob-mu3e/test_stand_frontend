/********************************************************************\

  Name:         power.cpp
  Created by:   Frederik Wauters

  Contents:     frontend to control to main Genesys power supplies 
                lab supplies are also be added
                One daisy chain of Genesys supplies is one "equipment" as it has a single IP address and "channels" to select
                A single HAMEG is one equipment to follow the same structure
                This Midas frontend instantiates custum C++ drivers which also take care of the ODB 
                The type of driver is derived from the equipment name 

  $Id$

\********************************************************************/

#include <stdio.h>
#include <iostream>
#include "midas.h"
#include "mfe.h"
#include "GenesysDriver.h"
#include "HMP4040Driver.h"



/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "Power Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 1000;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

std::vector<std::unique_ptr<PowerDriver>> drivers;

/*-- Function declarations -----------------------------------------*/

INT frontend_init();
INT frontend_exit();
INT begin_of_run(INT run_number, char *error);
INT end_of_run(INT run_number, char *error);
INT pause_run(INT run_number, char *error);
INT resume_run(INT run_number, char *error);
INT frontend_loop();
INT read_genesys_power(char *pevent, INT off);
INT read_hameg_power(char *pevent, INT off);
INT read_power(float* pdata);




/*-- Equipment list ------------------------------------------------*/

EQUIPMENT equipment[] = {

   {"Genesys",                       /* equipment name */
    {121, 0,                       /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                   /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     10000,                     /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""} ,                  /* device driver list */
     read_genesys_power,
    },
    
    {"HAMEG1",                       /* equipment name */
    	{122, 0,                       /* event ID, trigger mask */
     	"SYSTEM",                  /* event buffer */
     	EQ_PERIODIC,                   /* equipment type */
     	0,                         /* event source */
     	"MIDAS",                   /* format */
     	TRUE,                      /* enabled */
     	RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	10000,                     /* read every 10 sec */
     	0,                         /* stop run after this event limit */
    	0,                         /* number of sub events */
     	0,                         /* log history every event */
     	"", "", ""} ,                  /* device driver list */
     	read_hameg_power,
    },
    
    {""} //why is there actually this empty one here? FW
    
};



/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{  
	// Get N equipments
	int nEq = sizeof(equipment)/sizeof(equipment[0]);
	if(nEq<2) {cm_msg(MINFO,"power_fe","No Equipment defined"); return FE_ERR_DISABLED; }
	for(unsigned int i = 0; i<nEq-1; i++) cm_msg(MINFO,"power_fe","Init 'Equipment' nr %d name = %s, event ID = %d",i,equipment[i].name,equipment[i].info.event_id);
  
	//  allow equipment name starts to recognize supply type 
	std::vector<std::string> genysis_names = {"Gen","gen","tdk","TDK"};
	std::vector<std::string> hameg_names = {"HMP","hmp","ham","HAM","Lab","lab"};
  
	for(unsigned int eqID = 0; eqID<nEq-1; eqID++)
	{   
		std::cout << "start init Equipment id " << eqID << std::endl;
  	
  		std::string name(equipment[eqID].name); 
  		std::string shortname = name.substr(0, 3);
  	
  		//identify type and instatiate driver
  		if( std::find( genysis_names.begin(), genysis_names.end(), shortname ) != genysis_names.end() )
  		{
			drivers.emplace_back(std::make_unique<GenesysDriver>(equipment[eqID].name,&equipment[eqID].info));
  		}
  		else if( std::find( hameg_names.begin(), hameg_names.end(), shortname ) != hameg_names.end() )
  		{
			drivers.emplace_back(std::make_unique<HMP4040Driver>(equipment[eqID].name,&equipment[eqID].info));
		}
  		else
  		{
  			cm_msg(MINFO,"power_fe","Init 'Equipment' nr %d name = %s not recognizd",eqID,equipment[eqID].name);
  			continue;
  		}
  	
  		//initialize 
		set_equipment_status(equipment[eqID].name, "Initializing...", "yellowLight");  		
  	
  		equipment[eqID].status = drivers.at(eqID)->ConnectODB();
  		if(equipment[eqID].status == FE_ERR_ODB) 	
  		{
			set_equipment_status(equipment[eqID].name, "ODB Error", "redLight");
			cm_msg(MERROR, "initialize_equipment", "Equipment %s disabled because of %s", equipment[eqID].name, "ODB ERROR");
			continue;
		}
		if(!drivers.at(eqID)->Enabled()) //cross check before doing something
  		{
			set_equipment_status(equipment[eqID].name, "Disabled", "redLight");
			continue;
		}
		
		equipment[eqID].status = drivers.at(eqID)->Connect();
		if(equipment[eqID].status != FE_SUCCESS) 	
  		{
			set_equipment_status(equipment[eqID].name, "Connection Error", "redLight");
			cm_msg(MERROR, "initialize_equipment", "Equipment %s disabled because of %s", equipment[eqID].name, "CONNECTION ERROR");
			continue;
		}
	
		equipment[eqID].status = drivers.at(eqID)->Init();
		if(equipment[eqID].status != FE_SUCCESS)
		{
			set_equipment_status(equipment[eqID].name, "DRIVER Error", "redLight");
			cm_msg(MERROR, "initialize_equipment", "Equipment %s disabled because of %s", equipment[eqID].name, "DRIVER ERROR");
			continue;
		}
		else
		{
			drivers.at(eqID)->SetInitialized();
			drivers.at(eqID)->Print();
			std::cout << " read setting " << equipment[eqID].info.read_on <<std::endl;
			set_equipment_status(equipment[eqID].name, "Ok", "greenLight");
		}
		
  	}
  
	ss_sleep(5000);
  
	//Equipment ready

  
	return CM_SUCCESS;   
}





/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{  
  //gendriver->Print();
  ss_sleep(1000);
	return CM_SUCCESS;
}

INT read_hameg_power(char *pevent, INT off)
{
	std::cout << " read hameg power called" << std::endl;
	INT error;
	
	/* init bank structure */
  
	bk_init32a(pevent);
	float *pdata;
	
	bk_create(pevent,"LVH1", TID_FLOAT, (void **)&pdata);
  
	error = read_power(pdata);
	
	bk_close(pevent, pdata);
  	return bk_size(pevent);
}

INT read_genesys_power(char *pevent, INT off)
{
	std::cout << " read genesys power called" << std::endl;
	INT error;
	
	/* init bank structure */
  
	bk_init32a(pevent);
	float *pdata;
	
	bk_create(pevent,"LVG1", TID_FLOAT, (void **)&pdata);
  
	error = read_power(pdata);	
	bk_close(pevent, pdata);
  	return bk_size(pevent);
}


INT read_power(float* pdata)
{
	INT error;
	for(const auto& d: drivers)
	{
		if( !d->Initialized() ) continue;
		if(!(typeid(*d)==typeid(HMP4040Driver))) continue;
		error = d->ReadAll();
		if(error == FE_SUCCESS)
		{
			std::vector<float> voltage = d->GetVoltage();
			std::vector<float> current = d->GetCurrent();
			if(voltage.size() != current.size()) { continue; cm_msg(MERROR, "read_power", "Number of channel reads not consistent"); }
			for(unsigned int iChannel =0; iChannel < voltage.size(); iChannel++)
			{
				*pdata++ = voltage.at(iChannel);
				*pdata++ = current.at(iChannel);
			}	//std::cout << " close bank " << bk_name << std::endl; 
		}
 		else 
 		{
			cm_msg(MERROR, "power read", "Error in read: %d",error);
			return 0;
  		}
	}
	return error;
}

INT frontend_loop()
{
	//std::cout << " frontend_loop() called in power.cpp " << std::endl;
	ss_sleep(100);
	return CM_SUCCESS;
}





/*
            if (equipment[idx].status == FE_SUCCESS)
               strcpy(str, "Ok");
            else if (equipment[idx].status == FE_ERR_HW)
               strcpy(str, "Hardware error");
            else if (equipment[idx].status == FE_ERR_ODB)
               strcpy(str, "ODB error");
            else if (equipment[idx].status == FE_ERR_DRIVER)
               strcpy(str, "Driver error");
            else if (equipment[idx].status == FE_PARTIALLY_DISABLED)
               strcpy(str, "Partially disabled");
            else
               strcpy(str, "Error");
*/



//*************** Not used ******************//

/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
{
   return 1;
}

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   return 1;
}

/*-- Frontend Loop -------------------------------------------------*/



/*-- Begin of Run --------------------------------------------------*/
INT begin_of_run(INT run_number, char *error)
{
   return CM_SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/
INT end_of_run(INT run_number, char *error)
{
   return CM_SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/
INT pause_run(INT run_number, char *error)
{
   return CM_SUCCESS;
}

/*-- Resume Run ----------------------------------------------------*/
INT resume_run(INT run_number, char *error)
{
   return CM_SUCCESS;
}

/*------------------------------------------------------------------*/
