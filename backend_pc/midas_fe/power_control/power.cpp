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
#include <thread>
#include <chrono>
#include <future>
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

/* Overwrite equipment struct in ODB from values in code*/
BOOL equipment_common_overwrite = FALSE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 1000;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

std::vector<std::shared_ptr<PowerDriver>> drivers;

/*-- Function declarations -----------------------------------------*/

INT frontend_init();
INT frontend_exit();
INT begin_of_run(INT run_number, char *error);
INT end_of_run(INT run_number, char *error);
INT pause_run(INT run_number, char *error);
INT resume_run(INT run_number, char *error);
INT frontend_loop();
INT read_genesys_power(char *pevent, INT off);
INT read_hameg_power(char *pevent, INT off, std::string eq_name, std::string lvh_num);
INT read_hameg_power0(char *pevent, INT off);
INT read_hameg_power1(char *pevent, INT off);
INT read_hameg_power2(char *pevent, INT off);
INT read_hameg_power3(char *pevent, INT off);
INT read_hameg_power4(char *pevent, INT off);
INT read_hameg_power5(char *pevent, INT off);
INT read_hameg_power6(char *pevent, INT off);
INT read_hameg_power7(char *pevent, INT off);
INT read_hameg_power8(char *pevent, INT off);
INT read_power(float* pdata, const std::string& eqn);




/*-- Equipment list ------------------------------------------------*/

EQUIPMENT equipment[] = {

   /*{"Genesys",                       /* equipment name */
    /*{121, 0,                       /* event ID, trigger mask */
     /*"SYSTEM",                  /* event buffer */
     /*EQ_PERIODIC,                   /* equipment type */
     /*0,                         /* event source */
     /*"MIDAS",                   /* format */
     /*TRUE,                      /* enabled */
     /*RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     /*10000,                     /* read every 10 sec */
     /*0,                         /* stop run after this event limit */
     /*0,                         /* number of sub events */
     /*1,                         /* log history every event */
     /*"", "", ""} ,                  /* device driver list */
     /*read_genesys_power,
    },*/
    
    {"HAMEG0",                       /* equipment name */
    	{120, 0,                       /* event ID, trigger mask */
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
     	read_hameg_power0,    
    },
	
	{"HAMEG1",                       /* equipment name */
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
     	0,                         /* log history every event */
     	"", "", ""} ,                  /* device driver list */
     	read_hameg_power1,    
    },
    
    //{"HAMEG2",                       /* equipment name */
    	//{122, 0,                       /* event ID, trigger mask */
     	//"SYSTEM",                  /* event buffer */
     	//EQ_PERIODIC,                   /* equipment type */
     	//0,                         /* event source */
     	//"MIDAS",                   /* format */
     	//TRUE,                      /* enabled */
     	//RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	//10000,                     /* read every 10 sec */
     	//0,                         /* stop run after this event limit */
    	//0,                         /* number of sub events */
     	//0,                         /* log history every event */
     	//"", "", ""} ,                  /* device driver list */
     	//read_hameg_power2,    
    //},

	//{"HAMEG3",                       /* equipment name */
    	//{123, 0,                       /* event ID, trigger mask */
     	//"SYSTEM",                  /* event buffer */
     	//EQ_PERIODIC,                   /* equipment type */
     	//0,                         /* event source */
     	//"MIDAS",                   /* format */
     	//TRUE,                      /* enabled */
     	//RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	//10000,                     /* read every 10 sec */
     	//0,                         /* stop run after this event limit */
    	//0,                         /* number of sub events */
     	//0,                         /* log history every event */
     	//"", "", ""} ,                  /* device driver list */
     	//read_hameg_power3,    
    //},

	//{"HAMEG4",                       /* equipment name */
    	//{124, 0,                       /* event ID, trigger mask */
     	//"SYSTEM",                  /* event buffer */
     	//EQ_PERIODIC,                   /* equipment type */
     	//0,                         /* event source */
     	//"MIDAS",                   /* format */
     	//TRUE,                      /* enabled */
     	//RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	//10000,                     /* read every 10 sec */
     	//0,                         /* stop run after this event limit */
    	//0,                         /* number of sub events */
     	//0,                         /* log history every event */
     	//"", "", ""} ,                  /* device driver list */
     	//read_hameg_power4,    
    //},

	//{"HAMEG5",                       /* equipment name */
    	//{125, 0,                       /* event ID, trigger mask */
     	//"SYSTEM",                  /* event buffer */
     	//EQ_PERIODIC,                   /* equipment type */
     	//0,                         /* event source */
     	//"MIDAS",                   /* format */
     	//TRUE,                      /* enabled */
     	//RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	//10000,                     /* read every 10 sec */
     	//0,                         /* stop run after this event limit */
    	//0,                         /* number of sub events */
     	//0,                         /* log history every event */
     	//"", "", ""} ,                  /* device driver list */
     	//read_hameg_power5,    
    //},

	//{"HAMEG6",                       /* equipment name */
    	//{126, 0,                       /* event ID, trigger mask */
     	//"SYSTEM",                  /* event buffer */
     	//EQ_PERIODIC,                   /* equipment type */
     	//0,                         /* event source */
     	//"MIDAS",                   /* format */
     	//TRUE,                      /* enabled */
     	//RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	//10000,                     /* read every 10 sec */
     	//0,                         /* stop run after this event limit */
    	//0,                         /* number of sub events */
     	//0,                         /* log history every event */
     	//"", "", ""} ,                  /* device driver list */
     	//read_hameg_power6,    
    //},

	//{"HAMEG7",                       /* equipment name */
    	//{127, 0,                       /* event ID, trigger mask */
     	//"SYSTEM",                  /* event buffer */
     	//EQ_PERIODIC,                   /* equipment type */
     	//0,                         /* event source */
     	//"MIDAS",                   /* format */
     	//TRUE,                      /* enabled */
     	//RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	//10000,                     /* read every 10 sec */
     	//0,                         /* stop run after this event limit */
    	//0,                         /* number of sub events */
     	//0,                         /* log history every event */
     	//"", "", ""} ,                  /* device driver list */
     	//read_hameg_power7,    
    //},

	//{"HAMEG8",                       /* equipment name */
    	//{138, 0,                       /* event ID, trigger mask */
     	//"SYSTEM",                  /* event buffer */
     	//EQ_PERIODIC,                   /* equipment type */
     	//0,                         /* event source */
     	//"MIDAS",                   /* format */
     	//TRUE,                      /* enabled */
     	//RO_STOPPED | RO_RUNNING | RO_PAUSE,        /* all, but not write to odb */
     	//10000,                     /* read every 10 sec */
     	//0,                         /* stop run after this event limit */
    	//0,                         /* number of sub events */
     	//0,                         /* log history every event */
     	//"", "", ""} ,                  /* device driver list */
     	//read_hameg_power8,    
    //},
    
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
			drivers.emplace_back(std::make_shared<GenesysDriver>(equipment[eqID].name,&equipment[eqID].info));
  		}
  		else if( std::find( hameg_names.begin(), hameg_names.end(), shortname ) != hameg_names.end() )
  		{
			drivers.emplace_back(std::make_shared<HMP4040Driver>(equipment[eqID].name,&equipment[eqID].info));
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


INT read_power(float* pdata,const std::string& eq_name)
{
	INT error;
	for(const auto& d: drivers)
	{
		if( !d->Initialized() ) continue;
		//if(!(typeid(*d)==typeid(HMP4040Driver))) continue;
		if(d->GetName()!=eq_name) continue;
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


INT read_hameg_power(char *pevent, INT off, std::string eq_name, std::string lvh_num)
{
  INT error;

  /* init bank structure */
  bk_init32a(pevent);
  float *pdata;
  std::string LVH_str="LVH";
  LVH_str.append(lvh_num);

  bk_create(pevent, LVH_str.c_str(), TID_FLOAT, (void **)&pdata);
  error = read_power(pdata,eq_name);
  //printf("creating bank %s\n",LVH_str.c_str());
  bk_close(pevent, pdata);
  return bk_size(pevent);
}

INT read_hameg_power0(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG0", "0");
}

INT read_hameg_power1(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG1", "1");
}

INT read_hameg_power2(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG2", "2");
}

INT read_hameg_power3(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG3", "3");
}

INT read_hameg_power4(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG4", "4");
}

INT read_hameg_power5(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG5", "5");
}

INT read_hameg_power6(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG6", "6");
}

INT read_hameg_power7(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG7", "7");
}

INT read_hameg_power8(char *pevent, INT off)
{
  return read_hameg_power(pevent, off, "HAMEG8", "8");
}



//INT read_hameg_power0(char *pevent, INT off)
//{
	//std::cout << " read hameg power 0 called" << std::endl;
	//INT error;
	
	///* init bank structure */
  
	//bk_init32a(pevent);
	//float *pdata;
	
	//bk_create(pevent,"LVH2", TID_FLOAT, (void **)&pdata);
	//std::string eq_name = "HAMEG2"; //not super elegant, but the 'read' function does not now which equipment it comes from
	//error = read_power(pdata,eq_name);
	
	//bk_close(pevent, pdata);
  	//return bk_size(pevent);
//}


/*INT read_genesys_power(char *pevent, INT off)
{
	std::cout << " read genesys power called" << std::endl;
	INT error;
	
	/* init bank structure */
  
	/*bk_init32a(pevent);
	float *pdata;
	
	bk_create(pevent,"LVG1", TID_FLOAT, (void **)&pdata);
	std::string eq_name = "Genesys";
	error = read_power(pdata,eq_name);	
	bk_close(pevent, pdata);
  	return bk_size(pevent);
}*/



INT frontend_loop()
{
	std::this_thread::sleep_for (std::chrono::milliseconds(100));
	std::cout << " fe loop called " << std::endl;
	
	std::vector<std::future<int>> futures;
	for(auto& d: drivers)
    {
		if( !d->Initialized() ) continue;
        futures.emplace_back( std::async(&PowerDriver::ReadAll,d) );
    }
    
    for(auto& f : futures)
    {
        INT error = f.get();
        if(error != FE_SUCCESS) cm_msg(MERROR, "frontend_loop power", "Read failed");
    }	
	
	return SUCCESS;
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
