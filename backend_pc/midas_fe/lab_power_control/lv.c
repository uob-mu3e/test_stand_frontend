/********************************************************************\

  Name:         lv.c
  Created by:   Frederik Wauters

  Contents:     Low Voltage Class Driver, based on the hv and multi midas drivers

  $Id$

\********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../../../modules/midas/include/midas.h"

/*
 * Set values: demand voltage, current limit, set channel status
 * Read values: measured voltage, measured current, channel status
 */
typedef struct {
   /* ODB keys */
   HNDLE hDB, hKeyRoot;
   HNDLE hKeyDemand, hKeyMeasured, hKeyCurrent, hKeyCurrentSet, hKeyChSet, hKeyChStatus;

   /* globals */
   INT format;
   INT last_channel;
   DWORD last_update;
   //things I want
   INT num_channels;

   /* items in /Variables record */
   char *names;
   float *demand;
   float *measured;
   float *current;
   float *currentset;
   BOOL *chSet;
   BOOL *chStatus;

   /* items in /Settings */
   float *update_threshold;

   /* mirror arrays */
   //things I want, from hv driver
   float *demand_mirror;
   float *measured_mirror;
   float *current_mirror;
   float *currentset_mirror;
   BOOL *chSet_mirror;
   BOOL *chStatus_mirror;


   DEVICE_DRIVER **driver;
   INT *channel_offset;

} LV_INFO;

#ifndef abs
#define abs(a) (((a) < 0)   ? -(a) : (a))
#endif

/*----------------------------------------------------------------------------*/

static void free_mem(LV_INFO * lv_info)
{

   free(lv_info->names);
   free(lv_info->update_threshold);
   free(lv_info->channel_offset);
   free(lv_info->driver);


   free(lv_info->demand);
   free(lv_info->measured);
   free(lv_info->current);
   free(lv_info->currentset);
   free(lv_info->chSet);
   free(lv_info->chStatus);
   
   free(lv_info->demand_mirror);
   free(lv_info->measured_mirror);
   free(lv_info->current_mirror);
   free(lv_info->currentset_mirror);
   free(lv_info->chSet_mirror);
   free(lv_info->chStatus_mirror);
   
   free(lv_info);
}

/*----------------------------------------------------------------------------*/
 


// ********* read voltage ********
void lv_read(EQUIPMENT * pequipment, int channel)
{
   //printf("call lv_read\n");
   int i, status;
   DWORD actual_time;
   LV_INFO *lv_info;
   HNDLE hDB;

   lv_info = (LV_INFO *) pequipment->cd_info;
   cm_get_experiment_database(&hDB, NULL);

   //read
   if (channel == -1 || lv_info->driver[channel]->flags & DF_MULTITHREAD)
   {
     for (i = 0; i < lv_info->num_channels; i++) 
     {
       //printf("Call CMD_GET\n");
       status = device_driver(lv_info->driver[i], CMD_GET, i - lv_info->channel_offset[i], &lv_info->measured[i]);
       if (status != FE_SUCCESS) lv_info->measured[i] = (float)ss_nan();
       else lv_info->measured[i] = lv_info->measured[i]; //what the? FW
       //printf("measured the voltage of channel %d: %f V\n",i,(float)lv_info->measured[i]);
     }
   }  
   else 
   {
     status = device_driver(lv_info->driver[channel], CMD_GET, channel - lv_info->channel_offset[channel], &lv_info->measured[channel]);
     if (status != FE_SUCCESS) lv_info->measured[channel] = (float)ss_nan();
     else lv_info->measured[channel] = lv_info->measured[channel];
   }

   //****so what is ss_nan
   //see system.c
   //double ss_nan()  {
   //  double nan;
   //  nan = 0;
   //  nan = 0 / nan;
   //  return nan; }
   
   
   //Conditions not to update
   // there is a value, but it == the mirror value
   // one is ss_nan,and the other not
   for(i = 0; i < lv_info->num_channels; i++)
   
     if(     ( !ss_isnan(lv_info->measured[i]) && !ss_isnan(lv_info->measured_mirror[i]) &&  abs(lv_info->measured[i] - lv_info->measured_mirror[i]) >  lv_info->update_threshold[i] ) 
         ||  ( ss_isnan(lv_info->measured[i]) && !ss_isnan(lv_info->measured_mirror[i]))
	 ||  (!ss_isnan(lv_info->measured[i]) && ss_isnan(lv_info->measured_mirror[i]))      )    break;

   /* update if change is more than update_sensitivity or last update more
      than a minute ago */
   actual_time = ss_millitime();
   
   if(i < lv_info->num_channels || actual_time - lv_info->last_update > 60000) 
   {
      lv_info->last_update = actual_time;
      for (i = 0; i < lv_info->num_channels; i++)    lv_info->measured_mirror[i] = lv_info->measured[i];
      db_set_data(hDB, lv_info->hKeyMeasured, lv_info->measured, lv_info->num_channels * sizeof(float), lv_info->num_channels, TID_FLOAT);
      pequipment->odb_out++;
   }
}


// ********* read current ********
void lv_read_current(EQUIPMENT * pequipment, int channel)
{
   int i, status;
   DWORD actual_time;
   LV_INFO *lv_info;
   HNDLE hDB;
   lv_info = (LV_INFO *) pequipment->cd_info;
   cm_get_experiment_database(&hDB, NULL);

   //read
   if (channel == -1 || lv_info->driver[channel]->flags & DF_MULTITHREAD)
   {
     for (i = 0; i < lv_info->num_channels; i++) 
     {
       status = device_driver(lv_info->driver[i], CMD_GET_CURRENT, i - lv_info->channel_offset[i], &lv_info->current[i]);
       if (status != FE_SUCCESS) lv_info->current[i] = (float)ss_nan();
       else lv_info->current[i] = lv_info->current[i];
       printf("measured the current of channel %d: %f A\n",i,(float)lv_info->current[i]);
     }
   }  
   else 
   {
     status = device_driver(lv_info->driver[channel], CMD_GET_CURRENT, channel - lv_info->channel_offset[channel], &lv_info->current[channel]);
     if (status != FE_SUCCESS) lv_info->current[channel] = (float)ss_nan();
     else lv_info->current[channel] = lv_info->current[channel];
   }

   for(i = 0; i < lv_info->num_channels; i++)
   
     if(     ( !ss_isnan(lv_info->current[i]) && !ss_isnan(lv_info->current_mirror[i]) &&  abs(lv_info->current[i] - lv_info->current_mirror[i]) >  0.001*lv_info->update_threshold[i] ) 
         ||  ( ss_isnan(lv_info->current[i]) && !ss_isnan(lv_info->current_mirror[i]))
	 ||  (!ss_isnan(lv_info->current[i]) && ss_isnan(lv_info->current_mirror[i]))      )    break;

   actual_time = ss_millitime();
   
   if(i < lv_info->num_channels || actual_time - lv_info-> last_update > 60000) 
   {
      lv_info->last_update = actual_time;
      for (i = 0; i < lv_info->num_channels; i++)    lv_info->current_mirror[i] = lv_info->current[i];
      db_set_data(hDB, lv_info->hKeyCurrent, lv_info->current, lv_info->num_channels * sizeof(float), lv_info->num_channels, TID_FLOAT);
      pequipment->odb_out++;
   }
}




// ********* read status ********
void lv_read_status(EQUIPMENT * pequipment, int channel)
{
   int i, status;
   DWORD actual_time;
   LV_INFO *lv_info;
   HNDLE hDB;
   lv_info = (LV_INFO *) pequipment->cd_info;
   cm_get_experiment_database(&hDB, NULL);

   //read
   if (channel == -1 || lv_info->driver[channel]->flags & DF_MULTITHREAD)
   {
     for (i = 0; i < lv_info->num_channels; i++) 
     {
       status = device_driver(lv_info->driver[i], CMD_GET_STATUS, i - lv_info->channel_offset[i], &lv_info->chStatus[i]);
       if (status != FE_SUCCESS) lv_info->chStatus[i] = (float)ss_nan();
       else lv_info->chStatus[i] = lv_info->chStatus[i];
       //printf("measured the current of channel %d: %c\n",i,(float)lv_info->chStatus[i]);
     }
   }  
   else 
   {
     status = device_driver(lv_info->driver[channel], CMD_GET_STATUS, channel - lv_info->channel_offset[channel], &lv_info->chStatus[channel]);
     if (status != FE_SUCCESS) lv_info->chStatus[channel] = (float)ss_nan();
     else lv_info->chStatus[channel] = lv_info->chStatus[channel];
   }

   for(i = 0; i < lv_info->num_channels; i++)
   
     if(     ( !ss_isnan(lv_info->chStatus[i]) && !ss_isnan(lv_info->chStatus_mirror[i]) &&  abs(lv_info->chStatus[i] - lv_info->chStatus_mirror[i]) >  lv_info->update_threshold[i] ) 
         ||  ( ss_isnan(lv_info->chStatus[i]) && !ss_isnan(lv_info->chStatus_mirror[i]))
	 ||  (!ss_isnan(lv_info->chStatus[i]) && ss_isnan(lv_info->chStatus_mirror[i]))      )    break;


   actual_time = ss_millitime();
   
   if(i < lv_info->num_channels || actual_time - lv_info->last_update > 60000) 
   {
      lv_info->last_update = actual_time;
      for (i = 0; i < lv_info->num_channels; i++)    lv_info->chStatus_mirror[i] = lv_info->chStatus[i];
      db_set_data(hDB, lv_info->hKeyChStatus, lv_info->chStatus, lv_info->num_channels * sizeof(float), lv_info->num_channels, TID_BOOL);
      pequipment->odb_out++;
   }
}


/*----------------------------------------------------------------------------*/

void lv_read_output(EQUIPMENT * pequipment, int channel)
{
   /*float value;
   LV_INFO *lv_info;
   HNDLE hDB;

   lv_info = (LV_INFO *) pequipment->cd_info;
   cm_get_experiment_database(&hDB, NULL);

   device_driver(lv_info->driver_output[channel], CMD_GET,
                 channel - lv_info->channel_offset_output[channel], &value);



   if (value != lv_info->output_mirror[channel]) {
      lv_info->output_mirror[channel] = value;
      lv_info->var_input[channel] = value;

      db_set_record(hDB, lv_info->hKeyOutput, lv_info->output_mirror,
                    lv_info->num_channels_output * sizeof(float), 0);

      pequipment->odb_out++;
   }*/

}

/*----------------------------------------------------------------------------*/

void lv_output(INT hDB, INT hKey, void *info)
{
   INT i;
   DWORD act_time;
   LV_INFO *lv_info;
   EQUIPMENT *pequipment;

   pequipment = (EQUIPMENT *) info;
   lv_info = (LV_INFO *) pequipment->cd_info;

   act_time = ss_millitime();

   for (i = 0; i < lv_info->num_channels; i++)
   {
      /// only change voltage of if demand changes by more than by more than 1 mV 
      if(fabs(lv_info->demand[i] - lv_info->demand_mirror[i])>0.001) 
      {
         lv_info->demand_mirror[i] = lv_info->demand[i];
	 printf("changing voltage to %f\n",lv_info->demand_mirror[i]);
         device_driver(lv_info->driver[i], CMD_SET,  i - lv_info->channel_offset[i],  lv_info->demand_mirror[i]);
      }
   }
   pequipment->odb_in++;
}

/*----------------------------------------------------------------------------*/

void lv_set_current(INT hDB, INT hKey, void *info)
{
   INT i;
   DWORD act_time;
   LV_INFO *lv_info;
   EQUIPMENT *pequipment;

   pequipment = (EQUIPMENT *) info;
   lv_info = (LV_INFO *) pequipment->cd_info;

   act_time = ss_millitime();

   for (i = 0; i < lv_info->num_channels; i++)
   {
      if(fabs(lv_info->currentset[i] - lv_info->currentset_mirror[i])>0.001) 
      {
         lv_info->currentset_mirror[i] = lv_info->currentset[i];
	 printf("changing current to %f\n",lv_info->currentset_mirror[i]);
         device_driver(lv_info->driver[i], CMD_SET_CURRENT_LIMIT,  i - lv_info->channel_offset[i],  lv_info->currentset_mirror[i]);
      }
   }
   pequipment->odb_in++;
}

/*----------------------------------------------------------------------------*/

void lv_set_state(INT hDB, INT hKey, void *info)
{
   INT i;
   DWORD act_time;
   LV_INFO *lv_info;
   EQUIPMENT *pequipment;

   pequipment = (EQUIPMENT *) info;
   lv_info = (LV_INFO *) pequipment->cd_info;

   act_time = ss_millitime();

   for (i = 0; i < lv_info->num_channels; i++)
   {
      if( (lv_info->chSet[i] & 1) != (lv_info->chSet_mirror[i] & 1) )  
      {
	 printf("changing state from %d to %d for channel %d\n",lv_info->chSet_mirror[i] & 1,lv_info->chSet[i] & 1 , i );
         lv_info->chSet_mirror[i] = lv_info->chSet[i];
	 float value = 0.0 +  lv_info->chSet_mirror[i] ;
	 device_driver(lv_info->driver[i], CMD_SET_CHSTATE,  i - lv_info->channel_offset[i],  value);
      }
   }
   pequipment->odb_in++;
}

/*------------------------------------------------------------------*/

void lv_update_label(INT hDB, INT hKey, void *info)
{
   INT i, status;
   LV_INFO *lv_info;
   EQUIPMENT *pequipment;

   pequipment = (EQUIPMENT *) info;
   lv_info = (LV_INFO *) pequipment->cd_info;

   /* update channel labels based on the midas channel names */
   for (i = 0; i < lv_info->num_channels; i++)
   {  
     status = device_driver(lv_info->driver[i], CMD_SET_LABEL, i - lv_info->channel_offset[i],  lv_info->names + NAME_LENGTH * i);
   }

}

/*----------------------------------------------------------------------------*/

INT lv_init(EQUIPMENT * pequipment)
{
   printf("enter lv_init ... \n");
   int status, size, i, j, index, ch_offset;
   char str[256];
   HNDLE hDB, hKey, hNames;
   LV_INFO *lv_info;
   BOOL partially_disabled;
   
   /* allocate private data */
   pequipment->cd_info = calloc(1, sizeof(LV_INFO));
   lv_info = (LV_INFO *) pequipment->cd_info;

   /* get class driver root key */
   cm_get_experiment_database(&hDB, NULL);
   sprintf(str, "/Equipment/%s", pequipment->name);
   db_create_key(hDB, 0, str, TID_KEY);
   db_find_key(hDB, 0, str, &lv_info->hKeyRoot);

   /* save event format */
   size = sizeof(str);
   db_get_value(hDB, lv_info->hKeyRoot, "Common/Format", str, &size, TID_STRING, TRUE);
   if (equal_ustring(str, "Fixed"))      lv_info->format = FORMAT_FIXED;
   else if (equal_ustring(str, "MIDAS"))      lv_info->format = FORMAT_MIDAS;
   else if (equal_ustring(str, "YBOS"))      lv_info->format = FORMAT_YBOS;

   
   /* count total number of channels */
   for (i = 0, lv_info->num_channels = 0; pequipment->driver[i].name[0]; i++)
   {
      if(pequipment->driver[i].channels == 0)
      {
        cm_msg(MERROR, "lv_init", "Driver with zero channels not allowed");
        return FE_ERR_ODB;
      }
      lv_info->num_channels += pequipment->driver[i].channels;
   }
   if (lv_info->num_channels == 0)
   {
      cm_msg(MERROR, "lv_init", "No channels found in device driver list");
      return FE_ERR_ODB;
   }
   

   /* Allocate memory for buffers */
   if(lv_info->num_channels)
   {
     lv_info->demand = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->measured = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->current = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->currentset = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->chSet = (BOOL *) calloc(lv_info->num_channels, sizeof(BOOL));
     lv_info->chStatus = (BOOL *) calloc(lv_info->num_channels, sizeof(BOOL));
   
     lv_info->demand_mirror = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->measured_mirror = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->current_mirror = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->currentset_mirror = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->chSet_mirror = (BOOL *) calloc(lv_info->num_channels, sizeof(BOOL));
     lv_info->chStatus_mirror = (BOOL *) calloc(lv_info->num_channels, sizeof(BOOL));
   
     lv_info->driver = (DEVICE_DRIVER **) calloc(lv_info->num_channels, sizeof(void *));
     lv_info->names = (char *) calloc(lv_info->num_channels, NAME_LENGTH);
     lv_info->update_threshold = (float *) calloc(lv_info->num_channels, sizeof(float));
     lv_info->channel_offset = (INT *) calloc(lv_info->num_channels, sizeof(INT));
   }
   
   
   
   
 
   /*---- Create/Read settings ----*/

   if(lv_info->num_channels)
   {
     /* Update threshold */
     for (i = 0; i < lv_info->num_channels; i++) lv_info->update_threshold[i] = 0.1f;       /* default 0.1 */
     db_merge_data(hDB, lv_info->hKeyRoot, "Settings/Update Threshold", lv_info->update_threshold, lv_info->num_channels * sizeof(float), lv_info->num_channels, TID_FLOAT);
     db_find_key(hDB, lv_info->hKeyRoot, "Settings/Update Threshold", &hKey);
     db_open_record(hDB, hKey, lv_info->update_threshold, lv_info->num_channels * sizeof(float), MODE_READ, NULL, NULL);
   }
   
  
   
      
   /*---- Create/Read variables ----*/
   
   /* Demand */
   db_merge_data(hDB, lv_info->hKeyRoot, "Variables/Demand",lv_info->demand, sizeof(float) * lv_info->num_channels,lv_info->num_channels, TID_FLOAT);
   db_find_key(hDB, lv_info->hKeyRoot, "Variables/Demand", &lv_info->hKeyDemand); 
   //find key is called in db_merge_data function, so I removed it here FW
   //this method also calles db_create_key if the key does not exists, and sets the default values set in lv_info->demand in the db
   //if it exists, it resets the entry with the existing value from the bd (https://ladd00.triumf.ca/~olchansk/midas/group__odbfunctionc.html)
   
   /* Measured */
   db_merge_data(hDB, lv_info->hKeyRoot, "Variables/Measured",lv_info->measured, sizeof(float) * lv_info->num_channels,lv_info->num_channels, TID_FLOAT);
   db_find_key(hDB, lv_info->hKeyRoot, "Variables/Measured", &lv_info->hKeyMeasured);
   memcpy(lv_info->measured_mirror, lv_info->measured,lv_info->num_channels * sizeof(float));

   /* Current Limit Set */
   db_merge_data(hDB, lv_info->hKeyRoot, "Variables/CurrentLimit",lv_info->currentset, sizeof(float) * lv_info->num_channels,lv_info->num_channels, TID_FLOAT);
   db_find_key(hDB, lv_info->hKeyRoot, "Variables/CurrentLimit", &lv_info->hKeyCurrentSet);
   
   /* Current */
   db_merge_data(hDB, lv_info->hKeyRoot, "Variables/Current",lv_info->current, sizeof(float) * lv_info->num_channels,lv_info->num_channels, TID_FLOAT);
   db_find_key(hDB, lv_info->hKeyRoot, "Variables/Current", &lv_info->hKeyCurrent);
   memcpy(lv_info->current_mirror, lv_info->current, lv_info->num_channels * sizeof(float));
   
   /* Set Status */
   db_merge_data(hDB, lv_info->hKeyRoot, "Variables/ChSet", lv_info->chSet, sizeof(BOOL) * lv_info->num_channels, lv_info->num_channels, TID_BOOL);
   db_find_key(hDB, lv_info->hKeyRoot, "Variables/ChSet", &lv_info->hKeyChSet);
   
   /* Status */
   db_merge_data(hDB, lv_info->hKeyRoot, "Variables/State", lv_info->chStatus, sizeof(BOOL) * lv_info->num_channels, lv_info->num_channels, TID_BOOL);
   db_find_key(hDB, lv_info->hKeyRoot, "Variables/State", &lv_info->hKeyChStatus);
   memcpy(lv_info->chStatus_mirror, lv_info->chStatus,lv_info->num_channels * sizeof(BOOL));
   
   printf("start init drivers\n");

   /*---- Initialize device drivers ----*/

   /* call init method */
   partially_disabled = FALSE;
   for (i = 0; pequipment->driver[i].name[0]; i++)
   {
     sprintf(str, "Settings/Devices/%s", pequipment->driver[i].name);
     status = db_find_key(hDB, lv_info->hKeyRoot, str, &hKey);
     if (status != DB_SUCCESS)
     {
       db_create_key(hDB, lv_info->hKeyRoot, str, TID_KEY);
       status = db_find_key(hDB, lv_info->hKeyRoot, str, &hKey);
       if (status != DB_SUCCESS)
       {
         cm_msg(MERROR, "lv_init", "Cannot create %s entry in online database",str);
         free_mem(lv_info);
         return FE_ERR_ODB;
        }
      }

      /* check enabled flag */
      size = sizeof(pequipment->driver[i].enabled);
      pequipment->driver[i].enabled = 1;
      sprintf(str, "Settings/Devices/%s/Enabled", pequipment->driver[i].name);
      status = db_get_value(hDB, lv_info->hKeyRoot, str, &pequipment->driver[i].enabled, &size, TID_BOOL, TRUE);
      
      if (status != DB_SUCCESS)  return FE_ERR_ODB;

      if (pequipment->driver[i].enabled)
      {
        status = device_driver(&pequipment->driver[i], CMD_INIT, hKey);
        if (status != FE_SUCCESS)
	{
            free_mem(lv_info);
            return status;
         }
      } 
      else partially_disabled = TRUE;
   }   
   
   

   /* compose device driver channel assignment */
   for (i = 0, j = 0, index = 0, ch_offset = 0; i < lv_info->num_channels; i++, j++)
   {
     while(  pequipment->driver[index].name[0] &&   ( j >= pequipment->driver[index].channels || (pequipment->driver[index].flags & DF_INPUT) == 0 )  ) 
     {
       ch_offset += j;
       index++;
       j = 0;
      }
      lv_info->driver[i] = &pequipment->driver[index];
      lv_info->channel_offset[i] = ch_offset;
   }
   
          
   printf("init ... names ... \n");       
 
   /*---- get default names from device driver ----*/
   if(lv_info->num_channels)
   {
     for (i = 0; i < lv_info->num_channels; i++)
     {
       sprintf(lv_info->names + NAME_LENGTH * i, "Channel %d", i+1);
       device_driver(lv_info->driver[i], CMD_GET_LABEL, i - lv_info->channel_offset[i],   lv_info->names + NAME_LENGTH * i);
       /* merge existing names with labels from driver */
       status = db_find_key(hDB, lv_info->hKeyRoot, "Settings/Names", &hKey);
       if(status != DB_SUCCESS)
       {
         db_create_key(hDB, lv_info->hKeyRoot, "Settings/Names", TID_STRING);
         db_find_key(hDB, lv_info->hKeyRoot, "Settings/Names", &hKey);
         db_set_data(hDB, hKey, lv_info->names, NAME_LENGTH, 1, TID_STRING);
       }
       else
       {
         size = sizeof(str);
         db_get_data_index(hDB, hKey, str, &size, i, TID_STRING);
         if (!str[0])  db_set_data_index(hDB, hKey, lv_info->names+NAME_LENGTH*i, NAME_LENGTH, i, TID_STRING);
       }       
     }
   }   
   
   
   
   /*---- set labels from midas SC names ----*/
   if(lv_info->num_channels)
   {
     for (i = 0; i < lv_info->num_channels; i++)  
     { device_driver(lv_info->driver[i], CMD_SET_LABEL, i - lv_info->channel_offset[i],lv_info->names + NAME_LENGTH * i);      }
      /* open hotlink on channel names */
      if (db_find_key(hDB, lv_info->hKeyRoot, "Settings/Names", &hNames) == DB_SUCCESS) 
      {
	db_open_record(hDB, hNames, lv_info->names,NAME_LENGTH * lv_info->num_channels, MODE_READ, lv_update_label, pequipment);
	//printf("sucesssssssssssssssssssss\n");
      }
   }
   
   
   printf("open hot links \n");
   
   /* open hot link to demand record */
   db_open_record(hDB, lv_info->hKeyDemand, lv_info->demand,  lv_info->num_channels * sizeof(float), MODE_READ, lv_output, pequipment);
   db_open_record(hDB, lv_info->hKeyCurrentSet, lv_info->currentset,  lv_info->num_channels * sizeof(float), MODE_READ, lv_set_current, pequipment);
   db_open_record(hDB, lv_info->hKeyChSet, lv_info->chSet,  lv_info->num_channels * sizeof(BOOL), MODE_READ, lv_set_state, pequipment);
      

   /* set initial demand values */
   for (i = 0; i < lv_info->num_channels; i++)
   {
     /* use default value from ODB */
     lv_info->demand_mirror[i] = lv_info->demand[i];
     device_driver(lv_info->driver[i], CMD_SET, i - lv_info->channel_offset[i],  lv_info->demand_mirror[i]);     
     lv_info->current_mirror[i] = lv_info->current[i];
     device_driver(lv_info->driver[i], CMD_SET_CURRENT_LIMIT, i - lv_info->channel_offset[i],  lv_info->current_mirror[i]);   
     lv_info->chSet_mirror[i] = lv_info->chSet[i];
     device_driver(lv_info->driver[i], CMD_SET_CHSTATE, i - lv_info->channel_offset[i],  (float)lv_info->chSet_mirror[i]);
   }   
      
   /* initially read all input channels */
   //if(lv_info->num_channels)  lv_read(pequipment, -1);

   if (partially_disabled)
      return FE_PARTIALLY_DISABLED;
   
   
   printf("Class driver finished init\n");

   ss_sleep(5000); 
   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT lv_exit(EQUIPMENT * pequipment)
{
   INT i;
   free_mem((LV_INFO *) pequipment->cd_info);

   /* call exit method of device drivers */
   for (i = 0; pequipment->driver[i].dd != NULL; i++)
      device_driver(&pequipment->driver[i], CMD_EXIT);

   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT lv_start(EQUIPMENT * pequipment)
{
   INT i;

   /* call start method of device drivers */
   for (i = 0; pequipment->driver[i].dd != NULL ; i++)
      if (pequipment->driver[i].flags & DF_MULTITHREAD) {
         pequipment->driver[i].pequipment = &pequipment->info;
         device_driver(&pequipment->driver[i], CMD_START);
      }

   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT lv_stop(EQUIPMENT * pequipment)
{
   INT i;

   /* call close method of device drivers */
   for (i = 0; pequipment->driver[i].dd != NULL && pequipment->driver[i].flags & DF_MULTITHREAD ; i++)
      device_driver(&pequipment->driver[i], CMD_STOP);

   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT lv_idle(EQUIPMENT * pequipment)
{
   int i;
   LV_INFO *lv_info;

   lv_info = (LV_INFO *) pequipment->cd_info;

   //printf("In idle function\n");
  if(lv_info->last_channel < lv_info->num_channels)
  {
   //read stuff
   if(lv_info->driver[lv_info->last_channel]->flags & DF_MULTITHREAD)
   {
     lv_read(pequipment, -1);
     lv_read_current(pequipment,-1);
     lv_read_status(pequipment,-1);
     //lv_output(pequipment,-1);
   }
   else
   {
     if (lv_info->last_channel < lv_info->num_channels)
     {
       lv_read(pequipment, lv_info->last_channel);
       lv_read_current(pequipment,lv_info->last_channel);
       lv_read_status(pequipment,lv_info->last_channel);
       lv_info->last_channel++;
     }
   } 
  }
  else lv_info->last_channel=0;
   
   /*else 
   {
     if(!lv_info->num_channels_output) { lv_info->last_channel = 0; }
     else
     {
         // search output channel with DF_PRIO_DEV 
         for (i = lv_info->last_channel - lv_info->num_channels_input;
              i < lv_info->num_channels_output; i++)
            if (lv_info->driver_output[i]->flags & DF_PRIO_DEVICE)
               break;

         if (i < lv_info->num_channels_output) {
            // read output channel 
            lv_read_output(pequipment, i);

            lv_info->last_channel = i + lv_info->num_channels_input;
            if (lv_info->last_channel <
                lv_info->num_channels_input + lv_info->num_channels_output - 1)
               lv_info->last_channel++;
            else
               lv_info->last_channel = 0;
         } else
            lv_info->last_channel = 0;
      }
   }*/


   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT cd_lv_read(char *pevent, int offset)
{
   float *pdata;
   LV_INFO *lv_info;
   EQUIPMENT *pequipment;
#ifdef HAVE_YBOS
   DWORD *pdw;
#endif
   //printf("calling cd_lv_read");
   pequipment = *((EQUIPMENT **) pevent);
   lv_info = (LV_INFO *) pequipment->cd_info;

   if (lv_info->format == FORMAT_FIXED)
   {
//       memcpy(pevent, lv_info->var_input, sizeof(float) * lv_info->num_channels_input);
//       pevent += sizeof(float) * lv_info->num_channels_input;
// 
//       memcpy(pevent, lv_info->var_output, sizeof(float) * lv_info->num_channels_output);
//       pevent += sizeof(float) * lv_info->num_channels_output;
// 
//       return sizeof(float) * (lv_info->num_channels_input + lv_info->num_channels_output);
   }
   else if (lv_info->format == FORMAT_MIDAS) {
      bk_init32a(pevent);

      /* create INPT bank */
//       if (lv_info->num_channels_input) {
//          bk_create(pevent, "INPT", TID_FLOAT, (void **)&pdata);
//          memcpy(pdata, lv_info->var_input, sizeof(float) * lv_info->num_channels_input);
//          pdata += lv_info->num_channels_input;
//          bk_close(pevent, pdata);
//       }
// 
//       /* create OUTP bank */
//       if (lv_info->num_channels_output) {
//          bk_create(pevent, "OUTP", TID_FLOAT, (void **)&pdata);
//          memcpy(pdata, lv_info->var_output, sizeof(float) * lv_info->num_channels_output);
//          pdata += lv_info->num_channels_output;
//          bk_close(pevent, pdata);
//       }
      
      /* create LV_DMND bank */
      bk_create(pevent, "LV_DMND", TID_FLOAT, (void **)&pdata);
      memcpy(pdata, lv_info->demand, sizeof(float) * lv_info->num_channels);
      pdata += lv_info->num_channels;
      bk_close(pevent, pdata);
      
      /* create LV_MSRD bank */
      bk_create(pevent, "LV_MSRD", TID_FLOAT, (void **)&pdata);
      memcpy(pdata, lv_info->measured, sizeof(float) * lv_info->num_channels);
      pdata += lv_info->num_channels;
      bk_close(pevent, pdata);
      
      /* create LV_CRNT bank */
      bk_create(pevent, "LV_CRNT", TID_FLOAT, (void **)&pdata);
      memcpy(pdata, lv_info->current, sizeof(float) * lv_info->num_channels);
      pdata += lv_info->num_channels;
      bk_close(pevent, pdata);
      
      /* create STAT bank */
      bk_create(pevent, "STAT", TID_BOOL, (void **)&pdata);
      memcpy(pdata, lv_info->chStatus, sizeof(BOOL) * lv_info->num_channels); 
      pdata += lv_info->num_channels;
      bk_close(pevent, pdata);

      return bk_size(pevent);
      
   } 
   else if (lv_info->format == FORMAT_YBOS) {
#ifdef HAVE_YBOS
      printf("YBOS not supported!!!\n");
#endif
   }

   return 0;
}

/*----------------------------------------------------------------------------*/

INT cd_lv(INT cmd, EQUIPMENT * pequipment)
{
   INT status;

   switch (cmd) {
   case CMD_INIT:
      status = lv_init(pequipment);
      break;

   case CMD_EXIT:
      status = lv_exit(pequipment);
      break;

   case CMD_START:
      status = lv_start(pequipment);
      break;

   case CMD_STOP:
      status = lv_stop(pequipment);
      break;

   case CMD_IDLE:
      status = lv_idle(pequipment);
      break;

   default:
      cm_msg(MERROR, "LV class driver", "Received unknown command %d", cmd);
      status = FE_ERR_DRIVER;
      break;
   }

   return status;
}
