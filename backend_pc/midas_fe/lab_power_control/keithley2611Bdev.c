/********************************************************************\

  Name:         keithley2661Bdev.c
  Created by:   Frederik Wauters

  Contents:     Derived from Stefans Device Driver for Urs Rohrer's beamline control at PSI
                (http://people.web.psi.ch/rohrer_u/secblctl.htm)
                
                ramping approach:
                * don`t use ramping of hv class driver
                * when 'Set' is called, the current limit is raised if not done yet
                * I added CMD_IDLE to the class driver
                  -> if demand v != read v, I assume the device is ramping
                  -> when demand == read, set low current limit if possible
                * So you have several states: 
                  1) ramping (orange status)  -> demand != set
                  2) cool down after ramping (? status) -> demand == set, current limit set high
                  3) low V set (green status) -> demand == set, current limit set == low current limit
                  4) low current limit hit (red status) ->
                 * avoid to get stuck in a status, they are hardware states, leave the method, idle has to figure it out what is going on.
                 

  $Id$

\********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "../../../modules/midas/include/midas.h"
#include "../../../modules/midas/include/mfe.h"
#include <math.h>
#include "../../../modules/midas/include/msystem.h"
#include  <signal.h>
//#include "scpi.h"

/*---- globals -----------------------------------------------------*/

typedef struct {
   char dev_ip[32];
   int port;
   float max_voltage;
   float max_current;
   BOOL reset_on_init;
} KEITHLEY2661B_SETTINGS;



#define KEITHLEY2661B_SETTINGS_STR "\
Supply IP = STRING : [32] 10.32.113.235\n\
Port = INT : 5025\n\
Max_Voltage = FLOAT : -86 \n\
Max_Current = FLOAT : 0.001 \n\
Reset = BOOL : 0\n\
"
typedef struct {
   KEITHLEY2661B_SETTINGS keithley2661B_settings;
   INT num_channels;
   char *name;
   float *demand;
   float *measured;
   float current_limit;
   INT sock;
} KEITHLEY2661B_INFO;


static INT* global_socket;

#define MAX_ERROR        5      // send error after this amount of failed read attempts

/*---- network TCP connection --------------------------------------*/

static void INThandler(int);

static INT tcp_connect(char *host, int port, int *sock)
{
   struct sockaddr_in bind_addr;
   struct hostent *phe;
   int status;
   char str[256];

#ifdef OS_WINNT
   {
      WSADATA WSAData;

      /* Start windows sockets */
      if (WSAStartup(MAKEWORD(1, 1), &WSAData) != 0)
         return RPC_NET_ERROR;
   }
#endif

   /* create a new socket for connecting to remote server */
   *sock = socket(AF_INET, SOCK_STREAM, 0);
   if (*sock == -1) {
      mfe_error("cannot create socket");
      return FE_ERR_HW;
   }

   /* let OS choose any port number */
   memset(&bind_addr, 0, sizeof(bind_addr));
   bind_addr.sin_family = AF_INET;
   bind_addr.sin_addr.s_addr = 0;
   bind_addr.sin_port = 0;

   status = bind(*sock, (const sockaddr*) &bind_addr, sizeof(bind_addr));
   if (status < 0) {
      mfe_error("cannot bind");
      return RPC_NET_ERROR;
   }

   /* connect to remote node */
   memset(&bind_addr, 0, sizeof(bind_addr));
   bind_addr.sin_family = AF_INET;
   bind_addr.sin_addr.s_addr = 0;
   bind_addr.sin_port = htons((short) port);

#ifdef OS_VXWORKS
   {
      INT host_addr;

      host_addr = hostGetByName(host);
      memcpy((char *) &(bind_addr.sin_addr), &host_addr, 4);
   }
#else
   phe = gethostbyname(host);
   if (phe == NULL) {
      mfe_error("cannot get host name");
      return RPC_NET_ERROR;
   }
   memcpy((char *) &(bind_addr.sin_addr), phe->h_addr, phe->h_length);
#endif

#ifdef OS_UNIX
   do {
      status = connect(*sock, (const sockaddr*) &bind_addr, sizeof(bind_addr));

      /* don't return if an alarm signal was cought */
   } while (status == -1 && errno == EINTR);
#else
   status = connect(*sock, (void *) &bind_addr, sizeof(bind_addr));
#endif

   if (status != 0) {
      closesocket(*sock);
      *sock = -1;
      sprintf(str, "cannot connec to host %s", host);
      mfe_error(str);
      return FE_ERR_HW;
   }

   return FE_SUCCESS;
}

void  INThandler(int sig)
{
     signal(sig, SIG_IGN);
     printf("hit Ctrl-C, closing socket first\n");
     closesocket(*global_socket);
     exit(0);
}

/*---- device driver routines --------------------------------------*/

INT keithley2661B_complete(KEITHLEY2661B_INFO * info)
{
  int status = 0;
  char str[1024];
  send(info->sock, "*OPC?\n",strlen("*OPC?\n"), 0);  
  while (status != 1) 
  {
    status = recv_string(info->sock, str, sizeof(str), 10000);
  }
  return FE_SUCCESS;
  
}
/*----------------------------------------------------------------------------*/


INT keithley2661B_init(HNDLE hKey, void **pinfo, INT channels)
{
   int status, size;
   HNDLE hDB;
   char str[1024];
   KEITHLEY2661B_INFO *info;



   /* allocate info structure */
   info = (KEITHLEY2661B_INFO*) calloc(1, sizeof(KEITHLEY2661B_INFO));
   *pinfo = info;

   cm_get_experiment_database(&hDB, NULL);

   /* create settings record */
   status = db_create_record(hDB, hKey, "", KEITHLEY2661B_SETTINGS_STR);
   if (status != DB_SUCCESS)  return FE_ERR_ODB;
   size = sizeof(info->keithley2661B_settings);
   db_get_record(hDB, hKey, &info->keithley2661B_settings, &size, 0);

   
   
   /* initialize driver */
   info->num_channels = channels;
   info->name = (char*) calloc(channels, NAME_LENGTH);
   info->demand = (float*) calloc(channels, sizeof(float));
   info->measured = (float*) calloc(channels, sizeof(float));
   


   /* check power supply name name */
   if (strcmp(info->keithley2661B_settings.dev_ip, "PCxxx") == 0) {
      db_get_path(hDB, hKey, str, sizeof(str));
      cm_msg(MERROR, "keithley2661B_init", "Please set \"%s/hv supply IP\"", str);
      return FE_ERR_ODB;
   }
   
   
   /* contact power supply */
   printf("connecting to hv supply with ip %s , port:  %d ,...\n", info->keithley2661B_settings.dev_ip, info->keithley2661B_settings.port );
   status = tcp_connect(info->keithley2661B_settings.dev_ip, info->keithley2661B_settings.port, &info->sock);
   if (status != FE_SUCCESS)
   {
      printf("error while connecting to device\n");
      return FE_ERR_HW;
   }
   else
   {
     printf("tcp_connect successful\n");
     send(info->sock, "*IDN?\n",strlen("*IDN?\n"), 0);
     status = recv_string(info->sock, str, sizeof(str), 10000);
     if (status <= 0) {
      cm_msg(MERROR, "keithley2661B_init", "cannot retrieve data from %s", info->keithley2661B_settings.dev_ip);
      printf("status < 0 \n");
      return FE_ERR_HW;
     }
     printf("connected to %s\n",str); 
     signal(SIGINT, INThandler); //int handler to close socket when closing the program
     global_socket=&(info->sock);
   }
   
   
   /*reset device*/
   if(info->keithley2661B_settings.reset_on_init == 1)
   {
      send(info->sock, "*RST\n",strlen("*RST\n"), 0);
      printf("resetting device\n");
      keithley2661B_complete(info);
      status = recv_string(info->sock, str, sizeof(str), 10000);
   }
   else printf("No reset option selected\n");
   
  /*clear status*/
  info->current_limit = info->keithley2661B_settings.max_current;
  /* read all channels */


   //last_update = ss_time();
   printf("Device driver finished init\n");

   ss_sleep(50); 
   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_exit(KEITHLEY2661B_INFO * info)
{
   closesocket(info->sock);
   if (info->name)
      free(info->name);
   if (info->demand)
      free(info->demand);
   if (info->measured)
      free(info->measured);
   free(info);
   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_set_current(KEITHLEY2661B_INFO * info,INT channel,float value)
{
  if(fabs(value) > fabs(info->keithley2661B_settings.max_current) )
  {
    cm_msg(MERROR, "keithley2661B_set", "requested current %f not allowed\n", value);
    return FE_ERR_DRIVER;
  }
  char str[128];
  sprintf(str,"""smua.source.limiti=%f \n""",value);
  info->current_limit=value;
  printf("command = %s\n",str);
  send(info->sock, str,strlen(str), 0); 
  keithley2661B_complete(info);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_set(KEITHLEY2661B_INFO * info, INT channel, float value)
{
  //increase current limit unless already ramping
  if(value > 0.01 || fabs(value) > fabs(info->keithley2661B_settings.max_voltage) )
  {
    cm_msg(MERROR, "keithley2661B_set", "requested voltage %f not allowed\n", value);
    return FE_ERR_DRIVER;
  }
  char str[128];
  sprintf(str,"""smua.source.levelv=%f \n""",value);
  printf("command = %s\n",str);
  send(info->sock, str,strlen(str), 0); 
  keithley2661B_complete(info);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_get_voltage(KEITHLEY2661B_INFO * info, INT channel, float *pvalue)
{
  char str[128];
  strcpy(str,"""print(smua.measure.v())\n""");
  send(info->sock,str,strlen(str), 0);
  char str_reply[1024];
  int status = recv_string(info->sock, str_reply, sizeof(str_reply), 10000);
  if (status <= 0)
  {
    cm_msg(MERROR, "keithley2661B_get_voltage", "cannot voltage retrieve data from %s", info->keithley2661B_settings.dev_ip);
    return FE_ERR_HW;
  }
  *pvalue=atof(str_reply);
  return FE_SUCCESS;
}
/*----------------------------------------------------------------------------*/

INT keithley2661B_set_state(KEITHLEY2661B_INFO * info, INT channel,DWORD value)
{
  char str[128];
  if(value == 1 )  {    strcpy(str,"""smua.source.output=smua.OUTPUT_ON\n""");  }
  if(value == 0 )  
  {
    float voltage;
    int status = keithley2661B_get_voltage(info, channel, &voltage);
    if( fabs(voltage) > 1 )
    {
      cm_msg(MERROR, "keithley2661B_set_state", "voltage is %f, can`t switch off\n", voltage);
      return FE_SUCCESS;
    }  
    strcpy(str,"""smua.source.output=smua.OUTPUT_OFF\n""");  
  }  
  send(info->sock,str,strlen(str), 0);
  printf("command = %s from value = %d\n",str,value);
  send(info->sock, str,strlen(str), 0); 
  keithley2661B_complete(info);  
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_rall(KEITHLEY2661B_INFO * info)
{
  return FE_SUCCESS;   
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_get(KEITHLEY2661B_INFO * info, INT channel, float *pvalue)
{
 return FE_SUCCESS;
}



/*----------------------------------------------------------------------------*/

INT keithley2661B_get_current(KEITHLEY2661B_INFO * info, INT channel, float *pvalue)
{
  char str[128];
  strcpy(str,"""print(smua.measure.i())\n""");
  send(info->sock,str,strlen(str), 0);
  char str_reply[1024];
  int status = recv_string(info->sock, str_reply, sizeof(str_reply), 10000);
  if (status <= 0)
  {
    cm_msg(MERROR, "keithley2661B_get_current", "cannot current retrieve data from %s", info->keithley2661B_settings.dev_ip);
    return FE_ERR_HW;
  }
  *pvalue=atof(str_reply);
  return FE_SUCCESS;
}

INT keithley2661B_get_current_limit(KEITHLEY2661B_INFO * info, INT channel, float *pvalue)
{
  *pvalue=info->current_limit;
  printf("Ask for current limit %f\n",info->current_limit);
  return FE_SUCCESS;
}


/*----------------------------------------------------------------------------*/

INT keithley2661B_get_status(KEITHLEY2661B_INFO * info, INT channel, DWORD *pvalue)
{
  char str[128];
  strcpy(str,"""print(smua.source.output)\n""");
  send(info->sock,str,strlen(str), 0);
  char str_reply[1024];
  int status = recv_string(info->sock, str_reply, sizeof(str_reply), 10000);
  if (status <= 0)
  {
    cm_msg(MERROR, "keithley2661B_get_status", "cannot status retrieve data from %s", info->keithley2661B_settings.dev_ip);
    return FE_ERR_HW;
   }
  *pvalue=strtoul(str_reply,NULL,0);
  return FE_SUCCESS;  
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_get_demand(KEITHLEY2661B_INFO * info, INT channel, float *pvalue)
{
 return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_get_default_threshold(KEITHLEY2661B_INFO * info, INT channel, float *pvalue)
{
 return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT keithley2661B_get_label(KEITHLEY2661B_INFO * info, INT channel, char *name)
{
 return FE_SUCCESS;
}

INT keithley2661B_channel_select(KEITHLEY2661B_INFO * info, INT channel)
{
  return FE_SUCCESS;
}

/*---- device driver entry point -----------------------------------*/

INT keithley2611Bdev(INT cmd, ...)
{
   va_list argptr;
   HNDLE hKey;
   INT channel, status;
   float value, *pvalue;
   DWORD* pstatus;
   DWORD state;
   void *info;

   va_start(argptr, cmd);
   status = FE_SUCCESS;

   switch (cmd) {
   case CMD_INIT:
      hKey = va_arg(argptr, HNDLE);
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      status = keithley2661B_init(hKey, (void **) info, channel);
      break;

   case CMD_EXIT:
      info = va_arg(argptr, void *);
      status = keithley2661B_exit((KEITHLEY2661B_INFO*)info);
      break;

   case CMD_SET:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      value = (float) va_arg(argptr, double);
      status = keithley2661B_set((KEITHLEY2661B_INFO*)info, channel, value);
      break;

   case CMD_GET:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pvalue = va_arg(argptr, float *);
      status = keithley2661B_get_voltage((KEITHLEY2661B_INFO*)info, channel, pvalue);
     break;
      
   case CMD_GET_CURRENT:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pvalue = va_arg(argptr, float *);
      status = keithley2661B_get_current((KEITHLEY2661B_INFO*)info, channel, pvalue);
     break;
     
   case CMD_GET_CURRENT_LIMIT:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pvalue = va_arg(argptr, float *);
      status = keithley2661B_get_current_limit((KEITHLEY2661B_INFO*)info, channel, pvalue);
     break;
     
   case CMD_SET_CURRENT_LIMIT:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      value = (float)va_arg(argptr, double);
      printf("current limit demand %f A \n",value);
      status = keithley2661B_set_current((KEITHLEY2661B_INFO*)info, channel, value);
     break;
     
    case CMD_IDLE:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pvalue = va_arg(argptr, float* );
      //printf("idle state channel %d  get current limit of % f\n",channel,*pvalue);
      //status = keithley2661B_check_ramping(info, channel, *pvalue);
     break;
     
   case CMD_SET_CHSTATE:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      value = (float)va_arg(argptr, double);
      state = (DWORD)value;
      printf("channel %d, state demand value %0x08x\n\n",channel, state);
      status = keithley2661B_set_state((KEITHLEY2661B_INFO*)info, channel, (int)value);
     break;     

    
   case CMD_GET_STATUS:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pstatus = (DWORD *) va_arg (argptr, DWORD *);
      //printf("status requested from device driver for channel %d channel\n",channel);
      status = keithley2661B_get_status((KEITHLEY2661B_INFO*)info, channel, pstatus);
      break;


    

   default:
      break;
   }

   va_end(argptr);

   return status;
}

/*------------------------------------------------------------------*/
