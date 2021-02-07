/********************************************************************\

  Name:         HMP4040.c
  Created by:   Frederik

  Contents:     Derived from Stefans Device Driver for Urs Rohrer's beamline control at PSI
                (http://people.web.psi.ch/rohrer_u/secblctl.htm)

  $Id$

\********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include  <signal.h>
#include "../../../../modules/midas/include/midas.h"
#include "../../../../modules/midas/include/mfe.h"
#include "../../../../modules/midas/include/msystem.h"
//#include "scpi.h"

/*---- globals -----------------------------------------------------*/

typedef struct {
   char dev_ip[32];
   int port;
   BOOL reset_on_init;
} HMP4040_SETTINGS;



#define HMP4040_SETTINGS_STR "\
Supply IP = STRING : [32] 10.32.113.236\n\
Port = INT : 5025\n\
Reset = BOOL : 0\n\
"
typedef struct {
   HMP4040_SETTINGS hmp4040_settings;
   INT num_channels;
   char *name;
   float *demand;
   float *measured;
   INT sock;
} HMP4040_INFO;


//static INT* global_socket;

#define MAX_ERROR        5      // send error after this amount of failed read attempts

/*---- network TCP connection --------------------------------------*/

//static void INThandler(int);

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

/*void  INThandler(int sig)
{
     signal(sig, SIG_IGN);
     printf("hit Ctrl-C, closing socket first\n");
     closesocket(*global_socket);
     exit(0);
}*/

/*---- device driver routines --------------------------------------*/


INT hmp_4040_complete(HMP4040_INFO * info)
{
  int status = 0;
  char str[1024];
  send(info->sock, "*OPC?\n",strlen("*OPC?\n"), 0);  
  while (status != 1) 
  {
    status = recv_string(info->sock, str, sizeof(str), 10000);
    //cm_msg(MERROR, "hmp4040_complete", "cannot retrieve data from %s", info->hmp4040_settings.dev_ip);
    //return FE_ERR_HW;
  }
  //printf("device ready for new command\n");  
  return FE_SUCCESS;
  
}


/*----------------------------------------------------------------------------*/

INT hmp4040_channel_select(HMP4040_INFO * info, INT channel)
{
  char str[256];
  char* str1 = "INST OUT";
  char* str2 = (char*)calloc(1,sizeof(INT));
  char* endl = "\n";
  sprintf(str2,"%d",channel+1);  
  strcpy(str,str1);
  strcat(str,str2);
  strcat(str,endl);
  //printf("command %s\n",str);
  send(info->sock, str,strlen(str), 0);  
  return FE_SUCCESS;
}



/*----------------------------------------------------------------------------*/

INT hmp4040_init(HNDLE hKey, void **pinfo, INT channels)
{
   int status, size;
   HNDLE hDB;
   char str[1024];
   HMP4040_INFO *info;



   /* allocate info structure */
   info = (HMP4040_INFO*) calloc(1, sizeof(HMP4040_INFO));
   *pinfo = info;

   cm_get_experiment_database(&hDB, NULL);

   /* create hmp4040 settings record */
   status = db_create_record(hDB, hKey, "", HMP4040_SETTINGS_STR);
   if (status != DB_SUCCESS)  return FE_ERR_ODB;
   size = sizeof(info->hmp4040_settings);
   db_get_record(hDB, hKey, &info->hmp4040_settings, &size, 0);

   
   
   /* initialize driver */
   info->num_channels = channels;
   info->name = (char*) calloc(channels, NAME_LENGTH);
   info->demand = (float*) calloc(channels, sizeof(float));
   info->measured = (float*) calloc(channels, sizeof(float));


   /* check power supply name name */
   if (strcmp(info->hmp4040_settings.dev_ip, "PCxxx") == 0) {
      db_get_path(hDB, hKey, str, sizeof(str));
      cm_msg(MERROR, "hmp4040_init", "Please set \"%s/power supply IP\"", str);
      return FE_ERR_ODB;
   }
   
   
   /* contact power supply */
   printf("connecting to power supply with ip %s , port:  %d ,...\n", info->hmp4040_settings.dev_ip, info->hmp4040_settings.port );
   status = tcp_connect(info->hmp4040_settings.dev_ip, info->hmp4040_settings.port, &info->sock);
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
      cm_msg(MERROR, "hmp4040_init", "cannot retrieve data from %s", info->hmp4040_settings.dev_ip);
      return FE_ERR_HW;
     }
     printf("connected to %s\n",str);     
     //signal(SIGINT, INThandler); //int handler to close socket when closing the program
     //global_socket=&(info->sock);
   }
   
   
 
 
   /*reset device*/
   if(info->hmp4040_settings.reset_on_init == 1)
   {
      send(info->sock, "*RST\n",strlen("*RST\n"), 0);
      hmp_4040_complete(info);
      status = recv_string(info->sock, str, sizeof(str), 10000);
   }
   else printf("No reset option selected\n");
   
  /*clear status*/
   send(info->sock, "*CLS\n",strlen("*CLS\n"), 0);
   hmp_4040_complete(info);
   
   
   /* read all channels */


   //last_update = ss_time();
   printf("Device driver finished init\n");

   ss_sleep(50); 
   return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_exit(HMP4040_INFO * info)
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

INT hmp4040_set_current(HMP4040_INFO * info, INT channel, float value)
{
  hmp4040_channel_select(info,channel);
  hmp_4040_complete(info);
  char str[128];
  char* str1 = "CURR";
  char* str2 = " ";
  char* str3 = (char *)calloc(1,sizeof(float));
  sprintf(str3,"%f",value);
  char* str4 = "\n";  
  strcpy(str,str1);
  strcat(str,str2);
  strcat(str,str3);
  strcat(str,str4);
  send(info->sock,str,strlen(str), 0);
  printf("command = %s\n",str);
  send(info->sock, str,strlen(str), 0); 
  hmp_4040_complete(info);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_set(HMP4040_INFO * info, INT channel, float value)
{
  printf("asking to set the voltage of channel %d to %f\n",channel+1,value);
  hmp4040_channel_select(info,channel);
  hmp_4040_complete(info);
  char str[128];
  char* str1 = "VOLT";
  char* str2 = " ";
  char* str3 = (char *)calloc(1,sizeof(float));
  sprintf(str3,"%f",value);
  char* str4 = "\n";  
  strcpy(str,str1);
  strcat(str,str2);
  strcat(str,str3);
  strcat(str,str4);
  send(info->sock,str,strlen(str), 0);
  printf("command = %s\n",str);
  send(info->sock, str,strlen(str), 0); 
  hmp_4040_complete(info);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_set_state(HMP4040_INFO * info, INT channel,int value)
{
  printf("asking to set the state of channel %d to %d\n",channel+1,value&1);
  hmp4040_channel_select(info,channel);
  hmp_4040_complete(info);
  char str[128];
  char* str1 = "OUTP:STAT";
  char* str2 = " ";
  char* str3 = (char *)calloc(1,sizeof(int));
  sprintf(str3,"%d",value & 1);
  char* str4 = "\n";  
  strcpy(str,str1);
  strcat(str,str2);
  strcat(str,str3);
  strcat(str,str4);
  send(info->sock,str,strlen(str), 0);
  printf("command = %s\n",str);
  send(info->sock, str,strlen(str), 0); 
  hmp_4040_complete(info);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_rall(HMP4040_INFO * info)
{
  return FE_SUCCESS;   
}

/*----------------------------------------------------------------------------*/

INT hmp4040_get(HMP4040_INFO * info, INT channel, float *pvalue)
{
 return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_get_voltage(HMP4040_INFO * info, INT channel, float *pvalue)
{
  //printf("asking for the voltage of channel %d\n",channel);
  hmp4040_channel_select(info,channel);
  hmp_4040_complete(info);
  char* str = "MEAS:VOLT?\n";
  send(info->sock,str,strlen(str), 0);
  char str_reply[1024];
  int status = recv_string(info->sock, str_reply, sizeof(str_reply), 10000);
  if (status <= 0)
  {
    cm_msg(MERROR, "hmp4040_get_voltage", "cannot voltage retrieve data from %s", info->hmp4040_settings.dev_ip);
    return FE_ERR_HW;
   }
   //printf("voltage of channel %d is %f\n",channel,atof(str_reply));
  *pvalue=atof(str_reply);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_get_current(HMP4040_INFO * info, INT channel, float *pvalue)
{
  //printf("asking for the voltage of channel %d\n",channel);
  hmp4040_channel_select(info,channel);
  hmp_4040_complete(info);
  char* str = "MEAS:CURR?\n";
  send(info->sock,str,strlen(str), 0);
  char str_reply[1024];
  int status = recv_string(info->sock, str_reply, sizeof(str_reply), 10000);
  if (status <= 0)
  {
    cm_msg(MERROR, "hmp4040_get_current", "cannot current retrieve data from %s", info->hmp4040_settings.dev_ip);
    return FE_ERR_HW;
   }
   //printf("current of channel %d is %f\n",channel,atof(str_reply));
  *pvalue=atof(str_reply);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_get_status(HMP4040_INFO * info, INT channel, float *pvalue)
{
  //printf("asking for the voltage of channel %d\n",channel);
  hmp4040_channel_select(info,channel);
  hmp_4040_complete(info);
  char* str = "OUTP:STAT?\n";
  send(info->sock,str,strlen(str), 0);
  char str_reply[1024];
  int status = recv_string(info->sock, str_reply, sizeof(str_reply), 10000);
  if (status <= 0)
  {
    cm_msg(MERROR, "hmp4040_get_status", "cannot retrieve status data from %s", info->hmp4040_settings.dev_ip);
    return FE_ERR_HW;
   }
   //printf("status of channel %d is %d\n",channel,atoi(str_reply));
  //CHANNEL_SELECT + str(ch) + NL
  *pvalue=atoi(str_reply);
  return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_get_demand(HMP4040_INFO * info, INT channel, float *pvalue)
{
 return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT hmp4040_get_default_threshold(HMP4040_INFO * info, INT channel, float *pvalue)
{
 return FE_SUCCESS;
}



/*----------------------------------------------------------------------------*/

INT hmp4040_get_label(HMP4040_INFO * info, INT channel, char *name)
{
 return FE_SUCCESS;
}


/*---- device driver entry point -----------------------------------*/

INT hmp4040dev(INT cmd, ...)
{
   va_list argptr;
   HNDLE hKey;
   INT channel, status;
   float value, *pvalue;
   int status_value;
   void *info;


   va_start(argptr, cmd);
   status = FE_SUCCESS;

   switch (cmd) {
   case CMD_INIT:
      hKey = va_arg(argptr, HNDLE);
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      status = hmp4040_init(hKey,(void**)info, channel);
      break;

   case CMD_EXIT:
      info = va_arg(argptr, void *);
      status = hmp4040_exit((HMP4040_INFO* )info);
      break;

   case CMD_SET:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      value = (float) va_arg(argptr, double);
      status = hmp4040_set((HMP4040_INFO* ) info, channel, value);
      break;

   case CMD_GET:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pvalue = va_arg(argptr, float *);
      status = hmp4040_get_voltage((HMP4040_INFO* )info, channel, pvalue);
     break;
      
   case CMD_GET_CURRENT:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pvalue = va_arg(argptr, float *);
      status = hmp4040_get_current((HMP4040_INFO* )info, channel, pvalue);
     break;
     
   case CMD_SET_CURRENT_LIMIT:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      value = (float) va_arg(argptr, double);
      status = hmp4040_set_current((HMP4040_INFO* )info, channel, value);
     break;
     
   case CMD_SET_CHSTATE:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      value = (float)va_arg(argptr,double);
      printf("channel %d, value %f\n",channel, value);
      status = hmp4040_set_state((HMP4040_INFO* )info, channel, (int)value);
     break;
     

   case CMD_GET_DEMAND:
      //info = va_arg(argptr, void *);
      //channel = va_arg(argptr, INT);
      //pvalue = va_arg(argptr, float *);
      //status = hmp4040_get_demand(info, channel, pvalue);
      break;
      
   case CMD_GET_STATUS:
      info = va_arg(argptr, void *);
      channel = va_arg(argptr, INT);
      pvalue = va_arg(argptr, float *);
      //printf("status requested from device driver for channel %d channel\n",channel);
      status = hmp4040_get_status((HMP4040_INFO* ) info, channel, pvalue);
      break;

   case CMD_GET_LABEL:
      //info = va_arg(argptr, void *);
      //channel = va_arg(argptr, INT);
      //name = va_arg(argptr, char *);
      //status = hmp4040_get_label(info, channel, name);
      break;

   case CMD_GET_THRESHOLD:
      //info = va_arg(argptr, void *);
      //channel = va_arg(argptr, INT);
      //pvalue = va_arg(argptr, float *);
      //status = hmp4040_get_default_threshold(info, channel, pvalue);
      break;

   default:
      break;
   }

   va_end(argptr);

   return status;
}

/*------------------------------------------------------------------*/
