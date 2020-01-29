/********************************************************************\

  Name:         experim.h
  Created by:   ODBedit program

  Contents:     This file contains C structures for the "Experiment"
                tree in the ODB and the "/Analyzer/Parameters" tree.

                Additionally, it contains the "Settings" subtree for
                all items listed under "/Equipment" as well as their
                event definition.

                It can be used by the frontend and analyzer to work
                with these information.

                All C structures are accompanied with a string represen-
                tation which can be used in the db_create_record function
                to setup an ODB structure which matches the C structure.

  Created on:   Thu May 16 15:00:59 2019

\********************************************************************/

#ifndef EXCL_STREAM

#define STREAM_COMMON_DEFINED

typedef struct {
  WORD      event_id;
  WORD      trigger_mask;
  char      buffer[32];
  INT       type;
  INT       source;
  char      format[8];
  BOOL      enabled;
  INT       read_on;
  INT       period;
  double    event_limit;
  DWORD     num_subevents;
  INT       log_history;
  char      frontend_host[32];
  char      frontend_name[32];
  char      frontend_file_name[256];
  char      status[256];
  char      status_color[32];
  BOOL      hidden;
  INT       write_cache_size;
} STREAM_COMMON;

#define STREAM_COMMON_STR(_name) const char *_name[] = {\
"[.]",\
"Event ID = WORD : 1",\
"Trigger mask = WORD : 0",\
"Buffer = STRING : [32] SYSTEM",\
"Type = INT : 2",\
"Source = INT : 0",\
"Format = STRING : [8] MIDAS",\
"Enabled = BOOL : y",\
"Read on = INT : 257",\
"Period = INT : 100",\
"Event limit = DOUBLE : 0",\
"Num subevents = DWORD : 0",\
"Log history = INT : 0",\
"Frontend host = STRING : [32] localhost",\
"Frontend name = STRING : [32] Stream Frontend",\
"Frontend file name = STRING : [256] /home/labor/online/frontends/farm/farm_fe.cu",\
"Status = STRING : [256] Running",\
"Status color = STRING : [32] var(--mgreen)",\
"Hidden = BOOL : n",\
"Write cache size = INT : 100000",\
"",\
NULL }

#define STREAM_SETTINGS_DEFINED

typedef struct {
  struct {
    BOOL      enable;
    INT       divider;
    BOOL      enable_pixel;
    BOOL      enable_fibre;
    BOOL      enable_tile;
    INT       npixel;
    INT       nfibre;
    INT       ntile;
  } datagenerator;
} STREAM_SETTINGS;

#define STREAM_SETTINGS_STR(_name) const char *_name[] = {\
"[Datagenerator]",\
"Divider = INT : 1000",\
"Enable = BOOL : n",\
"",\
NULL }

#endif

#ifndef EXCL_SCALER

#define SCALER_COMMON_DEFINED

typedef struct {
  WORD      event_id;
  WORD      trigger_mask;
  char      buffer[32];
  INT       type;
  INT       source;
  char      format[8];
  BOOL      enabled;
  INT       read_on;
  INT       period;
  double    event_limit;
  DWORD     num_subevents;
  INT       log_history;
  char      frontend_host[32];
  char      frontend_name[32];
  char      frontend_file_name[256];
  char      status[256];
  char      status_color[32];
  BOOL      hidden;
  INT       write_cache_size;
} SCALER_COMMON;

#define SCALER_COMMON_STR(_name) const char *_name[] = {\
"[.]",\
"Event ID = WORD : 2",\
"Trigger mask = WORD : 0",\
"Buffer = STRING : [32] SYSTEM",\
"Type = INT : 1",\
"Source = INT : 0",\
"Format = STRING : [8] MIDAS",\
"Enabled = BOOL : y",\
"Read on = INT : 377",\
"Period = INT : 10000",\
"Event limit = DOUBLE : 0",\
"Num subevents = DWORD : 0",\
"Log history = INT : 1",\
"Frontend host = STRING : [32] localhost",\
"Frontend name = STRING : [32] Mu3e DAQ",\
"Frontend file name = STRING : [256] /home/labor/online/frontends/farm/mudaq_frontend.cu",\
"Status = STRING : [256] Ready",\
"Status color = STRING : [32] #00FF00",\
"Hidden = BOOL : n",\
"Write cache size = INT : 100000",\
"",\
NULL }

#endif

