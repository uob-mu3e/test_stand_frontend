
/********************************************************************\
 
 Name:         environment_control.cpp
 Created by: Luigi Vigani   
        Stefan Ritt
 		Frederik Wauters
 		Andreas Knecht
 
 Contents:     Slow control frontend for the mu3e  experiment, environment and pixel temperature sensors on 2 SCS-3000
  
 \********************************************************************/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "mscb.h"
#include "midas.h"
#include "mfe.h"
#include "device/mscbdev.h"
#include "class/multi.h"
#include "history.h"
#include <cmath>
#include <map>
#include <string>

double convert_temperature(float volt) {
    char get_str[256];
    double scale;
    int sizze = sizeof(double);
    sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Temperature/Scale");
    int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
    double expo;
    sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Temperature/Exp");
    int ok_expo = db_get_value(hDB, 0, get_str, &expo, &sizze, TID_DOUBLE, FALSE);
    double offset;
    sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Temperature/Offset");
    int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);

    return (scale * exp(-1*expo * (10000 * (volt / (5 - volt)))) - offset);
}
double convert_pressure(float volt) {
    char get_str[256];
    double scale;
    int sizze = sizeof(double);
    sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Pressure/Scale");
    int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
    return (volt * scale);
}
double convert_flow(float volt) {
    return 1000. / (volt / (2500. / 4095.)) / (78. * 60.);
}

double convert_temperature_pixels(float volt) {
    char get_str[256];
    double scale;
    int sizze = sizeof(double);
    sprintf(get_str, "/Equipment/EnvPixels/Conversion_Parameters/Temperature/Scale");
    int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
    double offset;
    sprintf(get_str, "/Equipment/EnvPixels/Conversion_Parameters/Temperature/Offset");
    int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);

    return offset + scale * volt / 2;
}

double lm35_function(float volt) {
    return 0.1 * 1000 * volt;
}

double honeywell_function(float volt, double temp) {
    double RH = ((volt / 5) - 0.16) / 0.0062;
    return RH / (1.0546 - 0.00216 * temp);
}

double pcmini52_rh_function(float volt) {
    return volt * 1000 * 0.02;
}

double pcmini52_t_function(float volt) {
    return volt * 1000 * 0.02 -20;
}

double o2_function(float volt) {
    return volt * 10;
}

double water_function(float volt) {
    if (volt < 300)
        return 0;
    else
        return 1;
}

double dewpoint_function(double temp, double rh) {
    //Magnus formula in Wikipedia
    float b = 17.67;
    float c = 243.5;
    double gamma = log(rh / 100) + (b*temp)/(c+temp);
    return (c * gamma) / (b - gamma);
}

double magnetic_field_function(float v1, float v2, float v3, float v4) {

    double V = (abs(v1 - v2) / abs(v3 - v4))*150*25;

    double pars[10] = { 0.560215e1, 0.102680e3,0.106348e-2, 0.732378e-5, 0.359616e-7,
        0.361184e-9, 0.879543e-12, -0.126396e-13, -0.867166e-17, 0.132761e-18 };
    double result = 0;
    double x = 1;
    for (int i = 0; i < 10; i++) {
        result += pars[i] * x;
        x *= V;
    }
    return result / 1e4;
}

double vhp_function(float v1, float v2) {
    return abs(v1 - v2)*1000;
}

double ihp_function(float v3, float v4) {
    return 1000* abs(v3 - v4) / 25;
}

double vhp_scaled_funtion(float v1, float v2, float v3, float v4) {
    return (abs(v1 - v2) / abs(v3 - v4)) * 150 * 25;
}

std::map<int, std::pair<std::string, std::string> > extra_env = {
    {0, {"Env_US1 - Water", "Water"}},
    {1, {"Env_US1 - PCMini52_RH", "PCMini52RH"} },
    {2, {"Env_US1 - PCMini52_T", "PCMini52T" }},
    {3, {"Env_US2 - O2-7-top", "O2" }},
    {4, {"Env_US2 - HIH4040_top", "Honeywell" }},
    {5, {"Env_US2 - LM35_Fibre", "LM35" }},
    {6, {"Env_US2 - LM35_FEC1", "LM35" }},
    {7, {"Env_US2 - LM35_top", "LM35" }},

    {9, {"Env_DS1 - Water", "Water" }},
    {11, {"Env_DS2 - O2-4-top", "O2" }},
    {12, {"Env_DS2 - HIH4040_top", "Honeywell" }},
    {13, {"Env_DS2 - LM35_Fibre", "LM35" }},
    {14, {"Env_DS2 - LM35_FEC1", "LM35" }},
    {15, {"Env_DS2 - LM35_top", "LM35" }},

    {18, {"Env_DS3 - O2-11-bottom", "O2" }},
    {19, {"Env_US3 - O2-6-central", "O2" }},
    {20, {"Env_US3 - HIH4040_bottom", "Honeywell" }},
    {21, {"Env_US3 - HIH4040_central", "Honeywell" }},
    //{23, {"Env_US6 - O2_3", "undefined" }},
    //{24, {"Env_US6 - LM35_3", "undefined" }},
    //{25, {"Env_US6 - HIH_3", "undefined" }},

    {27, {"Env_DS3 - O2-11-bottom", "O2" }},
    {28, {"Env_DS3 - O2-6-central", "O2" }},
    {29, {"Env_DS3 - HIH4040_bottom", "Honeywell" }},
    {30, {"Env_DS3 - HIH4040_central", "Honeywell" }}
    //{32, {"Env_DS6 - LM35_6", "undefined" }},
    //{33, {"Env_DS6 - HIH_6", "undefined" }},
    //{34, {"Env_DS6 - O2_6", "undefined" }}
};

std::string dew_point_temp = "Env_US1 - PCMini52_T";
std::string dew_point_rh = "Env_US1 - PCMini52_RH";
std::string dew_point_temp_us = "Env_US2 - LM35_top";
std::string dew_point_temp_ds = "Env_DS2 - LM35_top";
std::string dew_point_rh_us = "Env_US2 - HIH4040_top";
std::string dew_point_rh_ds = "Env_DS2 - HIH4040_top";

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "SC Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = TRUE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 0;//1000;

/* maximum event size produced by this frontend */
INT max_event_size = 5*100000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 2 *5* 100000;

/* Overwrite equipment struct in ODB from values in code*/
BOOL equipment_common_overwrite = FALSE;

/*-- Equipment list ------------------------------------------------*/

/* device driver list */

DEVICE_DRIVER multi_driver_env[] = {
    {"ADC", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

DEVICE_DRIVER multi_driver_pix[] = {
    {"ADC", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

EQUIPMENT equipment[] = {
        
    {"Environment",              /* equipment name */
        {150, 0,                      /* event ID, trigger mask */
            "SYSTEM",                  /* event buffer */
            EQ_SLOW,                   /* equipment type */
            0,                         /* event source */
            "MIDAS",                   /* format */
            TRUE,                      /* enabled */
            RO_ALWAYS,        /* read when running and on transitions */
            60000,                     /* produce event every 60 sec */
            1000,                      /* read one event every second */
            0,                         /* number of sub events */
            1,                         /* log history every event */
            "", "", ""} ,
        cd_multi_read,              /* readout routine */
        cd_multi,                   /* class driver main routine */
        multi_driver_env,               /* device driver list */
        NULL,                       /* init string */
    },
    
    {"EnvPixels",              /* equipment name */
        {151, 0,                      /* event ID, trigger mask */
            "SYSTEM",                  /* event buffer */
            EQ_SLOW,                   /* equipment type */
            0,                         /* event source */
            "MIDAS",                   /* format */
            TRUE,                      /* enabled */
            RO_ALWAYS,        /* read when running and on transitions */
            60000,                     /* produce event every 60 sec */
            1000,                      /* read one event every second */
            0,                         /* number of sub events */
            1,                         /* log history every event */
            "", "", ""} ,
        cd_multi_read,              /* readout routine */
        cd_multi,                   /* class driver main routine */
        multi_driver_pix,               /* device driver list */
        NULL,                       /* init string */
    },

    {""}
};

/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
{
    return 1;
};
INT interrupt_configure(INT cmd, INT source, PTYPE adr)
{
    return 1;
};

/*-- Function to define MSCB variables in a convenient way ---------*/

void mscb_define(const char *submaster,const char *equipment,const char *devname, DEVICE_DRIVER *driver,
                 int address, unsigned char var_index, char *name, double threshold)
{
    int i, dev_index, chn_index, chn_total;
    char str[256];
    float f_threshold;
    HNDLE hDB;
    
    cm_get_experiment_database(&hDB, NULL);
    
    if (submaster && submaster[0]) { //Default submaster: deprecated
        sprintf(str, "/Equipment/%s/Settings/Devices/%s/Device", equipment, devname);
        db_set_value(hDB, 0, str, submaster, 32, 1, TID_STRING);

        sprintf(str, "/Equipment/%s/Settings/Devices/%s/Pwd", equipment, devname);
        db_set_value(hDB, 0, str, "mu3e", 32, 1, TID_STRING);
    }
    else { //If no default submaster, look for it in the variables
        char dev_name[256];
        int size_dev_name = sizeof(dev_name);

        sprintf(str, "/Equipment/%s/Settings/Submaster", equipment);
        int has_subname = db_get_value(hDB, 0, str, dev_name, &size_dev_name, TID_STRING, FALSE);
        if (has_subname != DB_SUCCESS) {//If it is not set, set it to 'none'
            db_set_value(hDB, 0, str, "none", 32, 1, TID_STRING);
        }
        else {
            if (dev_name == "none") {
                sprintf(str, "/Equipment/%s/Settings/Devices/%s/Device", equipment, devname);
                db_set_value(hDB, 0, str, "none", 32, 1, TID_STRING);
            }
            else {//If it is present, set it also for the device
                sprintf(str, "/Equipment/%s/Settings/Devices/%s/Device", equipment, devname);
                db_set_value(hDB, 0, str, dev_name, size_dev_name, 1, TID_STRING);
            }
        }
    }
    
    /* find device in device driver */
    for (dev_index=0 ; driver[dev_index].name[0] ; dev_index++)
        if (equal_ustring(driver[dev_index].name, devname))
            break;
    
    if (!driver[dev_index].name[0]) {
        cm_msg(MERROR, "mscb_define", "Device \"%s\" not present in device driver list", devname);
        return;
    }
    
    /* count total number of channels */
    for (i=chn_total=0 ; i<=dev_index ; i++)
        chn_total += driver[i].channels;
    
    chn_index = driver[dev_index].channels;
    sprintf(str, "/Equipment/%s/Settings/Devices/%s/MSCB Address", equipment, devname);
    db_set_value_index(hDB, 0, str, &address, sizeof(int), chn_index, TID_INT, TRUE);
    sprintf(str, "/Equipment/%s/Settings/Devices/%s/MSCB Index", equipment, devname);
    db_set_value_index(hDB, 0, str, &var_index, sizeof(char), chn_index, TID_BYTE, TRUE);

    if (equipment == "Environment") {
        if (chn_index == 0) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Conversion_Parameters/Temperature/Scale", equipment);
            double scale_tmp;
            int size_scale_tmp = sizeof(double);
            int has_subname = db_get_value(hDB, 0, set_str, &scale_tmp, &size_scale_tmp, TID_DOUBLE, FALSE);
            if (has_subname != DB_SUCCESS) {
                double scale = 79.5;
                db_set_value(hDB, 0, set_str, &scale, sizeof(double), 1, TID_DOUBLE);
                sprintf(set_str, "/Equipment/%s/Conversion_Parameters/Temperature/Exp", equipment);
                double expo = 0.00011;
                db_set_value(hDB, 0, set_str, &expo, sizeof(double), 1, TID_DOUBLE);
                sprintf(set_str, "/Equipment/%s/Conversion_Parameters/Temperature/Offset", equipment);
                double offset = 1.76;
                db_set_value(hDB, 0, set_str, &offset, sizeof(double), 1, TID_DOUBLE);

                sprintf(set_str, "/Equipment/%s/Conversion_Parameters/Pressure/Scale", equipment);
                double scalep = 2.25;
                db_set_value(hDB, 0, set_str, &scalep, sizeof(double), 1, TID_DOUBLE);
            }
        }
        else if (chn_index == 45 || chn_index == 47 || chn_index == 49 || chn_index == 51) {
            int idx = (int)((chn_index - 45) / 2);
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Converted/Temperature_%d", equipment, idx);
            double temp = 30;
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);

            //Alarms
            char has_str[256];
            char got_str[256];
            int size_has_str = sizeof(got_str);
            sprintf(has_str, "/Alarms/Alarms/SSF_Cooling_TempHigh_Alarm_0/Condition");
            int has_subname = db_get_value(hDB, 0, has_str, &got_str, &size_has_str, TID_STRING, FALSE);

            if (/*has_subname != DB_SUCCESS*/ true) {
                char al_str[256];
                sprintf(al_str, "SSF_Cooling_TempHigh_Alarm_%d", idx);
                char al_cond[256];
                sprintf(al_cond, "/Equipment/%s/Converted/Temperature_%d > 30", equipment, idx);
                char al_message[256];
                sprintf(al_message, "Temperature too high for SSF chiller (Box %d)", idx + 1);
                al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);
                sprintf(al_str, "/Alarms/Alarms/SSF_Cooling_TempHigh_Alarm_%d/Active", idx);
                bool set0 = true;
                db_set_value(hDB, 0, al_str, &set0, sizeof(int), 1, TID_BOOL);

                char al_strw[256];
                sprintf(al_strw, "SSF_Cooling_TempHigh_Warning_%d", idx);
                char al_condw[256];
                sprintf(al_condw, "/Equipment/%s/Converted/Temperature_%d > 25", equipment, idx);
                char al_messagew[256];
                sprintf(al_messagew, "Temperature quite high for SSF chiller (Box %d)", idx + 1);
                al_define_odb_alarm(al_strw, al_condw, "Warning", al_messagew);
                sprintf(al_str, "/Alarms/Alarms/SSF_Cooling_TempHigh_Warning_%d/Active", idx);
                bool set1 = true;
                db_set_value(hDB, 0, al_str, &set1, sizeof(int), 1, TID_BOOL);

                sprintf(al_str, "SSF_Cooling_TempLow_Alarm_%d", idx);
                sprintf(al_cond, "/Equipment/%s/Converted/Temperature_%d < 10", equipment, idx);
                sprintf(al_message, "Temperature too low for SSF chiller (Box %d)", idx + 1);
                al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);
                sprintf(al_str, "/Alarms/Alarms/SSF_Cooling_TempLow_Alarm_%d/Active", idx);
                bool set2 = true;
                db_set_value(hDB, 0, al_str, &set2, sizeof(int), 1, TID_BOOL);

                sprintf(al_strw, "SSF_Cooling_TempLow_Warning_%d", idx);
                sprintf(al_condw, "/Equipment/%s/Converted/Temperature_%d < 15", equipment, idx);
                sprintf(al_messagew, "Temperature quite low for SSF chiller (Box %d)", idx + 1);
                al_define_odb_alarm(al_strw, al_condw, "Warning", al_messagew);
                sprintf(al_str, "/Alarms/Alarms/SSF_Cooling_TempLow_Warning_%d/Active", idx);
                bool set3 = true;
                db_set_value(hDB, 0, al_str, &set3, sizeof(int), 1, TID_BOOL);
            }
        }
        else if (chn_index == 46 || chn_index == 48 || chn_index == 50 || chn_index == 52) {
            char set_str[256];
            int idx = (int)((chn_index - 46) / 2);
            sprintf(set_str, "/Equipment/%s/Converted/Pressure_%d", equipment, idx);
            double temp = 1;
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        else if (chn_index == 42 || chn_index == 43) {
            char set_str[256];
            int idx = (int)((chn_index - 42));
            sprintf(set_str, "/Equipment/%s/Converted/Flow_%d", equipment, idx);
            double temp = 1;
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        else if (extra_env.find(chn_index) != extra_env.end()) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Converted/%s", equipment, extra_env[chn_index].first.c_str());
            double temp = 1.;
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
            std::string nickname = extra_env[chn_index].first.substr(0, 10);
            char has_str[256];
            char got_str[256];
            int size_has_str = sizeof(got_str);
            sprintf(has_str, "/Alarms/Alarms/InCage_%s_temp_AlHigh/Condition", nickname.c_str());
            int has_subname = db_get_value(hDB, 0, has_str, &got_str, &size_has_str, TID_STRING, FALSE);
            if (/*has_subname != DB_SUCCESS*/ true) {
                char al_str[256];
                sprintf(al_str, "%s_AH", extra_env[chn_index].first.c_str());
                char al_cond[256];
                char al_message[256];
                char al_strw[256];
                sprintf(al_strw, "%s_WH", extra_env[chn_index].first.c_str());
                char al_condw[256];
                char al_messagew[256];
                if (extra_env[chn_index].second == "PCMini52T") {//temperature
                    sprintf(al_cond, "/Equipment/%s/Converted/%s > 40", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_message, "Temperature too high at %s", extra_env[chn_index].first.c_str());
                    sprintf(al_condw, "/Equipment/%s/Converted/%s > 30", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_messagew, "Temperature quite high at %s", extra_env[chn_index].first.c_str());
                }
                else if (extra_env[chn_index].second == "PCMini52RH") {//humidity
                    sprintf(al_cond, "/Equipment/%s/Converted/%s > 80", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_message, "Humidity too high at %s", extra_env[chn_index].first.c_str());
                    sprintf(al_condw, "/Equipment/%s/Converted/%s > 70", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_messagew, "Humidity quite high at %s", extra_env[chn_index].first.c_str());
                }
                else if (extra_env[chn_index].second == "LM35") {//temperature
                    if (extra_env[chn_index].first.find("Fibre") == std::string::npos)
                        sprintf(al_cond, "/Equipment/%s/Converted/%s > 40", equipment, extra_env[chn_index].first.c_str());
                    else
                        sprintf(al_cond, "/Equipment/%s/Converted/%s > 25", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_message, "Temperature too high at %s", extra_env[chn_index].first.c_str());
                    sprintf(al_condw, "/Equipment/%s/Converted/%s > 30", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_messagew, "Temperature quite high at %s", extra_env[chn_index].first.c_str());
                }
                else if (extra_env[chn_index].second == "Honeywell") {//humidity
                    sprintf(al_cond, "/Equipment/%s/Converted/%s > 80", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_message, "Humidity too high at %s", extra_env[chn_index].first.c_str());
                    sprintf(al_condw, "/Equipment/%s/Converted/%s > 70", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_messagew, "Humidity quite high at %s", extra_env[chn_index].first.c_str());
                }
                else if (extra_env[chn_index].second == "O2") {//Oxygen
                    sprintf(al_cond, "/Equipment/%s/Converted/%s > 80", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_message, "Oxygen level too high at %s", extra_env[chn_index].first.c_str());
                    sprintf(al_condw, "/Equipment/%s/Converted/%s > 0.2", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_messagew, "oxygen level quite high at %s", extra_env[chn_index].first.c_str());
                }
                else if (extra_env[chn_index].second == "Water") {//Water
                    sprintf(al_cond, "/Equipment/%s/Converted/%s > 0.5 ", equipment, extra_env[chn_index].first.c_str());
                    sprintf(al_message, "Water detected at %s", extra_env[chn_index].first.c_str());
                }
                al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);
                char al_str_active[256];
                sprintf(al_str_active, "/Alarms/Alarms/%s/Active", al_str);
                bool set0 = true;
                db_set_value(hDB, 0, al_str_active, &set0, sizeof(int), 1, TID_BOOL);
                if (extra_env[chn_index].second != "Water") {
                    al_define_odb_alarm(al_strw, al_condw, "Warning", al_messagew);
                    sprintf(al_str_active, "/Alarms/Alarms/%s/Active", al_strw);
                    bool set1 = true;
                    db_set_value(hDB, 0, al_str_active, &set1, sizeof(int), 1, TID_BOOL);
                }
            }
        }
    }
    else if (equipment == "EnvPixels") {
        if (chn_index == 0) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Conversion_Parameters/Temperature/Scale", equipment);
            double scale_tmp;
            int size_scale_tmp = sizeof(double);
            int has_subname = db_get_value(hDB, 0, set_str, &scale_tmp, &size_scale_tmp, TID_DOUBLE, FALSE);
            if (has_subname != DB_SUCCESS) {
                double scale = -331.8;
                db_set_value(hDB, 0, set_str, &scale, sizeof(double), 1, TID_DOUBLE);
                sprintf(set_str, "/Equipment/%s/Conversion_Parameters/Temperature/Offset", equipment);
                double offset = 228.;
                db_set_value(hDB, 0, set_str, &offset, sizeof(double), 1, TID_DOUBLE);
            }
        }
        char set_str[256];
        if (chn_index < 42)
            sprintf(set_str, "/Equipment/%s/ConvertedMagnet/Temperature_%d", equipment, chn_index);
        else if (chn_index == 44)
            sprintf(set_str, "/Equipment/%s/ConvertedMagnet/B_Tesla", equipment);
        else if (chn_index == 45)
            sprintf(set_str, "/Equipment/%s/ConvertedMagnet/Vhp_mV", equipment);
        else if (chn_index == 46)
            sprintf(set_str, "/Equipment/%s/ConvertedMagnet/Vhp_scaled_mV", equipment);
        else if (chn_index == 47)
            sprintf(set_str, "/Equipment/%s/ConvertedMagnet/Ihp_mA", equipment);
        double temp = 0;
        int idx = chn_index;
        //db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), idx, TID_DOUBLE, TRUE);
        db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);

        char has_str[256];
        char got_str[256];
        int size_has_str = sizeof(got_str);
        sprintf(has_str, "/Alarms/Alarms/Pixel_Temperature_Alarm_0/Condition");
        int has_subname = db_get_value(hDB, 0, has_str, &got_str, &size_has_str, TID_STRING, FALSE);
        if (/*has_subname != DB_SUCCESS*/ true) {

            if (chn_index != 2
                && chn_index != 34 && chn_index != 35 && chn_index != 36 && chn_index != 37 && chn_index != 38
                && chn_index < 42) {
                //Alarms
                char al_str[256];
                sprintf(al_str, "Pixel_Temperature_Alarm_%d", chn_index);
                char al_cond[256];
                sprintf(al_cond, "/Equipment/%s/Converted/Temperature_%d > 70", equipment, chn_index);
                char al_message[256];
                sprintf(al_message, "Temperature too high for pixel (MSCB channel %d)", chn_index);
                al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);
                sprintf(al_str, "/Alarms/Alarms/Pixel_Temperature_Alarm_%d/Active", chn_index);
                bool set0 = true;
                db_set_value(hDB, 0, al_str, &set0, sizeof(int), 1, TID_BOOL);

                char al_strw[256];
                sprintf(al_strw, "Pixel_Temperature_Warning_%d", chn_index);
                char al_condw[256];
                sprintf(al_condw, "/Equipment/%s/Converted/Temperature_%d > 60", equipment, chn_index);
                char al_messagew[256];
                sprintf(al_messagew, "Temperature quite high for pixel (MSCB channel %d)", chn_index);
                al_define_odb_alarm(al_strw, al_condw, "Warning", al_messagew);
                sprintf(al_str, "/Alarms/Alarms/Pixel_Temperature_Warning_%d/Active", chn_index);
                bool set1 = true;
                db_set_value(hDB, 0, al_str, &set1, sizeof(int), 1, TID_BOOL);
            }
            if (chn_index == 44) {
                char al_str[256];
                sprintf(al_str, "MagnetCurrenLow_Wa");
                char al_cond[256];
                sprintf(al_cond, "/Equipment/EnvPixels/ConvertedMagnet/Ihp_mA < 5", equipment, chn_index);
                char al_message[256];
                sprintf(al_message, "Magnet current quite low", chn_index);
                al_define_odb_alarm(al_str, al_cond, "Warning", al_message);
                //sprintf(al_str, "/Alarms/Alarms/Pixel_Temperature_Alarm_%d/Active", chn_index);
                //bool set0 = true;
                //db_set_value(hDB, 0, al_str, &set0, sizeof(int), 1, TID_BOOL);
            }
        }
    }

    if (threshold != -1) {
        sprintf(str, "/Equipment/%s/Settings/Update Threshold", equipment);
        f_threshold = (float) threshold;
        db_set_value_index(hDB, 0, str, &f_threshold, sizeof(float), chn_total, TID_FLOAT, TRUE);
    }
    
    if (name && name[0]) {
        sprintf(str, "/Equipment/%s/Settings/Names Input", equipment);
        db_set_value_index(hDB, 0, str, name, 32, chn_total, TID_STRING, TRUE);
    }
    
    /* increment number of channels for this driver */
    driver[dev_index].channels++;
}

/*-- Error dispatcher causing communiction alarm -------------------*/

void scfe_error(const char *error)
{
    char str[256];
    
    strlcpy(str, error, sizeof(str));
    cm_msg(MERROR, "scfe_error", "%s", str);
    al_trigger_alarm("MSCB", str, "MSCB Alarm", "Communication Problem", AT_INTERNAL);
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
    HNDLE hDB;
    //int i, im, iv;
    //char str[80];
    
    cm_get_experiment_database(&hDB, NULL);
    
    /* set error dispatcher for alarm functionality */
    mfe_set_error(scfe_error);
    
    /* set maximal retry count */
    mscb_set_max_retry(100);
    
    /* SCS-3000 environment */
    printf("Initializing Environment...\n");
    for (unsigned int i=0 ; i<71 ; i++)
        //mscb_define("mscb400.mu3e", "Environment", "ADC", multi_driver_env, 65535, i, NULL, 0.0005);//Default submaster (deprecated)
        mscb_define("", "Environment", "ADC", multi_driver_env, 65535, i, NULL, 0.0005);

    /* SCS-3000 pixels */
    printf("Initializing EnvPixels...\n");
    for (unsigned int j = 0; j < 48; j++)
        //mscb_define("mscb334.mu3e", "EnvPixels", "ADC", multi_driver_pix, 0, j, NULL, 0.0005);//Default submaster (deprecated)
        mscb_define("", "EnvPixels", "ADC", multi_driver_pix, 0, j, NULL, 0.0005);

    //Alarms based on dew point
    char set_str_dp[256];
    sprintf(set_str_dp, "/Equipment/Environment/Computed/DewPoint");
    double tempdp0 = 100;
    db_set_value(hDB, 0, set_str_dp, &tempdp0, sizeof(double), 1, TID_DOUBLE);
    /*sprintf(set_str_dp, "/Equipment/Environment/Computed/DewPointUS");
    double tempdp = 1.;
    db_set_value(hDB, 0, set_str_dp, &tempdp, sizeof(double), 1, TID_DOUBLE);
    sprintf(set_str_dp, "/Equipment/Environment/Computed/DewPointDS");
    double tempdp2 = 1.;
    db_set_value(hDB, 0, set_str_dp, &tempdp2, sizeof(double), 1, TID_DOUBLE);*/
    char al_str[256];
    sprintf(al_str, "DewPoint_Alarm");
    char al_cond[256];
    sprintf(al_cond, "/Equipment/Environment/Computed/DewPoint > 20");
    char al_message[256];
    sprintf(al_message, "Dew Point too high!");
    al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);
    sprintf(al_str, "/Alarms/Alarms/DewPoint_Alarm/Active");
    bool set0 = true;
    db_set_value(hDB, 0, al_str, &set0, sizeof(int), 1, TID_BOOL);
    sprintf(al_str, "DewPoint_Warning");
    sprintf(al_cond, "/Equipment/Environment/Computed/DewPoint > 15");
    sprintf(al_message, "Dew Point quite high!");
    al_define_odb_alarm(al_str, al_cond, "Warning", al_message);
    sprintf(al_str, "/Alarms/Alarms/DewPoint_Warning/Active");
    bool set1 = true;
    db_set_value(hDB, 0, al_str, &set1, sizeof(int), 1, TID_BOOL);

    std::vector<std::string> histo_vars;
    for (int cha = 0; cha < 42; ++cha) {
        if (cha != 2
            && cha != 34 && cha != 35 && cha != 36 && cha != 37 && cha != 38
            && cha < 42) {
            char variab[256];
            int port = (int)(cha / 8);
            int in_ch = cha % 8;
            sprintf(variab, "EnvPixels:P%dUIn%d", port, in_ch);
            histo_vars.push_back((std::string)(variab));
        }
    }
    hs_define_panel("Pixel", "Temperature", histo_vars);
    for (int cha = 0; cha < 42; ++cha) {
        if (cha != 2
            && cha != 34 && cha != 35 && cha != 36 && cha != 37 && cha != 38
            && cha < 42) {
            char variab[256];
            int port = (int)(cha / 8);
            int in_ch = cha % 8;
            sprintf(variab, "/History/Display/Pixel/Temperature/Label");
            char vvalue[32];
            sprintf(vvalue, "Module %d", cha);
            db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), cha, TID_STRING, TRUE);
            sprintf(variab, "/History/Display/Pixel/Temperature/Formula");
            char get_str[256];
            double scale;
            int sizze = sizeof(double);
            sprintf(get_str, "/Equipment/EnvPixels/Conversion_Parameters/Temperature/Scale");
            int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
            double offset;
            sprintf(get_str, "/Equipment/EnvPixels/Conversion_Parameters/Temperature/Offset");
            int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);
            char formula[256];
            sprintf(formula, "%.3f*x+%.3f", scale, offset);
            db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), cha, TID_STRING, TRUE);
            //if (cha % 10 == 0)
            //    ss_sleep(100);
        }
    }

    char set_str_hlink[256];
    sprintf(set_str_hlink, "/History/Links/Environment_Computed");
    char set_str_hlinkvar[256];
    int size_hlinkvar = sizeof(set_str_hlinkvar);
    int ok_hlink = db_get_value(hDB, 0, set_str_hlink, &set_str_hlinkvar, &size_hlinkvar, TID_KEY, FALSE);
    if (ok_hlink != DB_SUCCESS) {
        sprintf(set_str_hlinkvar, "/Equipment/Environment/Computed/");
        db_set_value(hDB, 0, set_str_hlink, &set_str_hlinkvar, sizeof(set_str_hlinkvar), 1, TID_LINK);
    }
    sprintf(set_str_hlink, "/History/Links/Magnet_Conv");
    char set_str_hlinkvarmag[256];
    int size_hlinkvarmag = sizeof(set_str_hlinkvarmag);
    int ok_hlinkmag = db_get_value(hDB, 0, set_str_hlink, &set_str_hlinkvarmag, &size_hlinkvarmag, TID_KEY, FALSE);
    if (ok_hlink != DB_SUCCESS) {
        sprintf(set_str_hlinkvar, "/Equipment/EnvPixels/ConvertedMagnet");
        db_set_value(hDB, 0, set_str_hlink, &set_str_hlinkvar, sizeof(set_str_hlinkvar), 1, TID_LINK);
    }
    //sprintf(set_str_hlink, "/History/Links/PixelSC_Converted");
    //sprintf(set_str_hlinkvar, "/Equipment/EnvPixels/Converted");
    //db_set_value(hDB, 0, set_str_hlink, &set_str_hlinkvar, sizeof(set_str_hlinkvar), 1, TID_LINK);

    std::vector<std::string> histo_vars_temp;
    std::vector<std::string> histo_vars_press;
    for (int cha = 0; cha < 4; ++cha) {
        char variab[256];
        int port = 5;
        int in_ch = cha*2;
        sprintf(variab, "Environment:P%dUIn%d", port, in_ch);
        histo_vars_temp.push_back((std::string)(variab));

        in_ch +=1;
        sprintf(variab, "Environment:P%dUIn%d", port, in_ch);
        histo_vars_press.push_back((std::string)(variab));
    }

    hs_define_panel("Environment", "SSF-temperatures", histo_vars_temp);
    hs_define_panel("Environment", "SSF-pressures", histo_vars_press);

    for (int cha = 0; cha < 4; ++cha) {
        char variab[256];
        sprintf(variab, "/History/Display/Environment/SSF-temperatures/Label");
        char vvalue[32];
        sprintf(vvalue, "Box %d", cha+1);
        db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), cha, TID_STRING, TRUE);
        sprintf(variab, "/History/Display/Environment/SSF-temperatures/Formula");
        char get_str[256];
        double scale;
        int sizze = sizeof(double);
        sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Temperature/Scale");
        int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
        double expo;
        sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Temperature/Exp");
        int ok_expo = db_get_value(hDB, 0, get_str, &expo, &sizze, TID_DOUBLE, FALSE);
        double offset;
        sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Temperature/Offset");
        int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);
        char formula[256];
        sprintf(formula, "%.1f*Math.exp(-%.5f*(10000*(x/(5-x))))-%.2f", scale, expo, offset);
        db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), cha, TID_STRING, TRUE);

        sprintf(variab, "/History/Display/Environment/SSF-pressures/Label");
        sprintf(vvalue, "Box %d", cha + 1);
        db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), cha, TID_STRING, TRUE);
        sprintf(variab, "/History/Display/Environment/SSF-pressures/Formula");
        sprintf(get_str, "/Equipment/Environment/Conversion_Parameters/Pressure/Scale");
        int ok_scale2 = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
        sprintf(formula, "%.3f * x", scale);
        db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), cha, TID_STRING, TRUE);
    }

    std::vector<std::string> histo_vars_incage_temp;
    std::vector<std::string> histo_vars_incage_rh;
    std::vector<std::string> histo_vars_incage_o2;
    std::vector<std::string> histo_vars_incage_water;

    for (auto it : extra_env) {
        if (it.second.second == "undefined")
            continue;
        char variab[256];
        int port = int(it.first / 9);
        int in_ch = it.first % 9;
        sprintf(variab, "Environment:P%dUIn%d", port, in_ch);
        if (it.second.second == "PCMini52T") {
            histo_vars_incage_temp.push_back((std::string)(variab));
        }
        else if (it.second.second == "PCMini52RH") {
            histo_vars_incage_rh.push_back((std::string)(variab));
        }
        else if (it.second.second == "LM35") {
            histo_vars_incage_temp.push_back((std::string)(variab));
        }
        else if (it.second.second == "Honeywell") {
            histo_vars_incage_rh.push_back((std::string)(variab));
        }
        else if (it.second.second == "O2") {
            histo_vars_incage_o2.push_back((std::string)(variab));
        }
        else if (it.second.second == "Water") {
            histo_vars_incage_water.push_back((std::string)(variab));
        }
    }
    histo_vars_incage_temp.push_back("Environment_Computed:DewPoint");

    hs_define_panel("Environment", "In-cage-temperatures", histo_vars_incage_temp);
    hs_define_panel("Environment", "In-cage-RH", histo_vars_incage_rh);
    hs_define_panel("Environment", "In-cage-O2", histo_vars_incage_o2);
    hs_define_panel("Environment", "In-cage-Water", histo_vars_incage_water);

    int idx_temp = 0;
    int idx_rh = 0;
    int idx_o2 = 0;
    int idx_wat = 0;
    char variab[256];
    char vvalue[32];
    char formula[256];

    for (auto it : extra_env) {
        if (it.second.second == "undefined")
            continue;
        int port = int(it.first / 9);
        int in_ch = it.first % 9;
        sprintf(variab, "Environment:P%dUIn%d", port, in_ch);
        if (it.second.second == "PCMini52T") {
            sprintf(variab, "/History/Display/Environment/In-cage-temperatures/Label");
            sprintf(vvalue, "%s", it.second.first.c_str());
            db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_temp, TID_STRING, TRUE);
            sprintf(variab, "/History/Display/Environment/In-cage-temperatures/Formula");
            sprintf(formula, "x*1000*0.02 -20");
            db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), idx_temp, TID_STRING, TRUE);
            idx_temp++;
        }
        else if (it.second.second == "PCMini52RH") {
            sprintf(variab, "/History/Display/Environment/In-cage-RH/Label");
            sprintf(vvalue, "%s", it.second.first.c_str());
            db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_rh, TID_STRING, TRUE);
            sprintf(variab, "/History/Display/Environment/In-cage-RH/Formula");
            sprintf(formula, "x*1000*0.02");
            db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), idx_rh, TID_STRING, TRUE);
            idx_rh++;
        }
        else if (it.second.second == "LM35") {
            sprintf(variab, "/History/Display/Environment/In-cage-temperatures/Label");
            sprintf(vvalue, "%s", it.second.first.c_str());
            db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_temp, TID_STRING, TRUE);
            sprintf(variab, "/History/Display/Environment/In-cage-temperatures/Formula");
            sprintf(formula, "0.1*1000*x");
            db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), idx_temp, TID_STRING, TRUE);
            idx_temp++;
        }
        else if (it.second.second == "Honeywell") {
            sprintf(variab, "/History/Display/Environment/In-cage-RH/Label");
            sprintf(vvalue, "%s", it.second.first.c_str());
            db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_rh, TID_STRING, TRUE);
            sprintf(variab, "/History/Display/Environment/In-cage-RH/Formula");
            sprintf(formula, "(((x/5)-0.16)/0.0062)/(1.0546-(0.00216*30))");
            db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), idx_rh, TID_STRING, TRUE);
            idx_rh++;
        }
        else if (it.second.second == "O2") {
            sprintf(variab, "/History/Display/Environment/In-cage-O2/Label");
            sprintf(vvalue, "%s", it.second.first.c_str());
            db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_o2, TID_STRING, TRUE);
            sprintf(variab, "/History/Display/Environment/In-cage-O2/Formula");
            sprintf(formula, "x*10");
            db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), idx_o2, TID_STRING, TRUE);
            idx_o2++;
        }
        else if (it.second.second == "Water") {
            sprintf(variab, "/History/Display/Environment/In-cage-O2/Label");
            sprintf(vvalue, "%s", it.second.first.c_str());
            db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_wat, TID_STRING, TRUE);
            //sprintf(variab, "/History/Display/Environment/In-cage-O2/Formula");
            //sprintf(formula, "x*10");
            //db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), idx_wat, TID_STRING, TRUE);
            idx_wat++;
        }
    }
    sprintf(variab, "/History/Display/Environment/In-cage-temperatures/Label");
    sprintf(vvalue, "Dew Point");
    db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_temp, TID_STRING, TRUE);
    //sprintf(variab, "/History/Display/Environment/In-cage-temperatures/Formula");
    //sprintf(vvalue, "x");
    //db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), idx_temp, TID_STRING, TRUE);


    return CM_SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
    return CM_SUCCESS;
}

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

/*-- Resuem Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
    return CM_SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{
    for (int ch = 0; ch < 4; ++ch) {
        char get_str[256];
        float volt;
        int size_volt = sizeof(float);
        sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", 45 + ch*2);
        int ok_temp = db_get_value(hDB, 0, get_str, &volt, &size_volt, TID_FLOAT, FALSE);
        if (ok_temp == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/Environment/Converted/Temperature_%d", ch);
            double temp = convert_temperature(volt);
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        float volt2;
        sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", 46 + ch * 2);
        int ok_press = db_get_value(hDB, 0, get_str, &volt2, &size_volt, TID_FLOAT, FALSE);
        if (ok_press == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/Environment/Converted/Pressure_%d", ch);
            double temp = convert_pressure(volt2);
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        if (ch > 1)
            continue;
        float volt3;
        sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", 42 + ch);
        int ok_flow = db_get_value(hDB, 0, get_str, &volt3, &size_volt, TID_FLOAT, FALSE);
        if (ok_flow == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/Environment/Converted/Flow_%d", ch);
            double temp = convert_flow(volt3);
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
    }

    for (int chp = 0; chp < 42; ++chp) {
        if (chp == 2 ||
            chp == 34 || chp == 35 || chp == 36 || chp == 37 || chp == 38)
            continue;
        char get_str[256];
        float volt;
        int size_volt = sizeof(float);
        sprintf(get_str, "/Equipment/EnvPixels/Variables/Input[%d]", chp);
        int ok_temp = db_get_value(hDB, 0, get_str, &volt, &size_volt, TID_FLOAT, FALSE);
        if (ok_temp == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/EnvPixels/Converted/Temperature_%d", chp);
            double temp = convert_temperature_pixels(volt);
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
    }

    for (auto it : extra_env) {
        if (it.second.second == "undefined")
            continue;

        char get_str[256];
        float volt;
        int size_volt = sizeof(float);
        sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", it.first);
        int ok_temp = db_get_value(hDB, 0, get_str, &volt, &size_volt, TID_FLOAT, FALSE);
        char set_str[256];
        if (it.second.second == "Honeywell") {
            char get_strt[256];
            float voltt;
            float temp = 20;
            sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", it.first);
            if (it.first == 4) {
                sprintf(get_strt, "/Equipment/Environment/Variables/Input[7]");
                int ok_tempt = db_get_value(hDB, 0, get_strt, &voltt, &size_volt, TID_FLOAT, FALSE);
                temp = lm35_function(voltt);
            }
            if (it.first == 12) {
                sprintf(get_strt, "/Equipment/Environment/Variables/Input[15]");
                int ok_tempt = db_get_value(hDB, 0, get_strt, &voltt, &size_volt, TID_FLOAT, FALSE);
                temp = lm35_function(voltt);
            }
            double rh = honeywell_function(volt, temp);
            sprintf(set_str, "/Equipment/Environment/Converted/%s", it.second.first.c_str());
            db_set_value(hDB, 0, set_str, &rh, sizeof(double), 1, TID_DOUBLE);
        }
        else if (it.second.second == "LM35") {
            double temp = lm35_function(volt);
            sprintf(set_str, "/Equipment/Environment/Converted/%s", it.second.first.c_str());
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        else if (it.second.second == "O2") {
            double temp = o2_function(volt);
            sprintf(set_str, "/Equipment/Environment/Converted/%s", it.second.first.c_str());
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        else if (it.second.second == "Water") {
            double temp = water_function(volt);
            sprintf(set_str, "/Equipment/Environment/Converted/%s", it.second.first.c_str());
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        else if (it.second.second == "PCMini52T") {
            double temp = pcmini52_t_function(volt);
            sprintf(set_str, "/Equipment/Environment/Converted/%s", it.second.first.c_str());
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
        else if (it.second.second == "PCMini52RH") {
            double rh = pcmini52_rh_function(volt);
            sprintf(set_str, "/Equipment/Environment/Converted/%s", it.second.first.c_str());
            db_set_value(hDB, 0, set_str, &rh, sizeof(double), 1, TID_DOUBLE);
        }
    }

    char get_str[256];
    double temp = 30;
    int size_temp= sizeof(double);
    sprintf(get_str, "/Equipment/Environment/Converted/%s", dew_point_temp.c_str());
    int ok_temp = db_get_value(hDB, 0, get_str, &temp, &size_temp, TID_DOUBLE, FALSE);
    double rh = 40;
    int size_rh = sizeof(double);
    sprintf(get_str, "/Equipment/Environment/Converted/%s", dew_point_rh.c_str());
    int ok_temp2 = db_get_value(hDB, 0, get_str, &rh, &size_rh, TID_DOUBLE, FALSE);
    char set_str[256];
    double dewpoint = dewpoint_function(temp, rh);
    sprintf(set_str, "/Equipment/Environment/Computed/DewPoint");
    db_set_value(hDB, 0, set_str, &dewpoint, sizeof(double), 1, TID_DOUBLE);

    float volt1;
    int size_volt1 = sizeof(float);
    sprintf(get_str, "/Equipment/EnvPixels/Variables/Input[44]");
    int ok_volt1 = db_get_value(hDB, 0, get_str, &volt1, &size_volt1, TID_FLOAT, FALSE);
    float volt2;
    int size_volt2 = sizeof(float);
    sprintf(get_str, "/Equipment/EnvPixels/Variables/Input[45]");
    int ok_volt2 = db_get_value(hDB, 0, get_str, &volt2, &size_volt2, TID_FLOAT, FALSE);
    float volt3;
    int size_volt3 = sizeof(float);
    sprintf(get_str, "/Equipment/EnvPixels/Variables/Input[46]");
    int ok_volt3 = db_get_value(hDB, 0, get_str, &volt3, &size_volt3, TID_FLOAT, FALSE);
    float volt4;
    int size_volt4 = sizeof(float);
    sprintf(get_str, "/Equipment/EnvPixels/Variables/Input[47]");
    int ok_volt4 = db_get_value(hDB, 0, get_str, &volt4, &size_volt4, TID_FLOAT, FALSE);

    double magfield = magnetic_field_function(volt1, volt2, volt3, volt4);
    sprintf(set_str, "/Equipment/EnvPixels/ConvertedMagnet/B_Tesla");
    db_set_value(hDB, 0, set_str, &magfield, sizeof(double), 1, TID_DOUBLE);
    double vhp = vhp_function(volt1, volt2);
    sprintf(set_str, "/Equipment/EnvPixels/ConvertedMagnet/Vhp_mV");
    db_set_value(hDB, 0, set_str, &vhp, sizeof(double), 1, TID_DOUBLE);
    double vhp_scaled = vhp_scaled_funtion(volt1, volt2, volt3, volt4);
    sprintf(set_str, "/Equipment/EnvPixels/ConvertedMagnet/Vhp_scaled_mV");
    db_set_value(hDB, 0, set_str, &vhp_scaled, sizeof(double), 1, TID_DOUBLE);
    double ihp = ihp_function(volt3, volt4);
    sprintf(set_str, "/Equipment/EnvPixels/ConvertedMagnet/Ihp_mA");
    db_set_value(hDB, 0, set_str, &ihp, sizeof(double), 1, TID_DOUBLE);

    return CM_SUCCESS;
}

/*------------------------------------------------------------------*/
