
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
    sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Temperature/Scale");
    int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
    double expo;
    sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Temperature/Exp");
    int ok_expo = db_get_value(hDB, 0, get_str, &expo, &sizze, TID_DOUBLE, FALSE);
    double offset;
    sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Temperature/Offset");
    int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);

    return (scale * exp(-1*expo * (10000 * (volt / (5 - volt)))) - offset);
}
double convert_pressure(float volt) {
    char get_str[256];
    double scale;
    int sizze = sizeof(double);
    sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Pressure/Scale");
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
    sprintf(get_str, "/Equipment/Pixel-SC-Temperature/Converted/Parameters/Temperature/Scale");
    int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
    double offset;
    sprintf(get_str, "/Equipment/Pixel-SC-Temperature/Converted/Parameters/Temperature/Offset");
    int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);

    return offset + scale * volt;
}

std::map<int, std::pair<std::string, std::string> > extra_env = {
    {0, {"Env_US1 - Water", "undefined"}},
    {1, {"Env_US1 - PCMini52_RH", "undefined"} },
    {2, {"Env_US1 - PCMini52_T", "undefined" }},
    {3, {"Env_US2 - O2-7-top", "undefined" }},
    {4, {"Env_US2 - HIH4040_top", "undefined" }},
    {5, {"Env_US2 - LM35_Fibre", "undefined" }},
    {6, {"Env_US2 - LM35_FEC1", "undefined" }},
    {7, {"Env_US2 - LM35_top", "undefined" }},

    {9, {"Env_DS1 - Water", "undefined" }},
    {11, {"Env_DS2 - O2-4-top", "undefined" }},
    {12, {"Env_DS2 - HIH4040_top", "undefined" }},
    {13, {"Env_DS2 - LM35_Fibre", "undefined" }},
    {14, {"Env_DS2 - LM35_FEC1", "undefined" }},
    {15, {"Env_DS2 - LM35_top", "undefined" }},

    {18, {"Env_DS3 - O2-11-bottom", "undefined" }},
    {19, {"Env_US3 - O2-6-central", "undefined" }},
    {20, {"Env_US3 - HIH4040_bottom", "undefined" }},
    {21, {"Env_US3 - HIH4040_central", "undefined" }},
    //{23, {"Env_US6 - O2_3", "undefined" }},
    //{24, {"Env_US6 - LM35_3", "undefined" }},
    //{25, {"Env_US6 - HIH_3", "undefined" }},

    {27, {"Env_DS3 - O2-11-bottom", "undefined" }},
    {28, {"Env_DS3 - O2-6-central", "undefined" }},
    {29, {"Env_DS3 - HIH4040_bottom", "undefined" }},
    {30, {"Env_DS3 - HIH4040_central", "undefined" }}
    //{32, {"Env_DS6 - LM35_6", "undefined" }},
    //{33, {"Env_DS6 - HIH_6", "undefined" }},
    //{34, {"Env_DS6 - O2_6", "undefined" }}
};

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
    
    {"Pixel-SC-Temperature",              /* equipment name */
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
            sprintf(set_str, "/Equipment/%s/Converted/Parameters/Temperature/Scale", equipment);
            double scale_tmp;
            int size_scale_tmp = sizeof(double);
            int has_subname = db_get_value(hDB, 0, set_str, &scale_tmp, &size_scale_tmp, TID_DOUBLE, FALSE);
            if (has_subname != DB_SUCCESS) {
                double scale = 79.5;
                db_set_value(hDB, 0, set_str, &scale, sizeof(double), 1, TID_DOUBLE);
                sprintf(set_str, "/Equipment/%s/Converted/Parameters/Temperature/Exp", equipment);
                double expo = 0.00011;
                db_set_value(hDB, 0, set_str, &expo, sizeof(double), 1, TID_DOUBLE);
                sprintf(set_str, "/Equipment/%s/Converted/Parameters/Temperature/Offset", equipment);
                double offset = 1.76;
                db_set_value(hDB, 0, set_str, &offset, sizeof(double), 1, TID_DOUBLE);

                sprintf(set_str, "/Equipment/%s/Converted/Parameters/Pressure/Scale", equipment);
                double scalep = 2.25;
                db_set_value(hDB, 0, set_str, &scalep, sizeof(double), 1, TID_DOUBLE);
            }
        }
        else if (chn_index == 45 || chn_index == 47 || chn_index == 49 || chn_index == 51) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Converted/Temperature", equipment);
            double temp = 30;
            int idx = (int)((chn_index - 45) / 2);

            db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), idx, TID_DOUBLE, TRUE);

            //Alarms
            char al_str[256];
            sprintf(al_str, "SSF_Cooling_TempHigh_Alarm_%d", idx);
            char al_cond[256];
            sprintf(al_cond, "/Equipment/%s/Converted/Temperature[%d] > 30", equipment, idx);
            char al_message[256];
            sprintf(al_message, "Temperature too high for SSF chiller (Box %d)", idx+1);
            al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);

            char al_strw[256];
            sprintf(al_strw, "SSF_Cooling_TempHigh_Warning_%d", idx);
            char al_condw[256];
            sprintf(al_condw, "/Equipment/%s/Converted/Temperature[%d] > 25", equipment, idx);
            char al_messagew[256];
            sprintf(al_messagew, "Temperature quite high for SSF chiller (Box %d)", idx+1);
            al_define_odb_alarm(al_strw, al_condw, "Warning", al_messagew);

            sprintf(al_str, "SSF_Cooling_TempLow_Alarm_%d", idx);
            sprintf(al_cond, "/Equipment/%s/Converted/Temperature[%d] < 10", equipment, idx);
            sprintf(al_message, "Temperature too low for SSF chiller (Box %d)", idx + 1);
            al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);

            sprintf(al_strw, "SSF_Cooling_TempLow_Warning_%d", idx);
            sprintf(al_condw, "/Equipment/%s/Converted/Temperature[%d] < 15", equipment, idx);
            sprintf(al_messagew, "Temperature quite low for SSF chiller (Box %d)", idx + 1);
            al_define_odb_alarm(al_strw, al_condw, "Warning", al_messagew);
        }
        else if (chn_index == 46 || chn_index == 48 || chn_index == 50 || chn_index == 52) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Converted/Pressure", equipment);
            double temp = 1;
            int idx = (int)((chn_index - 46) / 2);
            db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), idx, TID_DOUBLE, TRUE);
        }
        else if (chn_index == 42 || chn_index == 43) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Converted/Flow", equipment);
            double temp = 1;
            int idx = (int)((chn_index - 42));
            db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), idx, TID_DOUBLE, TRUE);
        }
        else if (extra_env.find(chn_index) != extra_env.end()) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Converted/%s", equipment, extra_env[chn_index].first.c_str());
            double temp = 1.;
            db_set_value(hDB, 0, set_str, &temp, sizeof(double), 1, TID_DOUBLE);
        }
    }
    else if (equipment == "Pixel-SC-Temperature") {
        if (chn_index == 0) {
            char set_str[256];
            sprintf(set_str, "/Equipment/%s/Converted/Parameters/Temperature/Scale", equipment);
            double scale_tmp;
            int size_scale_tmp = sizeof(double);
            int has_subname = db_get_value(hDB, 0, set_str, &scale_tmp, &size_scale_tmp, TID_DOUBLE, FALSE);
            if (has_subname != DB_SUCCESS) {
                double scale = -331.8;
                db_set_value(hDB, 0, set_str, &scale, sizeof(double), 1, TID_DOUBLE);
                sprintf(set_str, "/Equipment/%s/Converted/Parameters/Temperature/Offset", equipment);
                double offset = 228.;
                db_set_value(hDB, 0, set_str, &offset, sizeof(double), 1, TID_DOUBLE);
            }
        }
        char set_str[256];
        sprintf(set_str, "/Equipment/%s/Converted/Temperature", equipment);
        double temp = 0;
        int idx = chn_index;
        db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), idx, TID_DOUBLE, TRUE);
        if (chn_index != 2
            && chn_index != 34 && chn_index != 35 && chn_index != 36 && chn_index != 37 && chn_index != 38
            && chn_index < 42) {
            //Alarms
            char al_str[256];
            sprintf(al_str, "Pixel_Temperature_Alarm_%d", chn_index);
            char al_cond[256];
            sprintf(al_cond, "/Equipment/%s/Converted/Temperature[%d] > 70", equipment, chn_index);
            char al_message[256];
            sprintf(al_message, "Temperature too high for pixel (MSCB channel %d)", chn_index);
            al_define_odb_alarm(al_str, al_cond, "Alarm", al_message);

            char al_strw[256];
            sprintf(al_strw, "Pixel_Temperature_Warning_%d", chn_index);
            char al_condw[256];
            sprintf(al_condw, "/Equipment/%s/Converted/Temperature[%d] > 60", equipment, chn_index);
            char al_messagew[256];
            sprintf(al_messagew, "Temperature quite high for pixel (MSCB channel %d)", chn_index);
            al_define_odb_alarm(al_strw, al_condw, "Warning", al_messagew);
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
    printf("Initializing Pixel-SC-Temperature...\n");
    for (unsigned int j = 0; j < 48; j++)
        //mscb_define("mscb334.mu3e", "Pixel-SC-Temperature", "ADC", multi_driver_pix, 0, j, NULL, 0.0005);//Default submaster (deprecated)
        mscb_define("", "Pixel-SC-Temperature", "ADC", multi_driver_pix, 0, j, NULL, 0.0005);

    std::vector<std::string> histo_vars;
    for (int cha = 0; cha < 42; ++cha) {
        if (cha != 2
            && cha != 34 && cha != 35 && cha != 36 && cha != 37 && cha != 38
            && cha < 42) {
            char variab[256];
            int port = (int)(cha / 8);
            int in_ch = cha % 8;
            sprintf(variab, "Pixel-SC-Temperature:P%dUIn%d", port, in_ch);
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
            sprintf(get_str, "/Equipment/Pixel-SC-Temperature/Converted/Parameters/Temperature/Scale");
            int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
            double offset;
            sprintf(get_str, "/Equipment/Pixel-SC-Temperature/Converted/Parameters/Temperature/Offset");
            int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);
            char formula[256];
            sprintf(formula, "%.3f*x + %.3f", scale, offset);
            db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), cha, TID_STRING, TRUE);
        }
    }

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
        sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Temperature/Scale");
        int ok_scale = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
        double expo;
        sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Temperature/Exp");
        int ok_expo = db_get_value(hDB, 0, get_str, &expo, &sizze, TID_DOUBLE, FALSE);
        double offset;
        sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Temperature/Offset");
        int ok_offset = db_get_value(hDB, 0, get_str, &offset, &sizze, TID_DOUBLE, FALSE);
        char formula[256];
        sprintf(formula, "%.1f*Math.exp(-%.5f*(10000*(x/(5-x))))-%.2f", scale, expo, offset);
        db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), cha, TID_STRING, TRUE);

        sprintf(variab, "/History/Display/Environment/SSF-pressures/Label");
        sprintf(vvalue, "Box %d", cha + 1);
        db_set_value_index(hDB, 0, variab, &vvalue, sizeof(vvalue), cha, TID_STRING, TRUE);
        sprintf(variab, "/History/Display/Environment/SSF-pressures/Formula");
        sprintf(get_str, "/Equipment/Environment/Converted/Parameters/Pressure/Scale");
        int ok_scale2 = db_get_value(hDB, 0, get_str, &scale, &sizze, TID_DOUBLE, FALSE);
        sprintf(formula, "%.3f * x", scale);
        db_set_value_index(hDB, 0, variab, &formula, sizeof(formula), cha, TID_STRING, TRUE);
    }


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
    ss_sleep(100);
    for (int ch = 0; ch < 4; ++ch) {
        char get_str[256];
        float volt;
        int size_volt = sizeof(float);
        sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", 45 + ch*2);
        int ok_temp = db_get_value(hDB, 0, get_str, &volt, &size_volt, TID_FLOAT, FALSE);
        if (ok_temp == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/Environment/Converted/Temperature");
            double temp = convert_temperature(volt);
            db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), ch, TID_DOUBLE, TRUE);
        }
        float volt2;
        sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", 46 + ch * 2);
        int ok_press = db_get_value(hDB, 0, get_str, &volt2, &size_volt, TID_FLOAT, FALSE);
        if (ok_press == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/Environment/Converted/Pressure");
            double temp = convert_pressure(volt2);
            db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), ch, TID_DOUBLE, TRUE);
        }
        if (ch > 1)
            continue;
        float volt3;
        sprintf(get_str, "/Equipment/Environment/Variables/Input[%d]", 42 + ch);
        int ok_flow = db_get_value(hDB, 0, get_str, &volt3, &size_volt, TID_FLOAT, FALSE);
        if (ok_flow == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/Environment/Converted/Flow");
            double temp = convert_flow(volt3);
            db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), ch, TID_DOUBLE, TRUE);
        }
    }

    for (int chp = 0; chp < 42; ++chp) {
        if (chp == 2 ||
            chp == 34 || chp == 35 || chp == 36 || chp == 37 || chp == 38)
            continue;
        char get_str[256];
        float volt;
        int size_volt = sizeof(float);
        sprintf(get_str, "/Equipment/Pixel-SC-Temperature/Variables/Input[%d]", chp);
        int ok_temp = db_get_value(hDB, 0, get_str, &volt, &size_volt, TID_FLOAT, FALSE);
        if (ok_temp == DB_SUCCESS) {
            char set_str[256];
            sprintf(set_str, "/Equipment/Pixel-SC-Temperature/Converted/Temperature");
            double temp = convert_temperature_pixels(volt);
            db_set_value_index(hDB, 0, set_str, &temp, sizeof(double), chp, TID_DOUBLE, TRUE);
        }
    }

    return CM_SUCCESS;
}

/*------------------------------------------------------------------*/
