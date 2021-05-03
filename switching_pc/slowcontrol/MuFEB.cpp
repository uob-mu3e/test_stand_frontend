/********************************************************************\

  Name:         MuFEB.h
  Created by:   Konrad Briggl

Contents:       Definition of common functions to talk to a FEB. In particular common readout methods for status events and methods for slow control mapping are implemented here.

\********************************************************************/

#include "MuFEB.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h" //for set_equipment_status

#include "mudaq_device.h"
#include "asic_config_base.h"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <thread>
#include <chrono>

using std::cout;
using std::endl;

//status flags from FEB
#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1

using midas::odb;

int MuFEB::WriteFEBID(){
    for(auto FEB: febs){
       if(!FEB.IsScEnabled()) continue; //skip disabled fibers
       if(FEB.SB_Number()!=SB_number) continue; //skip commands not for this SB
       uint32_t val=0xFEB1FEB0; // TODO: Where does this hard-coded value come from?
       val+=(FEB.GetLinkID()<<16)+FEB.GetLinkID();

       char reportStr[255];
       sprintf(reportStr,"Setting FEBID of %s: Link%u, SB%u.%u to (%4.4x)-%4.4x",
             FEB.GetLinkName().c_str(),FEB.GetLinkID(),
             FEB.SB_Number(),FEB.SB_Port(),(val>>16)&0xffff,val&0xffff);
       cm_msg(MINFO,"MuFEB::WriteFEBID",reportStr);
       feb_sc.FEB_register_write(FEB.SB_Port(),  FPGA_ID_REGISTER_RW, val);
    }
    return 0;
}

void MuFEB::ReadFirmwareVersionsToODB()
{
    vector<uint32_t> arria(1);
    vector<uint32_t> max(1);

    odb arriaversions("/Equipment/Switching/Variables/FEBFirmware/Arria V Firmware Version");
    odb maxversions("/Equipment/Switching/Variables/FEBFirmware/Max 10 Firmware Version");
    odb febversions("/Equipment/Switching/Variables/FEBFirmware/FEB Version");

    for(auto FEB: febs){
        if(!FEB.IsScEnabled()) continue; //skip disabled fibers
        if(FEB.SB_Number()!=SB_number) continue; //skip commands not for this SB
         if(feb_sc.FEB_register_read(FEB.SB_Port(), GIT_HASH_REGISTER_R, arria) != FEBSlowcontrolInterface::ERRCODES::OK)
            cm_msg(MINFO,"MuFEB::ReadFirmwareVersionsToODB", "Failed to read Arria firmware version");
         else
            arriaversions[FEB.GetLinkID()] = arria[0];
         if(feb_sc.FEB_register_read(FEB.SB_Port(), MAX10_VERSION_REGISTER_R, max) != FEBSlowcontrolInterface::ERRCODES::OK)
            cm_msg(MINFO,"MuFEB::ReadFirmwareVersionsToODB", "Failed to read Max firmware version");
         else
            maxversions[FEB.GetLinkID()] = max[0];

         if(feb_sc.FEB_register_read(FEB.SB_Port(), MAX10_STATUS_REGISTER_R, max) != FEBSlowcontrolInterface::ERRCODES::OK)
            cm_msg(MINFO,"MuFEB::ReadFirmwareVersionsToODB", "Failed to read Max status register");
         else{
             // TODO: Handle this properly
             if(GET_MAX10_STATUS_BIT_SPI_ARRIA_CLK(max[0]))
                febversions[FEB.GetLinkID()] = 20;
             else
                febversions[FEB.GetLinkID()] = 20;
         }
    }
}

void MuFEB::LoadFirmware(std::string filename, uint16_t FPGA_ID)
{

    auto FEB = febs[FPGA_ID];

    FILE * f = fopen(filename.c_str(), "rb");
    if(!f){
       cm_msg(MERROR,"MuFEB::LoadFirmware", "Failed to open %s", filename.c_str());
       return;
    }
    // Get the file size
    fseek (f , 0 , SEEK_END);
    long fsize = ftell(f);
    rewind (f);

    cm_msg(MINFO,"MuFEB::LoadFirmware", "Programming %s of size %ld", filename.c_str(), fsize);
    printf("Programming %s of size %ld\n",filename.c_str(), fsize);

    //clear the FIFO
    feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_CTRL_W,2);
    feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_CTRL_W,0);

    feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_CTRL_W,1);


    long pos =0;
    uint32_t addr=0;
    while(pos*4 < fsize){
        uint32_t buffer[256];
        fread(buffer,sizeof(uint32_t),256, f);
        vector<uint32_t> data(buffer, buffer+256);
        feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_DATA_W,data,true);

        for(int i=0; i < 4; i++){
            feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_ADDR_W,addr);            
            uint32_t readback = 2;
            uint32_t count = 0;
            uint32_t limit = 1e6;
            if((addr & 0xFFFF) == 0)
                limit = 1e5;
            while(readback & 0x2 && count < limit){
                int ec = feb_sc.FEB_register_read(FEB.SB_Port(),PROGRAMMING_STATUS_R,readback);
                if(ec != FEBSlowcontrolInterface::ERRCODES::OK){
                    printf("Error reading back!\n");
                    feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_CTRL_W,0);
                    return;
                }
                count++;
                usleep(100);
            }
            addr += 256;
            if(count == limit){
                printf("Timeout\n");
                feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_CTRL_W,0);
                return;
            }
        }
        pos  += 256;
        if(pos%(4096)==0)
            printf("Loaded %f of file\n", (double)pos*4/fsize);

    }

    feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_CTRL_W,0);

    feb_sc.FEB_register_write(FEB.SB_Port(),PROGRAMMING_CTRL_W,0);

    cm_msg(MINFO,"MuFEB::LoadFirmware", "Done programming");

    fclose(f);
}

int MuFEB::ReadBackRunState(uint16_t FPGA_ID){
    //map to SB fiber
    auto FEB = febs[FPGA_ID];

    //skip disabled fibers
    if(!FEB.IsScEnabled())
        return SUCCESS;

    //skip commands not for this SB
    if(FEB.SB_Number()!=SB_number)
        return SUCCESS;

   vector<uint32_t> val(2);
   char set_str[255];
   int status = feb_sc.FEB_register_read(FEB.SB_Port(), RUN_STATE_RESET_BYPASS_REGISTER_RW , val);
   if(status!=FEBSlowcontrolInterface::ERRCODES::OK) return status;

   //val[0] is reset_bypass register
   //val[1] is reset_bypass payload
   char path[255];

   BOOL bypass_enabled=true;
   if(((val[0])&0x1ff)==0x000) bypass_enabled=false;
    // set odb value_index index = FPGA_ID, value = bypass_enabled
        
    sprintf(path, "%s/Variables", odb_prefix);
    odb variables_feb_run_state((std::string)path);

    sprintf(set_str, "Bypass enabled %d", FPGA_ID);
    variables_feb_run_state[set_str] = bypass_enabled;

    // set odb value_index index = FPGA_ID, value = value
    DWORD value=(val[0]>>16) & 0x3ff;
    sprintf(set_str, "Run state %d", FPGA_ID);
    variables_feb_run_state[set_str] = value;

   return SUCCESS;
}

uint32_t MuFEB::ReadBackMergerRate(uint16_t FPGA_ID){
    auto FEB = febs[FPGA_ID];
    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!=SB_number) return SUCCESS; //skip commands not for this SB

    vector<uint32_t> mergerRate(1);
    feb_sc.FEB_register_read(FEB.SB_Port(), MERGER_RATE_REGISTER_R, mergerRate);
    return mergerRate[0];
}

uint32_t MuFEB::ReadBackResetPhase(uint16_t FPGA_ID){
    auto FEB = febs[FPGA_ID];
    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!=SB_number) return SUCCESS; //skip commands not for this SB

    vector<uint32_t> resetPhase(1);
    feb_sc.FEB_register_read(FEB.SB_Port(), RESET_PHASE_REGISTER_R, resetPhase);
    return resetPhase[0] & 0xFFFF;
}

uint32_t MuFEB::ReadBackTXReset(uint16_t FPGA_ID){
    auto FEB = febs[FPGA_ID];
    if(!FEB.IsScEnabled()) return SUCCESS; //skip disabled fibers
    if(FEB.SB_Number()!=SB_number) return SUCCESS; //skip commands not for this SB

    vector<uint32_t> TXReset(1);
    feb_sc.FEB_register_read(FEB.SB_Port(), RESET_OPTICAL_LINKS_REGISTER_RW, TXReset);
    return TXReset[0] & 0xFFFFFFFC;
}

DWORD* MuFEB::fill_SSFE(DWORD *pdata)
{
    uint32_t index = 0;

    for(auto FEB: febs){
       if(!FEB.IsScEnabled()) continue; //skip disabled fibers
       if(FEB.SB_Number()!=SB_number) continue;

       uint32_t port = FEB.SB_Port();
       // Fill in zeroes for non-existing ports
       while(index < port){
            // 26 is per_fe_SSFE_size - need to find a header for that...
            for(uint32_t j=0; j < 26; j++){
                *pdata++ = 0;
             }
           index++;
       }

       // And here we actually fill the bank
        pdata = read_SSFE_OneFEB(pdata, index, FEB.GetVersion());
        index++;
    }



    // Fill in zeroes for non-existing ports
    while(index < N_FEBS[SB_number]){
        for(uint32_t j=0; j < 26; j++){
            *pdata++ = 0;
         }
       index++;
    }

    return pdata;
}

DWORD *MuFEB::read_SSFE_OneFEB(DWORD *pdata, uint32_t index, uint32_t version)
{
    uint32_t data;
    // Start with FEB index
    *pdata++ = index;
    // Arria V temperature
    feb_sc.FEB_register_read(index, ARRIA_TEMP_REGISTER_RW, data);
    *(float*)pdata++ = ArriaVTempConversion(data);
    vector<uint32_t> adcdata(5);
    // Read the MAX10 ADC
    feb_sc.FEB_register_read(index, MAX10_ADC_0_1_REGISTER_R, adcdata);
    // What the data here mean depends on the FEB version
    if(version == 20){
        *(float*)pdata++ = Max10TempConversion(adcdata[4]>>16);  // Max temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[4] & 0xFFFF);  // Si1 temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[3]>>16); //Si2 temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[3] & 0xFFFF); // Arria temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[2]>>16); //DCDC temperature
        *(float*)pdata++ = Max10VoltageConversion(adcdata[2] & 0xFFFF); // 1.1V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[1] >> 16);    // 1.8V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[1] & 0xFFFF); // 2.5V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[0] >> 16);    // 3.3V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[0] & 0xFFFF,10.0f); // 20V
    } else {
        *(float*)pdata++ = Max10TempConversion(adcdata[4]>>16);  // Max temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[3]>>16);  // Si1 temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[3] & 0xFFFF); //Si2 temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[0]>>16); // Arria temperature
        *(float*)pdata++ = Max10ExternalTemeperatureConversion(adcdata[2]>>16); //DCDC temperature
        *(float*)pdata++ = Max10VoltageConversion(adcdata[2] & 0xFFFF); // 1.1V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[1] >> 16);    // 1.8V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[1] & 0xFFFF,3.0/2.0); // 2.5V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[4] & 0xFFFF, 2.0f);    // 3.3V
        *(float*)pdata++ = Max10VoltageConversion(adcdata[0] & 0xFFFF,10.0f); // 20V
    }
    // Conversion factors: See Firefly B04 datasheet, page 62
    vector<uint32_t> fireflydata(14);
    feb_sc.FEB_register_read(index, FIREFLY1_TEMP_REGISTER_R, fireflydata);
    *(float*)pdata++ = (float)(int8_t)fireflydata[0]; // FF1 Temperature
    *(float*)pdata++ = ((float)fireflydata[1])/1E4f; // FF1 Voltage
    *(float*)pdata++ = ((float)fireflydata[2])/1E7f; // FF1 RX1 Power
    *(float*)pdata++ = ((float)fireflydata[3])/1E7f; // FF1 RX2 Power
    *(float*)pdata++ = ((float)fireflydata[4])/1E7f; // FF1 RX3 Power
    *(float*)pdata++ = ((float)fireflydata[5])/1E7f; // FF1 RX4 Power
    *pdata++ = fireflydata[6]; // FF1 Alarms
    *(float*)pdata++ = (float)(int8_t)fireflydata[6]; // FF2 Temperature
    *(float*)pdata++ = ((float)fireflydata[7])/1E4f; // FF2 Voltage
    *(float*)pdata++ = ((float)fireflydata[8])/1E7f; // FF2 RX1 Power
    *(float*)pdata++ = ((float)fireflydata[9])/1E7f; // FF2 RX2 Power
    *(float*)pdata++ = ((float)fireflydata[10])/1E7f; // FF2 RX3 Power
    *(float*)pdata++ = ((float)fireflydata[11])/1E7f; // FF2 RX4 Power
    *pdata++ = fireflydata[12]; // FF2 Alarms

    return pdata;
}




//Helper functions
uint32_t MuFEB::reg_setBit  (uint32_t reg_in, uint8_t bit, bool value){
    if(value)
        return (reg_in | 1<<bit);
    else
        return (reg_in & (~(1<<bit)));
}
uint32_t MuFEB::reg_unsetBit(uint32_t reg_in, uint8_t bit){return reg_setBit(reg_in,bit,false);}

bool MuFEB::reg_getBit(uint32_t reg_in, uint8_t bit){
    return (reg_in & (1<<bit)) != 0;
}

uint32_t MuFEB::reg_getRange(uint32_t reg_in, uint8_t length, uint8_t offset){
    return (reg_in>>offset) & ((1<<length)-1);
}
uint32_t MuFEB::reg_setRange(uint32_t reg_in, uint8_t length, uint8_t offset, uint32_t value){
    return (reg_in & ~(((1<<length)-1)<<offset)) | ((value & ((1<<length)-1))<<offset);
}



float MuFEB::ArriaVTempConversion(uint32_t reg)
{
    // The magic numbers here come from the Intel temperature sensor user guide UG-01074, page 11
    vector<uint32_t> steps = {0x3A, 0x4E, 0x62, 0x6C, 0x76, 0x80, 0x8A, 0x9E, 0xB2, 0xD0, 0xD5, 0xE4, 0xFF};
    vector<float> temps    = { -70,  -50,  -30,  -20,  -10,    0,   10,   30,   50,   80,   85,  100,  127};

    if(reg < steps[0])
        return temps[0];
    if(reg > steps[steps.size()-1])
        return temps[steps.size()-1];

    for(int i = 1; i < steps.size()-1; i++){
        if(reg < steps[i])
            return temps[i] + (temps[i+1]-temps[i]) * ((int)reg -  (int)steps[i])/(steps[i+1]-steps[i]);
    }

    return -666; // should not get here
}

float MuFEB::Max10TempConversion(uint32_t reg)
{
    if(reg < 3431)
        return 125.0f;
    if(reg >3798)
        return -40.0f;
    auto it = std::upper_bound(maxadcvals.begin(),maxadcvals.end(),reg);
    size_t index = std::distance(maxadcvals.begin(), it);
    return maxtempvals[index];
}

float MuFEB::Max10VoltageConversion(uint16_t reg, float divider)
{
    // We have a 12 bit ADC with 2.5 V refernce voltage:
    return (double)reg * (2.5/4096.0)*divider;
}

float MuFEB::Max10ExternalTemeperatureConversion(uint16_t reg)
{
    // We have a 12 bit ADC with 2.5 V refernce voltage:
    double vout = (double)reg * (2.5/4096.0);
    // For the conversion to temperature we assume, we are in -40 to +100,
    // See TI TMP235 data sheet, page 8: Voffs = 500 mV, Tc = 10 mV/C, Tinfl = 0 C
    return (vout - 0.5)/0.01 + 0;
}


// The following is from the MAX10 ADC UG

const vector<uint32_t> MuFEB::maxadcvals ={3431,
                                           3432,
                                           3440,
                                           3445,
                                           3449,
                                           3450,
                                           3451,
                                           3456,
                                           3459,
                                           3460,
                                           3461,
                                           3465,
                                           3468,
                                           3471,
                                           3474,
                                           3477,
                                           3480,
                                           3483,
                                           3486,
                                           3489,
                                           3490,
                                           3492,
                                           3494,
                                           3496,
                                           3498,
                                           3500,
                                           3501,
                                           3504,
                                           3507,
                                           3510,
                                           3513,
                                           3516,
                                           3519,
                                           3522,
                                           3524,
                                           3525,
                                           3526,
                                           3530,
                                           3534,
                                           3538,
                                           3542,
                                           3546,
                                           3547,
                                           3548,
                                           3549,
                                           3550,
                                           3551,
                                           3552,
                                           3555,
                                           3558,
                                           3561,
                                           3564,
                                           3567,
                                           3570,
                                           3573,
                                           3576,
                                           3579,
                                           3582,
                                           3585,
                                           3589,
                                           3590,
                                           3591,
                                           3592,
                                           3593,
                                           3594,
                                           3595,
                                           3598,
                                           3601,
                                           3604,
                                           3607,
                                           3610,
                                           3613,
                                           3616,
                                           3619,
                                           3622,
                                           3625,
                                           3628,
                                           3630,
                                           3632,
                                           3634,
                                           3636,
                                           3638,
                                           3640,
                                           3641,
                                           3642,
                                           3643,
                                           3645,
                                           3648,
                                           3651,
                                           3654,
                                           3656,
                                           3658,
                                           3660,
                                           3662,
                                           3664,
                                           3666,
                                           3667,
                                           3670,
                                           3673,
                                           3676,
                                           3677,
                                           3678,
                                           3680,
                                           3682,
                                           3684,
                                           3688,
                                           3695,
                                           3696,
                                           3697,
                                           3698,
                                           3699,
                                           3700,
                                           3702,
                                           3703,
                                           3704,
                                           3707,
                                           3709,
                                           3711,
                                           3713,
                                           3715,
                                           3717,
                                           3719,
                                           3720,
                                           3721,
                                           3725,
                                           3727,
                                           3730,
                                           3731,
                                           3732,
                                           3733,
                                           3736,
                                           3738,
                                           3740,
                                           3742,
                                           3744,
                                           3746,
                                           3748,
                                           3750,
                                           3751,
                                           3752,
                                           3754,
                                           3756,
                                           3759,
                                           3762,
                                           3764,
                                           3765,
                                           3766,
                                           3768,
                                           3770,
                                           3771,
                                           3773,
                                           3775,
                                           3777,
                                           3779,
                                           3780,
                                           3781,
                                           3782,
                                           3785,
                                           3786,
                                           3788,
                                           3790,
                                           3792,
                                           3793,
                                           3795,
                                           3796,
                                           3798,
                                           99999};
const vector<float> MuFEB::maxtempvals ={125,
                                         124,
                                         123,
                                         122,
                                         121,
                                         120,
                                         119,
                                         118,
                                         117,
                                         116,
                                         115,
                                         114,
                                         113,
                                         112,
                                         111,
                                         110,
                                         109,
                                         108,
                                         107,
                                         106,
                                         105,
                                         104,
                                         103,
                                         102,
                                         101,
                                         100,
                                         99,
                                         98,
                                         97,
                                         96,
                                         95,
                                         94,
                                         93,
                                         92,
                                         91,
                                         90,
                                         89,
                                         88,
                                         87,
                                         86,
                                         85,
                                         84,
                                         83,
                                         82,
                                         81,
                                         80,
                                         79,
                                         78,
                                         77,
                                         76,
                                         75,
                                         74,
                                         73,
                                         72,
                                         71,
                                         70,
                                         69,
                                         68,
                                         67,
                                         66,
                                         65,
                                         64,
                                         63,
                                         62,
                                         61,
                                         60,
                                         59,
                                         58,
                                         57,
                                         56,
                                         55,
                                         54,
                                         53,
                                         52,
                                         51,
                                         50,
                                         49,
                                         48,
                                         47,
                                         46,
                                         45,
                                         44,
                                         43,
                                         42,
                                         41,
                                         40,
                                         39,
                                         38,
                                         37,
                                         36,
                                         35,
                                         34,
                                         33,
                                         32,
                                         31,
                                         30,
                                         29,
                                         28,
                                         27,
                                         26,
                                         25,
                                         24,
                                         23,
                                         22,
                                         21,
                                         20,
                                         19,
                                         18,
                                         17,
                                         16,
                                         15,
                                         14,
                                         13,
                                         12,
                                         11,
                                         10,
                                         9,
                                         8,
                                         7,
                                         6,
                                         5,
                                         4,
                                         3,
                                         2,
                                         1,
                                         0,
                                         -1,
                                         -2,
                                         -3,
                                         -4,
                                         -5,
                                         -6,
                                         -7,
                                         -8,
                                         -9,
                                         -10,
                                         -11,
                                         -12,
                                         -13,
                                         -14,
                                         -15,
                                         -16,
                                         -17,
                                         -18,
                                         -19,
                                         -20,
                                         -21,
                                         -22,
                                         -23,
                                         -24,
                                         -25,
                                         -26,
                                         -27,
                                         -28,
                                         -29,
                                         -30,
                                         -31,
                                         -32,
                                         -33,
                                         -34,
                                         -35,
                                         -36,
                                         -37,
                                         -38,
                                         -39,
                                         -40,
                                         -40};

