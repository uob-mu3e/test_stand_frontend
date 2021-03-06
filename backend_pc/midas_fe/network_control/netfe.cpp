// Midas frontend for the DHCP and DNS server on the mu3e gateway

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <cassert>
#include "midas.h"
#include "odbxx.h"
#include "mfe.h"

using namespace std;
using midas::odb;

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "DHCP DNS";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* Overwrite equipment struct in ODB from values in code*/
BOOL equipment_common_overwrite = FALSE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 1000;

/* maximum event size produced by this frontend */
INT max_event_size = 10000*2;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 10240*2;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000*2;


vector <string> ips;
vector <string> requestedHostnames;
vector <string> macs;
vector <string> expiration;

vector <string> prev_ips;
vector <string> prev_requestedHostnames;
vector <string> active_ips;
vector <string> unknown_ips;

vector <string> reserved_mac_addr;
vector <string> reserved_ips;
vector <string> reserved_hostnames;



/*-- Function declarations -----------------------------------------*/

INT read_cr_event(char *pevent, INT off);
void netfe_settings_changed(HNDLE, HNDLE, int, void *);

/*-- Equipment list ------------------------------------------------*/

/* Default values for /Equipment/Clock Reset/Settings */
const char *cr_settings_str[] = {
"DHCPD Active = BOOL : 1",
"DNS Active = BOOL : 1",
"usercmdReserve = BOOL : 0",
"usercmdRmReserve = BOOL : 0",
"usereditReserveIP = STRING : [32] 000.000.000.000",
"usereditReserveMAC = STRING : [32] 00:00:00:00:00",
"usereditReserveHost = STRING : [32] hostname",
"usereditRemReserveIP = STRING : [32] 000.000.000.000",
nullptr
};

EQUIPMENT equipment[] = {

   {"DHCP DNS",              /* equipment name */
    {102, 0,                     /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,               /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     10000,                     /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     0,                         /* log history every event */
     "", "", ""} ,
    read_cr_event,              /* readout routine */
   },

   {""}
};


/*-- read leased ip addresses from dhcp ------------------------------------------------*/
void read_leases(string path, vector <string> *ips, vector <string> *requestedHostnames,vector <string> *expiration, vector <string> *macs){
    string line;
    ifstream leases(path);
    bool hostname_requested = false;
    bool expirationfound = false;
    bool macfound = false;
    //int lease_number = -1;

    if (leases.is_open()){
        while ( getline (leases,line)){
            if(line.substr(0,5)=="lease"){
                ips->push_back(line.substr(6,line.length()-8));
                hostname_requested=false;
                expirationfound = false;
                macfound = false;

                // search for requested hostname, macaddr, etc. in this lease now:
                while(line != "}" and getline(leases,line)){
                    if(line.length()<17) continue;
                    if(line.substr(2,15)=="client-hostname"){
                        hostname_requested=true;
                        requestedHostnames->push_back(line.substr(19,line.length()-21));
                    }
                    if(line.substr(2,6)=="starts"){
                        expirationfound=true;
                        expiration->push_back(line.substr(10,line.length()-11));
                    }
                    if(line.substr(2,17)=="hardware ethernet"){
                        macs->push_back(line.substr(20,line.length()-21));
                        macfound = true;

                    }

                }

                if(!hostname_requested) requestedHostnames->push_back("-");
                if(!expirationfound) expiration->push_back("not found");
                if(!macfound) macs->push_back("-");
            }

        }
        leases.close();
    }
    else cout << "Unable to open lease file" <<endl;
}

/*-- rewrite ip/hostname table for dns server ------------------------------------------------*/
void write_dns_table(string path, vector <string> ips, vector <string> requestedHostnames){
  ofstream dnstable (path);
  int nDNS=0;
  if (dnstable.is_open())
  {
    //TODO: this needs to be changed for a different gateway server  --> read on frontend init ??
    dnstable << "$TTL 2D\n@\t\tIN SOA\t\tDHCP-214.mu3e.kph.\troot.DHCP-214.mu3e.kph. (\n\t\t\t\t2019062601\t; serial\n\t\t\t\t3H\t\t; refresh\n\t\t\t\t1H\t\t; retry\n\t\t\t\t1W\t\t; expiry\n\t\t\t\t1D )\t\t; minimum\n\nmu3e.\t\tIN NS\t\tDHCP-214.mu3e.kph.\n";

    for(size_t i = 0; i<ips.size(); i++){
        if(requestedHostnames[i]!="-"){
            dnstable<<requestedHostnames[i]<<"              IN      A       "<<ips[i]<<"\n";
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/DNSips", ips[i].c_str(), sizeof(reserved_ips[i]), nDNS, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/DNSHostnames", requestedHostnames[i].c_str(), sizeof(reserved_ips[i]), nDNS, TID_STRING, FALSE);
            nDNS++;
        }
    }
    db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/nDNS",to_string(nDNS).c_str(), sizeof(to_string(nDNS).c_str()), 1,TID_STRING);
    dnstable.close();
  }
  else cout << "Unable to write dns file"<<endl;
}

/*-- find active ips in subnet ------------------------------------------------*/
vector <string> find_active_ips(string subnet){
    vector <string> active_ips;
    string pingcmd;
    string ip;

    for(int i=2; i<255;i++){
        ip="";
        ip.append(subnet);
        ip.append(to_string(i));

        pingcmd="if (timeout 0.05 ping -c1 ";
        pingcmd.append(ip);
        pingcmd.append(" |grep \"1 received\") >/dev/null; then sleep 0.05; exit \"1\"; else exit \"0\"; fi");

        if(system(pingcmd.c_str())){
            //cout<<"found "<<ip<<endl;
            active_ips.push_back(ip);
        }
    }

    return active_ips;
}

/*-- identify unregistered fixed ips ------------------------------------------------*/
vector <string> find_unknown(vector <string> ips, vector <string> active_ips){
    vector <string> unknown_ips;
    int known;
    for(size_t i=0; i<active_ips.size();i++){
        known = 0;
        for(size_t j=0; j<ips.size();j++){
            if(ips[j]==active_ips[i])
                known = 1;
        }

        if (known != 1)
            unknown_ips.push_back(active_ips[i]);
    }
    return unknown_ips;
}

/*-- read reserved ips, a reserved fixed ip will not be returned by find_unknown()------------------------------------------*/
void read_reserved(string path, vector <string> *ips, vector <string> *hostnames, vector <string> *mac){
    string line;
    ifstream reserved(path);

    if (reserved.is_open()){
        while ( getline (reserved,line)){
            if(line.substr(0,4)=="host"){
                hostnames->push_back(line.substr(5,line.length()-7));

                // search for reserved ip, macaddr:
                while(line != "}" and getline(reserved,line)){
                    if(line.length()<17) continue;
                    if(line.substr(2,17)=="hardware ethernet"){
                        mac->push_back(line.substr(20,line.length()-22));
                    }
                    if(line.substr(2,13)=="fixed-address"){
                        ips->push_back(line.substr(16,line.length()-17));
                    }
                }
            }

        }
        reserved.close();
    }
    else cm_msg(MERROR, "read_reserved", "unable to read dhcp config");
}

void reserve_ip(){
        char ip[100];
        char mac[100];
        char hostname[100];
        int sizeip = sizeof(ip);
        int sizemac = sizeof(mac);
        int sizehostname = sizeof(hostname);

        db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditReserveIP",ip, &sizeip, TID_STRING, TRUE);
        db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditReserveMAC",mac, &sizemac, TID_STRING, TRUE);
        db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditReserveHost",hostname, &sizehostname, TID_STRING, TRUE);


        ofstream dhcpdconf ("/var/lib/dhcp/etc/dhcpd_reservations.conf", std::ios_base::app);
        if (dhcpdconf.is_open())
        {
            dhcpdconf << "host "<<hostname<<" {\n  hardware ethernet "<<mac<<";\n  fixed-address "<<ip<<";\n  ddns-hostname \""<<hostname<<"\";\n  option host-name \""<<hostname<<"\";\n}\n\n";

            dhcpdconf.close();
        }
        else cm_msg(MERROR, "reserve_ip", "unable to write dhcp config");

        system("rcdhcpd restart");
}

void rm_reserve_ip(){
    char removeip[100];
    int sizeip = sizeof(removeip);
    cm_msg(MINFO, "netfe_settings_changed", "Execute remove reserved IP");
    db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditRemReserveIP",removeip, &sizeip, TID_STRING, TRUE);
    ofstream dhcpdconf ("/var/lib/dhcp/etc/dhcpd_reservations.conf");
    int n = reserved_ips.size();
    if (dhcpdconf.is_open())
    {
        for(int i = 0; i<n; i++){
            if(reserved_ips[i]!=removeip){
                dhcpdconf << "host "<<reserved_hostnames[i]<<" {\n  hardware ethernet "<<reserved_mac_addr[i]<<";\n  fixed-address "<<reserved_ips[i]<<";\n  ddns-hostname \""<<reserved_hostnames[i]<<"\";\n  option host-name \""<<reserved_hostnames[i]<<"\";\n}\n\n";
            }
        }
        dhcpdconf.close();
    }
    else cm_msg(MERROR, "rm_reserve_ip", "unable to write dhcp config");

    system("rcdhcpd restart");
}

/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT, INT, BOOL)
{
   return 1;
}

INT interrupt_configure(INT, INT, POINTER_T)
{
   return 1;
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
    odb settings = {
        {"DHCPD Active", true},
        {"DNS Active", true},
        {"usercmdReserve", false},
        {"usercmdRmReserve", false},

        {"usereditReserveIP", "000.000.000.000"},
        {"usereditReserveMAC", "00:00:00:00:00"},
        {"usereditReserveHost", "hostname"},
        {"usereditRemReserveIP", "000.000.000.000"},

        {"leasedIPs", std::array<std::string, 255>()},
        {"leasedHostnames", std::array<std::string, 255>()},
        {"leasedMACs", std::array<std::string, 255>()},
        {"expiration", std::array<std::string, 255>()},
        {"reservedIPs", std::array<std::string, 255>()},
        {"reservedHostnames", std::array<std::string, 255>()},
        {"reservedMACs", std::array<std::string, 255>()},
        {"unknownIPs", std::array<std::string, 255>()},
        {"DNSips", std::array<std::string, 255>()},
        {"DNSHostnames", std::array<std::string, 255>()}
    };

    settings.connect("/Equipment/DHCP DNS/Settings", true);

    // add custom page to ODB
    odb custom("/Custom");
    custom["DHCP DNS&"] = "net.html";

    return CM_SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
   return CM_SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{

   return CM_SUCCESS;
}

/*-- Begin of Run --------------------------------------------------*/

INT begin_of_run(INT, char *)
{
   return CM_SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT, char *)
{
   return CM_SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(INT, char *)
{
   return CM_SUCCESS;
}

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT, char *)
{
   return CM_SUCCESS;
}

/*--- Read Clock and Reset Event to be put into data stream --------*/

INT read_cr_event(char *, INT)
{
    // slow down
    //sleep(10);
    bool cmdreserveip=false;
    int cmdreserveipsize=sizeof(TID_BOOL);

    db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usercmdReserve",&cmdreserveip, &cmdreserveipsize, TID_BOOL, TRUE);
    if(cmdreserveip==true){
         cmdreserveip = false;
         db_set_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usercmdReserve",&cmdreserveip, cmdreserveipsize, 1, TID_BOOL);
         //reserve_ip();
    }
    db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usercmdRmReserve",&cmdreserveip, &cmdreserveipsize, TID_BOOL, TRUE);
    if(cmdreserveip==true){
         cmdreserveip = false;
         db_set_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usercmdRmReserve",&cmdreserveip, cmdreserveipsize, 1, TID_BOOL);
         //rm_reserve_ip();
    }

    string subnet = "192.168.0.";  // only X.X.X. here !!!
    string dhcpd_lease_path = "/var/lib/dhcp/db/dhcpd.leases";
    string dns_zone_def_path = "/var/lib/named/master/mu3e";

    // string dhcpd_conf_path = "/etc/dhcpd.conf";
    // i dont want to edit /etc/dhcpd.conf directly
    // --> manually insert   include "/etc/dhcpd-reservations.conf";   into "/etc/dhcpd.conf";
    // and use this instead:
    string dhcpd_conf_path = "/var/lib/dhcp/etc/dhcpd_reservations.conf";

    prev_ips = ips;
    prev_requestedHostnames = requestedHostnames;
    ips.clear();
    requestedHostnames.clear();
    expiration.clear();
    macs.clear();
    reserved_ips.clear();
    reserved_hostnames.clear();
    reserved_mac_addr.clear();

    read_leases(dhcpd_lease_path, &ips, &requestedHostnames, &expiration, &macs);
    read_reserved(dhcpd_conf_path, &reserved_ips, &reserved_hostnames, &reserved_mac_addr);

    // append reserved ips tp ip list:
    ips.insert(ips.end(), reserved_ips.begin(), reserved_ips.end());
    requestedHostnames.insert(requestedHostnames.end(), reserved_hostnames.begin(), reserved_hostnames.end());
    macs.insert(macs.end(),reserved_mac_addr.begin(),reserved_mac_addr.end());
    for(size_t i = 0; i< reserved_ips.size();i++)
        expiration.push_back("inf");

    // if new dhcp request:   --> update dns table

    if(prev_ips!=ips){
        if(ips.size()!=0 && prev_ips.size()!=0 ){
            for (size_t i = 0; i < ips.size(); i++){
                if(ips.at(i)!=prev_ips.at(i)){
                    cm_msg(MINFO, "netfe_settings_changed", "new or updated dhcp lease of %s to %s",ips.at(i).c_str(),requestedHostnames.at(i).c_str());
                    break;
                }
            }
        }

        //write_dns_table(dns_zone_def_path,ips,requestedHostnames);
        //system("rcnamed restart");

        //update odb only if there was a change (prev_ips!=ips). Rewrite everything if something changed
        db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/leasedIPs", ips[0].c_str(), sizeof(ips[0]), 1, TID_STRING);
        for (size_t i = 0; i < ips.size(); i++) {
            //TODO: find a way to do this in a single command !!! (without loop of db_set_value)
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/leasedIPs", ips[i].c_str(), sizeof(ips[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/leasedHostnames", requestedHostnames[i].c_str(), sizeof(requestedHostnames[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/expiration", expiration[i].c_str(), sizeof(expiration[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/leasedMACs", macs[i].c_str(), sizeof(macs[i]), i, TID_STRING, FALSE);
        }
        db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/nLeased", to_string(ips.size()).c_str(), sizeof(to_string(ips.size()).c_str()), 1,TID_STRING);

        for (size_t i = 0; i < reserved_ips.size(); i++) {
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/reservedIPs", reserved_ips[i].c_str(), sizeof(reserved_ips[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/reservedHostnames", reserved_hostnames[i].c_str(), sizeof(reserved_hostnames[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/reservedMACs", reserved_mac_addr[i].c_str(), sizeof(reserved_mac_addr[i]), i, TID_STRING, FALSE);
        }
        db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/nReserved",to_string(reserved_ips.size()).c_str(), sizeof(to_string(reserved_ips.size()).c_str()), 1,TID_STRING);

    }

    // ping everything in subnet
    //active_ips = find_active_ips("192.168.0.");

    unknown_ips = find_unknown(ips, active_ips);
    int nUnknown= unknown_ips.size();

    if(nUnknown>0){
        //cout<<"---------------------------------------"<<endl;
        //cout<<"WARNING: found unknown fixed IPs ------"<<endl;
        //cout<<"---------------------------------------"<<endl;
        //cout<<"remove them or give them a name in dhcpd.conf:"<<endl;
        for(int i=0; i<nUnknown; i++){
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/unknownIPs", unknown_ips[i].c_str(), sizeof(unknown_ips[i]), i, TID_STRING, FALSE);
        }
         db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/nUnknown",to_string(nUnknown).c_str(), sizeof(to_string(nUnknown).c_str()), 1,TID_STRING);
    }

   return 0;
}

/*--- Called whenever settings have changed ------------------------*/


void netfe_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *)
{
   KEY key;

   db_get_key(hDB, hKey, &key);

   if (std::string(key.name) == "DHCPD Active") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if(value==true){
        cm_msg(MINFO, "netfe_settings_changed", "DHCPD activated by user");
        system("wall DHCPD activated by user");
      }
      else{
        cm_msg(MINFO, "netfe_settings_changed", "DHCPD deactivated by user");
        system("wall DHCPD deactivated by user");
      }
   }

    if (std::string(key.name) == "DNS Active") {
        BOOL value;
        int size = sizeof(value);
        db_get_data(hDB, hKey, &value, &size, TID_BOOL);
        if(value==true)
            cm_msg(MINFO, "netfe_settings_changed", "DNS activated by user");
        else
            cm_msg(MINFO, "netfe_settings_changed", "DNS deactivated by user");
   }

   if (std::string(key.name) == "usercmdReserve") {
        BOOL value;
        int size = sizeof(value);
        char ip[100];
        char mac[100];
        char hostname[100];
        //char nReservedstr[100];
        int sizeip = sizeof(ip);
        int sizemac = sizeof(mac);
        int sizehostname = sizeof(hostname);
        db_get_data(hDB, hKey, &value, &size, TID_BOOL);
        if(value){
            cm_msg(MINFO, "netfe_settings_changed", "Execute reserve IP");
            value = FALSE; // reset flag in ODB
            db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);

            db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditReserveIP",ip, &sizeip, TID_STRING, TRUE);
            db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditReserveMAC",mac, &sizemac, TID_STRING, TRUE);
            db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditReserveHost",hostname, &sizehostname, TID_STRING, TRUE);


//            ofstream dhcpdconf ("/var/lib/dhcp/etc/dhcpd_reservations.conf", std::ios_base::app);
//            if (dhcpdconf.is_open())
//            {
//                dhcpdconf << "host "<<hostname<<" {\n  hardware ethernet "<<mac<<";\n  fixed-address "<<ip<<";\n  ddns-hostname \""<<hostname<<"\";\n  option host-name \""<<hostname<<"\";\n}\n\n";

//                dhcpdconf.close();
//            }
//            else cout << "Unable to write dhcpdconf"<<endl;

//            system("rcdhcpd restart");
        }
   }

    if (std::string(key.name) == "usercmdRmReserve") {
        BOOL value;
        int size = sizeof(value);
        char removeip[100];
        int sizeip = sizeof(removeip);
        db_get_data(hDB, hKey, &value, &size, TID_BOOL);
        if (value) {
            cm_msg(MINFO, "netfe_settings_changed", "Execute remove reserved IP");
            value = FALSE; // reset flag in ODB
            db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
            db_get_value(hDB, 0, "/Equipment/DHCP DNS/Settings/usereditRemReserveIP",removeip, &sizeip, TID_STRING, TRUE);

//            ofstream dhcpdconf ("/var/lib/dhcp/etc/dhcpd_reservations.conf");
//            int n = reserved_ips.size();
//            if (dhcpdconf.is_open())
//            {
//                for(int i = 0; i<n; i++){
//                    if(reserved_ips[i]!=removeip){
//                        dhcpdconf << "host "<<reserved_hostnames[i]<<" {\n  hardware ethernet "<<reserved_mac_addr[i]<<";\n  fixed-address "<<reserved_ips[i]<<";\n  ddns-hostname \""<<reserved_hostnames[i]<<"\";\n  option host-name \""<<reserved_hostnames[i]<<"\";\n}\n\n";
//                    }
//                }
//                dhcpdconf.close();
//            }
//            else cout << "Unable to write dhcpdconf"<<endl;

//            system("rcdhcpd restart");
        }
    }

}
