// Midas frontend for the DHCP and DNS server on the mu3e gateway

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <cassert>
#include "midas.h"
#include "mfe.h"

using namespace std;

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "DHCP DNS";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = TRUE;

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
"nUnknown = STRING[1]:",
    "[32] 0",
"nReserved = STRING[1]:",
    "[32] 0",
"nLeased = STRING[1]:",
    "[32] 0",
"nDNS = STRING[1]:",
    "[32] 0",
"usereditReserveIP = STRING[1]:"
"[32] 000.000.000.000",
"usereditReserveMAC = STRING[1]:"
"[32] 00:00:00:00:00",
"usereditReserveHost = STRING[1]:"
"[32] hostname",
"usereditRemReserveIP = STRING[1]:"
"[32] 000.000.000.000",
"leasedIPs = STRING[255] :",
"[32] 0",
"leasedHostnames = STRING[255] :",
"[32] 0",
"DNSips = STRING[255] :",
"[32] 0",
"DNSHostnames = STRING[255] :",
"[32] 0",
nullptr
};

EQUIPMENT equipment[] = {

   {"DHCP DNS",              /* equipment name */
    {11, 0,                     /* event ID, trigger mask */
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
    
    for(int i = 0; i<ips.size(); i++){
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
    for(int i=0; i<active_ips.size();i++){
        known = 0;
        for(int j=0; j<ips.size();j++){
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
    else cout << "Unable to read dhcpd.conf" <<endl;
}


/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
{
   return 1;
};

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   return 1;
};

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
   HNDLE hKey;
   
   system("touch /var/lib/dhcp/etc/dhcpd_reservations.conf");

   // create Settings structure in ODB
   db_create_record(hDB, 0, "Equipment/DHCP DNS/Settings", strcomb(cr_settings_str));
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/leasedIPs", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/leasedHostnames", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/leasedMACs", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/expiration", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/reservedIPs", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/reservedHostnames", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/reservedMACs", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/unknownIPs", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/DNSips", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/DNSHostnames", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/usereditReserveIP", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/usereditReserveMAC", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/usereditReserveHost", TID_STRING);
   db_create_key(hDB, 0, "Equipment/DHCP DNS/Settings/usereditRemReserveIP", TID_STRING);

   db_find_key(hDB, 0, "/Equipment/DHCP DNS", &hKey);
   assert(hKey);

   db_watch(hDB, hKey, netfe_settings_changed, nullptr);

   // add custom page to ODB
   db_create_key(hDB, 0, "Custom/DHCP DNS&", TID_STRING);
   const char * name = "net.html";
   db_set_value(hDB,0,"Custom/DHCP DNS&",name, sizeof(name), 1,TID_STRING);

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
    // slow down
    sleep(10);
    
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
    for(int i = 0; i< reserved_ips.size();i++)
        expiration.push_back("inf");
    
    // if new dhcp request:   --> update dns table

    if(prev_ips!=ips){
        cm_msg(MINFO, "netfe_settings_changed", "new dhcp lease, odb updated");
        
        system("wall new dhcp lease, update and restart of dns server");
        write_dns_table(dns_zone_def_path,ips,requestedHostnames);
        system("rcnamed restart");
        
        //update odb only if there was a change (prev_ips!=ips). Rewrite everything if something changed
        db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/leasedIPs", ips[0].c_str(), sizeof(ips[0]), 1, TID_STRING);
        for (int i = 0; i < ips.size(); i++) {
            //TODO: find a way to do this in a single command !!! (without loop of db_set_value)
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/leasedIPs", ips[i].c_str(), sizeof(ips[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/leasedHostnames", requestedHostnames[i].c_str(), sizeof(requestedHostnames[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/expiration", expiration[i].c_str(), sizeof(expiration[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/leasedMACs", macs[i].c_str(), sizeof(macs[i]), i, TID_STRING, FALSE);
        }
        db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/nLeased", to_string(ips.size()).c_str(), sizeof(to_string(ips.size()).c_str()), 1,TID_STRING);
        
        for (int i = 0; i < reserved_ips.size(); i++) {
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/reservedIPs", reserved_ips[i].c_str(), sizeof(reserved_ips[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/reservedHostnames", reserved_hostnames[i].c_str(), sizeof(reserved_hostnames[i]), i, TID_STRING, FALSE);
            db_set_value_index(hDB, 0, "Equipment/DHCP DNS/Settings/reservedMACs", reserved_mac_addr[i].c_str(), sizeof(reserved_mac_addr[i]), i, TID_STRING, FALSE);
        }
        db_set_value(hDB,0,"Equipment/DHCP DNS/Settings/nReserved",to_string(reserved_ips.size()).c_str(), sizeof(to_string(reserved_ips.size()).c_str()), 1,TID_STRING);
        
    }
    
    // ping everything in subnet
    active_ips = find_active_ips("192.168.0.");
    
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

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
   return CM_SUCCESS;
}

/*--- Read Clock and Reset Event to be put into data stream --------*/

INT read_cr_event(char *pevent, INT off)
{
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
      if(value==true)
        cm_msg(MINFO, "netfe_settings_changed", "DHCPD activated by user");
      else
        cm_msg(MINFO, "netfe_settings_changed", "DHCPD deactivated by user");
      system("wall dhcpd changed");
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

            
            ofstream dhcpdconf ("/var/lib/dhcp/etc/dhcpd_reservations.conf", std::ios_base::app);
            if (dhcpdconf.is_open())
            {
                dhcpdconf << "host "<<hostname<<" {\n  hardware ethernet "<<mac<<";\n  fixed-address "<<ip<<";\n  ddns-hostname \""<<hostname<<"\";\n  option host-name \""<<hostname<<"\";\n}\n\n";
                
                dhcpdconf.close();
            }
            else cout << "Unable to write dhcpdconf"<<endl;
            
            system("rcdhcpd restart");
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
            else cout << "Unable to write dhcpdconf"<<endl;
            
            system("rcdhcpd restart");
        }
    }
   
} 