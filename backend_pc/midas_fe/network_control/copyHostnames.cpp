#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

using namespace std;

void read_leases(string path, vector <string> *ips, vector <string> *requestedHostnames){
    string line;
    ifstream leases(path);
    bool hostname_requested = false;
    //int lease_number = -1;

    if (leases.is_open()){
        while ( getline (leases,line)){
            if(line.substr(0,5)=="lease"){
                ips->push_back(line.substr(6,line.length()-8));
                hostname_requested=false;

                // search for requested hostname, macaddr, etc. in this lease now:
                while(line != "}" and getline(leases,line)){
                    if(line.length()<17) continue;
                    if(line.substr(2,15)=="client-hostname"){
                        hostname_requested=true;
                        requestedHostnames->push_back(line.substr(19,line.length()-21));
                    }
                }

                if(!hostname_requested) requestedHostnames->push_back("-");
            }

        }
        leases.close();
    }
    else cout << "Unable to open file" <<endl;
}


void write_dns_table(string path, vector <string> ips, vector <string> requestedHostnames){
  ofstream dnstable (path);
  if (dnstable.is_open())
  {
    dnstable << "$TTL 2D\n@\t\tIN SOA\t\tDHCP-214.mu3e.kph.\troot.DHCP-214.mu3e.kph. (\n\t\t\t\t2019062601\t; serial\n\t\t\t\t3H\t\t; refresh\n\t\t\t\t1H\t\t; retry\n\t\t\t\t1W\t\t; expiry\n\t\t\t\t1D )\t\t; minimum\n\nmu3e.\t\tIN NS\t\tDHCP-214.mu3e.kph.\n";
    for(size_t i = 0; i<ips.size(); i++){
        if(requestedHostnames[i]!="-")
            dnstable<<requestedHostnames[i]<<"              IN      A       "<<ips[i]<<"\n";
    }
    dnstable.close();
  }
  else cout << "Unable to open file"<<endl;
}

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
    else cout << "Unable to open file" <<endl;
}




int main(int, char **)
{
    vector <string> ips;
    vector <string> requestedHostnames;
    vector <string> prev_ips;
    vector <string> prev_requestedHostnames;
    vector <string> active_ips;
    vector <string> unknown_ips;

    vector <string> reserved_mac_addr;
    vector <string> reserved_ips;
    vector <string> reserved_hostnames;

    string subnet = "192.168.0.";  // only X.X.X. here !!!
    string dhcpd_lease_path = "/var/lib/dhcp/db/dhcpd.leases";
    string dns_zone_def_path = "/var/lib/named/master/mu3e";
    string dhcpd_conf_path = "/etc/dhcpd.conf";

    //string dhcpd_lease_path_test = "/home/labor/Desktop/dhcpd.leases";
    //string dns_zone_def_path_test = "/home/labor/Desktop/mu3e";

    while(true){
        prev_ips = ips;
        prev_requestedHostnames = requestedHostnames;
        ips.clear();
        requestedHostnames.clear();
        reserved_ips.clear();
        reserved_hostnames.clear();
        reserved_mac_addr.clear();

        read_leases(dhcpd_lease_path, &ips, &requestedHostnames);
        read_reserved(dhcpd_conf_path, &reserved_ips, &reserved_hostnames, &reserved_mac_addr);

        // append reserved ips tp ip list:
        ips.insert(ips.end(), reserved_ips.begin(), reserved_ips.end());
        requestedHostnames.insert(requestedHostnames.end(), reserved_hostnames.begin(), reserved_hostnames.end());

        // if new dhcp request:   --> update dns table
        if(prev_ips!=ips or prev_requestedHostnames != requestedHostnames){

            system("wall new dhcp lease, update and restart of dns server");
            cout<<"new ip or hostname in dhcp lease list, updating and restarting DNS server"<<endl;

            cout<<"-----------------------------------------"<<endl;
            cout<<"reserved ips:"<<endl;
            for(size_t i=0; i<reserved_ips.size(); i++){
                cout<<reserved_ips[i]<<" \t"<< reserved_hostnames[i]<<" \t"<< reserved_mac_addr[i]<<endl;
            }

            cout<<"-----------------------------------------"<<endl;
            cout<<"previous leased ips:"<<endl;
            cout<<"leased ip \t"<<"requested Hostname"<< endl;
            for(size_t i=0; i<prev_ips.size(); i++){
                cout<<prev_ips[i]<<" \t"<< prev_requestedHostnames[i]<<endl;
            }

            cout<<"-----------------------------------------"<<endl;
            cout<<"new leased ips:"<<endl;
            cout<<"leased ip \t"<<"requested Hostname"<< endl;
            for(size_t i=0; i<ips.size(); i++){
                cout<<ips[i]<<" \t"<< requestedHostnames[i]<<endl;
            }

            cout<< "updating dns table"<<endl;
            write_dns_table(dns_zone_def_path,ips,requestedHostnames);

            cout<< "restarting dns server"<<endl;
            system("rcnamed restart");

        }

        // slow down
        sleep(10);

        // ping everything in subnet
        active_ips = find_active_ips("192.168.0.");

        unknown_ips = find_unknown(ips, active_ips);

        if(unknown_ips.size()>0){
            cout<<"---------------------------------------"<<endl;
            cout<<"WARNING: found unknown fixed IPs ------"<<endl;
            cout<<"---------------------------------------"<<endl;
            cout<<"remove them or give them a name in dhcpd.conf:"<<endl;
            for(size_t i=0; i<unknown_ips.size(); i++){
                cout<<unknown_ips[i]<<endl;
            }
        }
    }
    return 0;
}
