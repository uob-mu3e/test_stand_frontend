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
    for(int i = 0; i<ips.size(); i++){
        if(requestedHostnames[i]!="-")
            dnstable<<requestedHostnames[i]<<"              IN      A       "<<ips[i]<<"\n";
    }
    dnstable.close();
  }
  else cout << "Unable to open file"<<endl;
    
}

int main(int argc, char * argv[])
{
    vector <string> ips;
    vector <string> requestedHostnames;
    vector <string> prev_ips;
    vector <string> prev_requestedHostnames;
    
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
        read_leases(dhcpd_lease_path, &ips, &requestedHostnames);
        
        if(prev_ips!=ips or prev_requestedHostnames != requestedHostnames){
            
            system("wall new dhcp lease, update and restart of dns server");
            cout<<"new ip or hostname in dhcp lease list, updating and restarting DNS server"<<endl;
            
            cout<<"-----------------------------------------"<<endl;
            cout<<"previous leased ips:"<<endl;
            cout<<"leased ip \t"<<"requested Hostname"<< endl;
            for(int i=0; i<prev_ips.size(); i++){
                cout<<prev_ips[i]<<" \t"<< prev_requestedHostnames[i]<<endl;
            }
            
            cout<<"-----------------------------------------"<<endl;
            cout<<"new leased ips:"<<endl;
            cout<<"leased ip \t"<<"requested Hostname"<< endl;
            for(int i=0; i<ips.size(); i++){
                cout<<ips[i]<<" \t"<< requestedHostnames[i]<<endl;
            }
            
            cout<< "updating dns table"<<endl;
            write_dns_table(dns_zone_def_path,ips,requestedHostnames);
            
            cout<< "restarting dns server"<<endl;
            system("rcnamed restart");
        }    
        sleep(10);
    }
    return 0;
}
