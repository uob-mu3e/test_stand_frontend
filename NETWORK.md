mu3e online will run in a private network segment (typicall the 196.168. segment)
The backend PC will also be connected with the network going to the storage node.
Either the backend PC or anothe box will serve as a gateway to the wider network.

If the gateway runs OpenSuse, configure both network interfaces properly, enable IPv4 forwarding
and configure IPTABLES as follows:
~~~~
# iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
# iptables -A FORWARD -i eth0 -o eth1 -m state --state RELATED,ESTABLISHED -j ACCEPT
# iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
~~~~

A ip address can be reserved for a specific device by adding a entry of the following form
to /etc/dhcpd.conf:
~~~~
# host MSCB263 { 
#   hardware ethernet 00:50:c2:46:d1:07; 
#   fixed-address 192.168.0.100;
#   option host-name "MSCB263";
# }
~~~~

A ip lease list can be found in /var/lib/dhcp/db/dhcpd.leases
Monitoring of DHCP requests can be done with:
~~~~
#  sudo tcpdump -i eth<insert number> port 67 or port 68 -e -n 
~~~~

hostnames can be assigned by running a nameserver on the DHCP gateway 
or adding ip and hostname to /etc/hosts on ALL machines