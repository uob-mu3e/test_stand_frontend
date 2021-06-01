mu3e online will run in a private network segment (typicall the 196.168. segment)
The backend PC will also be connected with the network going to the storage node.
Either the backend PC or anothe box will serve as a gateway to the wider network.

If the gateway runs OpenSuse, configure both network interfaces properly, enable IPv4 forwarding
and configure IPTABLES as follows (assuming eth0 is internal Lan and eth1 extern):
~~~~
# iptables -F
# iptables -A INPUT -i lo -j ACCEPT
# iptables -A INPUT -s 192.168.0.0/24 -i eth0 -j ACCEPT
# iptables -t  nat  -A POSTROUTING -s 192.168.0.0/24 -o eth1 -j MASQUERADE
# echo 1 > /proc/sys/net/ipv4/ip_forward
~~~~

Start dhcp and nameserver with 
~~~~
# sudo rcnamed restart
# sudo rcdhcpd restart
~~~~

Status can be checked with
~~~~
# systemctl status named.service
# systemctl status dhcpd.service
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

hostnames will be assigned by a nameserver running on the mu3eGateway machine.
To do Port forwarding to access midas behind the gateway server execute this command on the gateway machine: 
(ToDo: proper solution with iptables port forwarding that does not interfer with dhcp configuration above)
~~~~
#  ssh -L \*:8081:localhost:8081 ip.of.midas.host
~~~~