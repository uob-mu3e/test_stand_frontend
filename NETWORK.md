mu3e online will run in a private network segment (typicall the 196.168. segment)
The backend PC will also be connected with the network going to the storage node.
Either the backend PC or anothe box will serve as a gateway to the wider network.

If the gateway runs OpenSuse, configure both network nterfaces properly, anebale IPv4 forwarding
and configure IPTABLES as follows:
~~~~
# iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
# iptables -A FORWARD -i eth0 -o eth1 -m state --state RELATED,ESTABLISHED -j ACCEPT
# iptables -A FORWARD -i eth1 -o eth0 -j ACCEPT
~~~~
