#

proc puts_ip_config { name } {
    set ip [ get_ips $name ]
    set properties [ list_property $ip ]

    foreach property $properties {
#        if { [ string match "CONFIG.*" $property ] } {
            puts "        $property [ get_property $property $ip ] \\"
#        }
    }
}
