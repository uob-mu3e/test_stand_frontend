#

proc ip_config_puts { name } {
    set ip [ get_ips $name ]
    set properties [ list_property $ip ]

    foreach property $properties {
        puts "    $property [ get_property $property $ip ] \\"
    }
}

proc ip_config_diff {
    name
    { dir .cache/ }
} {
    set ip [ get_ips $name ]
    set properties [ list_property $ip ]

    set name_1 ${name}_1
    if { [ string equal [ get_ips $name_1 ] "" ] } {
        create_ip -vlnv [ get_property IPDEF $ip ] \
                  -module_name $name_1 -dir $dir
    }
    set ip_1 [ get_ips $name_1 ]

    puts "set_property -dict \[ list \\"
    foreach property $properties {
        if { ! [ string match "CONFIG.*" $property ] } continue;
        if { [ string equal $property "CONFIG.Component_Name" ] } continue;

        set val [ get_property $property $ip ]
        set val_1 [ get_property $property $ip_1 ]
        if { [ string equal $val $val_1 ] } continue

        puts "    $property \{$val\} \\"
    }
    puts "\] \[ get_ips \$module_name \]"
}
