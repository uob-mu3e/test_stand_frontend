# Clock and Reset Box

## Vivado project

To open project in Vivado GUI run `./gui.tcl`:

The ip's are generated from tcl scripts located in `ip/` directory.

To extract setting for modified ip run following in '`Tcl Console`':

```
source tcl/ip_config.tcl
ip_config_diff $module_name
```

Save '`set_property ...`' output to corresponding tcl script.
