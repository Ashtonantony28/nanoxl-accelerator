# Dry run of a fpga/<board>/build.tcl in plain tclsh: every Vivado command the scripts use is a stub that
# logs its arguments (one Tcl list per line) and returns a plausible value; any other unknown command is
# an error, so a typo or an undefined proc fails the run. File copies are logged, not performed.
#
#   tclsh tests/fpga/vivado_stub.tcl LOG fpga/kv260/build.tcl -out /tmp/x -jobs 2
# It checks Tcl syntax and the scripts' own logic (argument parsing, dicts, procs), not Vivado semantics.

set nx_logf [open [lindex $argv 0] w]
set nx_script [lindex $argv 1]
set argv [lrange $argv 2 end]
set argc [llength $argv]

proc nx_log {args} { puts $::nx_logf $args; flush $::nx_logf }

# Vivado commands -> return value (a script body evaluated in the stub with $args)
set nx_cmds {
    create_project            {return proj}
    current_project           {return proj}
    current_fileset           {return sources_1}
    get_filesets              {return [lindex $args end]}
    add_files                 {return [lindex $args end]}
    update_compile_order      {return {}}
    set_property              {return {}}
    get_property              {
        set p [lindex $args 0]
        switch -- $p {
            PROGRESS  {return 100%}
            SLACK     {return 0.125}
            DIRECTORY {return /stub/impl_1}
            NAME      {return [lindex $args 1]}
            default   {return {}}
        }
    }
    get_board_parts           {return {}}
    get_files                 {
        if {[lsearch $args -all] >= 0} { return /stub/gen/nanoxl_bd.hwh }
        return [lindex $args end]
    }
    create_bd_design          {return [lindex $args end]}
    create_bd_cell            {return [lindex $args end]}
    apply_bd_automation       {return {}}
    get_bd_pins               {return $args}
    get_bd_intf_pins          {return $args}
    connect_bd_net            {return {}}
    connect_bd_intf_net       {return {}}
    get_bd_addr_spaces        {return [lindex $args end]}
    get_bd_addr_segs          {
        if {[lsearch $args -of_objects] >= 0} {
            set p [lindex $args end]
            return [list $p/X_DDR_LOW $p/X_DDR_HIGH $p/X_LPS_OCM $p/X_QSPI]
        }
        return [lindex $args end]
    }
    assign_bd_address         {return {}}
    exclude_bd_addr_seg       {return {}}
    validate_bd_design        {return {}}
    save_bd_design            {return {}}
    generate_target           {return {}}
    make_wrapper              {return /stub/hdl/nanoxl_bd_wrapper.v}
    get_runs                  {return [lindex $args end]}
    launch_runs               {return {}}
    wait_on_run               {return {}}
    open_run                  {return {}}
    report_timing_summary     {return {}}
    report_utilization        {return {}}
    report_ram_utilization    {return {}}
    report_clock_utilization  {return {}}
    report_design_analysis    {return {}}
    get_timing_paths          {return path}
    write_hw_platform         {return {}}
}
dict for {c body} $nx_cmds {
    proc $c {args} "nx_log $c {*}\$args\n$body"
}

# `file copy` is logged only (the build copies Vivado outputs that do not exist here); run with -out in a temp dir
rename file nx_real_file
proc file {sub args} {
    if {$sub eq "copy"} { nx_log file $sub {*}$args; return {} }
    return [nx_real_file $sub {*}$args]
}

if {[catch {source $nx_script} err opts]} {
    puts stderr "STUB-RUN ERROR: $err"
    puts stderr [dict get $opts -errorinfo]
    close $nx_logf
    exit 1
}
nx_log STUB_DONE
close $nx_logf
exit 0
