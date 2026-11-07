# Shared Vivado procedures for the NanoXL overlay builds (fpga/kv260/build.tcl, fpga/pynqz2/build.tcl).
#
# UNTESTED: written without Vivado or hardware (SG9a). Targets Vivado ML Standard 2024.2; every command
# used here exists since 2020.2. tests/fpga only checks syntax and the file/parameter lists.
#
# Flow: create project -> add RTL -> block design (board script) -> validate -> global synthesis
# (so that fpga/<board>/*.xdc reaches the scratchpad RAMs) -> implementation -> bitstream ->
# timing / utilization reports -> overlay/nanoxl.{bit,hwh} for PYNQ.

set NX_COMMON [file normalize [file dirname [info script]]]
set NX_ROOT   [file normalize [file join $NX_COMMON .. ..]]
source [file join $NX_COMMON rtl_files.tcl]

# ---------------------------------------------------------------- command line
# -jobs N            parallel runs (default 4)
# -out DIR           build directory (default fpga/<board>/build)
# -clk MHZ           PL clock override (default: the board's target)
# -params {K V ...}  nanoxl_top parameter overrides, e.g. -params {N 16 L 16}
# -strategy NAME     implementation strategy (default Performance_ExplorePostRoutePhysOpt)
# -proj_only         stop after the block design is validated and saved (open it in the GUI)
# -no_bit            stop after implementation (reports, no bitstream)
proc nx_parse_args {argv defaults} {
    set opt [dict merge {jobs 4 out "" clk "" params {} strategy Performance_ExplorePostRoutePhysOpt
                         proj_only 0 no_bit 0} $defaults]
    set i 0
    while {$i < [llength $argv]} {
        set a [lindex $argv $i]
        switch -- $a {
            -jobs - -out - -clk - -params - -strategy {
                incr i
                dict set opt [string range $a 1 end] [lindex $argv $i]
            }
            -proj_only { dict set opt proj_only 1 }
            -no_bit    { dict set opt no_bit 1 }
            default    { error "unknown argument '$a' (see fpga/common/nanoxl_build.tcl)" }
        }
        incr i
    }
    return $opt
}

# ---------------------------------------------------------------- project + RTL
proc nx_create_project {name part dir} {
    global NX_ROOT NANOXL_RTL
    file mkdir $dir
    create_project -force $name $dir -part $part
    set_property target_language Verilog [current_project]
    set files {}
    foreach f $NANOXL_RTL {
        set p [file join $NX_ROOT $f]
        if {![file exists $p]} { error "missing RTL file $p" }
        lappend files $p
    }
    add_files -norecurse -fileset sources_1 $files
    foreach f [get_files -of_objects [get_filesets sources_1] *.sv] {
        set_property file_type SystemVerilog $f
    }
    update_compile_order -fileset sources_1
}

proc nx_add_xdc {xdc} {
    add_files -fileset constrs_1 -norecurse $xdc
    # the RAM_STYLE constraints are synthesis constraints
    set_property used_in_synthesis true [get_files $xdc]
    set_property used_in_implementation true [get_files $xdc]
}

# The NanoXL core as a module reference (rtl/nanoxl_axi_wrap.v is plain Verilog, as module references
# require). `params` is a dict of nanoxl_top parameters (fpga/boards.py RTL_PARAMS).
proc nx_add_core {params} {
    set c [create_bd_cell -type module -reference nanoxl_axi_wrap nanoxl_0]
    set cfg {}
    dict for {k v} $params { lappend cfg CONFIG.$k $v }
    set_property -dict $cfg $c
    return $c
}

# Map the slave segments of `slave_intf` whose path matches `seg_glob` (e.g. *DDR_LOW*) into the master
# address space `space`, and exclude the others (OCM, QSPI, PCIe windows, DDR_HIGH above 4 GiB, which a
# 32-bit master cannot reach anyway).
proc nx_map_master {space slave_intf seg_glob} {
    set n 0
    set sp [get_bd_addr_spaces $space]
    foreach s [get_bd_addr_segs -of_objects [get_bd_intf_pins $slave_intf]] {
        if {[string match $seg_glob $s]} {
            assign_bd_address -target_address_space $sp $s
            incr n
        } else {
            catch {exclude_bd_addr_seg -target_address_space $sp $s}
        }
    }
    if {$n == 0} { error "no address segment matching $seg_glob on $slave_intf" }
}

# ---------------------------------------------------------------- synth / impl / export
proc nx_finish {proj_dir bd_name opt} {
    set bd [get_files $bd_name.bd]
    validate_bd_design
    save_bd_design
    # global synthesis: one synth_1 run sees the whole design and the board XDC (RAM_STYLE on the
    # scratchpad banks); out-of-context runs would not.
    set_property synth_checkpoint_mode None $bd
    generate_target all $bd
    set wrapper [make_wrapper -files $bd -top]
    add_files -norecurse $wrapper
    set_property top ${bd_name}_wrapper [current_fileset]
    update_compile_order -fileset sources_1
    if {[dict get $opt proj_only]} {
        puts "NANOXL: project ready in $proj_dir (block design saved, not built)"
        return
    }
    set_property strategy [dict get $opt strategy] [get_runs impl_1]
    set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
    set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
    set jobs [dict get $opt jobs]
    launch_runs synth_1 -jobs $jobs
    wait_on_run synth_1
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synth_1 failed: see $proj_dir" }
    if {[dict get $opt no_bit]} {
        launch_runs impl_1 -jobs $jobs
    } else {
        launch_runs impl_1 -to_step write_bitstream -jobs $jobs
    }
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "impl_1 failed: see $proj_dir" }

    set rpt [file join $proj_dir reports]
    file mkdir $rpt
    open_run impl_1
    report_timing_summary -max_paths 20 -report_unconstrained -file [file join $rpt timing_summary.rpt]
    report_utilization -file [file join $rpt utilization.rpt]
    report_utilization -hierarchical -hierarchical_depth 6 -file [file join $rpt utilization_hier.rpt]
    report_ram_utilization -file [file join $rpt ram_utilization.rpt]
    report_clock_utilization -file [file join $rpt clock_utilization.rpt]
    report_design_analysis -logic_level_distribution -file [file join $rpt logic_levels.rpt]
    set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
    set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
    set f [open [file join $rpt summary.txt] w]
    puts $f "WNS $wns"
    puts $f "WHS $whs"
    close $f
    puts "NANOXL: WNS = $wns ns, WHS = $whs ns (reports in $rpt)"
    if {$wns < 0 || $whs < 0} {
        puts "NANOXL: WARNING timing NOT met -- see docs/fpga.md §5 (lower -clk or apply the mitigations)"
    }
    if {[dict get $opt no_bit]} { return }

    # overlay for PYNQ: nanoxl.bit + nanoxl.hwh (same base name), plus the .xsa for reference
    set ov [file join $proj_dir overlay]
    file mkdir $ov
    set impl_dir [get_property DIRECTORY [get_runs impl_1]]
    file copy -force [file join $impl_dir ${bd_name}_wrapper.bit] [file join $ov nanoxl.bit]
    set hwh [lindex [get_files -all -of_objects $bd ${bd_name}.hwh] 0]
    if {$hwh eq ""} {
        set hwh [lindex [glob -nocomplain [file join $proj_dir * sources_1 bd $bd_name hw_handoff ${bd_name}.hwh]] 0]
    }
    if {$hwh eq ""} { error "hwh not found (look under $proj_dir/*.gen/sources_1/bd/$bd_name/hw_handoff)" }
    file copy -force $hwh [file join $ov nanoxl.hwh]
    write_hw_platform -fixed -include_bit -force -file [file join $ov nanoxl.xsa]
    puts "NANOXL: overlay written to $ov (nanoxl.bit, nanoxl.hwh, nanoxl.xsa)"
}
