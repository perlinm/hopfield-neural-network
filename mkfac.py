#!/usr/bin/env python
import sys, os, glob, re, subprocess

testing_mode = False
hide_warnings = False
for arg in sys.argv[1:]:
    if arg.lower() == "test":
        testing_mode = True
    elif arg.lower() == "whide":
        hide_warnings = True
    else:
        print('useage: {} [test] [whide]'.format(sys.argv[0]))
        exit(1)

executable = "simulate.exe"
eigen_dirs = ".eigen-dirs"
mkl_root = ".mkl-root"

ignore_dirs = [ "~/.ccache/" ]

language_standard_flag = "-std=c++11"
warning_flags = "-Wall -Werror"
link_time_optimization_flag = "-flto"
common_flags = " ".join([ language_standard_flag,
                          warning_flags,
                          link_time_optimization_flag ])

debug_flag = "-g"
optimization_flag = "-O3"
ignored_warning_flags = " ".join(["-Wno-unused-variable",
                                  "-Wno-unused-but-set-variable",
                                  "-Wno-unused-local-typedefs"])

mkl_flags = ("-Wl,--no-as-needed,-rpath=$(cat {0})/lib/intel64/" + \
             " -L $(cat {0})/lib/intel64/ -lmkl_intel_lp64 -lmkl_core" + \
             " -lmkl_gnu_thread -lpthread -lm -ldl -fopenmp -m64" + \
             " -I $(cat {0})/include/").format(mkl_root)

lib_flags = {"eigen3" : ["$(cat {})".format(eigen_dirs), eigen_dirs],
             "USE_MKL" : ["{}".format(mkl_flags), mkl_root],
             "boost" : ["-lboost_system"],
             "boost/filesystem" : ["-lboost_filesystem"],
             "boost/program_options" : ["-lboost_program_options"],
             "gsl" : "-lgsl"}

fac_text = ""
used_libraries = []
used_headers = []
sim_files = sorted(glob.glob("*.cpp"))

def fac_rule(libraries, headers, out_file, in_files, link=False):
    rule_text = "| g++ {} {} ".format(common_flags, (debug_flag if testing_mode
                                                     else optimization_flag))
    if hide_warnings: rule_text += ignored_warning_flags + " "
    rule_text += " ".join(libraries) + " "
    if not link: rule_text += "-c "
    rule_text += "-o {} ".format(out_file)
    rule_text += " ".join(in_files)+"\n"

    for dependency in headers + in_files:
        rule_text += "< {}\n".format(dependency)
    for ignore_dir in ignore_dirs:
        rule_text += "C {}\n".format(ignore_dir)
    rule_text += "> {}\n\n".format(out_file)
    return rule_text

for sim_file in sim_files:
    out_file = sim_file.replace(".cpp",".o")
    libraries = []
    include_files = [sim_file]
    headers = []
    with open(sim_file,'r') as f:
        for line in f:
            if "#include" in line or "#define" in line:
                for tag in lib_flags.keys():
                    if tag in line and lib_flags[tag][0] not in libraries:
                        libraries += [lib_flags[tag][0]]
                        if len(lib_flags[tag]) > 1:
                            include_files += [lib_flags[tag][1]]
                if re.search('"*\.h"',line):
                    headers += [line.split('"')[-2]]

    fac_text += fac_rule(libraries, headers, out_file, include_files)
    for library in libraries:
        if library not in used_libraries:
            used_libraries += [library]
    for header in headers:
        if header not in used_headers:
            used_headers += [header]


out_files = [ sim_file.replace(".cpp",".o") for sim_file in sim_files ]
fac_text += fac_rule(used_libraries, used_headers, executable, out_files, link=True)
fac_text += "| etags *.cpp *.h\n< {}\n> TAGS\n".format(executable)


with open(".{}".format(executable.replace(".exe",".fac")),"w") as f:
    f.write(fac_text)

exit(subprocess.call(["fac"]))
