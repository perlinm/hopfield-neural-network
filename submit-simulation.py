#!/usr/bin/env python3
import sys, os, subprocess, socket

whide_flag = "whide"
fac = "~/src/fac/fac"
mkfac = "./mkfac.py"
job_dir = "jobs"
simulate = "simulate.exe"

# process inputs
if len(sys.argv) < 2:
    print("usage: {} [{}] walltime_in_hours" + \
          " [simulation arguments]".format(sys.argv[0]),whide_flag)
    exit(1)

whide = whide_flag in sys.argv
if whide: sys.argv.remove(whide_flag)

walltime_in_hours = sys.argv[1]
sim_args = sys.argv[2:]

# before proceeding further, build the project
print("building project locally...")
subprocess.call([mkfac] + ([ whide_flag ] if whide else []))

# determine simulation file basename
suffix_cmd = "./simulate.exe --suffix "+" ".join(sim_args)
suffix = (subprocess.check_output(suffix_cmd, shell = True)).split()[-1].decode("utf-8")
basename = "network" + ".".join(suffix.split(".")[:-1])
job_file = "{}/{}.sh".format(job_dir,basename)

# set batch options
options = [ [ "output", basename + ".out" ],
            [ "error", basename + ".err" ],
            [ "time", walltime_in_hours + ":00:00" ],
            [ "nodes", 1 ],
            [ "ntasks", 1 ],
            [ "ntasks-per-node", 1 ],
            [ "account", "clphys72300117" ] ]

# construct and write job file
job_text = "#!/usr/bin/env sh\n"
for option in options:
    job_text += "#SBATCH --{} {}\n".format(option[0],option[1])
job_text += "\n"
job_text += "./{} ".format(simulate) + " ".join(sim_args) + "\n"

if not os.path.isdir(job_dir):
    os.makedirs(job_dir)
with open(job_file, "w") as f:
    f.write(job_text)

if "colorado.edu" in socket.getfqdn():
    modules = " ".join([ "gcc/6.1.0", "openmpi/1.10.2", "boost/1.61.0", "python/3.5.1" ])
    build_list = [ "ssh", "scompile",
                   "module load {}; cd {}; {}".format(modules, os.getcwd(), fac) ]
    print("recompiling on a compile node...")
    subprocess.call(build_list)
    compile_cmd = "sbatch"

else:
    compile_cmd = "sh"

subprocess.call([ compile_cmd, job_file ])
