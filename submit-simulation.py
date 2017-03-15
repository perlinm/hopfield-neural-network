#!/usr/bin/env python3
import sys, os, subprocess, socket

whide_flag = "whide"
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

on_rc_server = ("colorado.edu" in socket.getfqdn())
rc_modules = " ".join([ "gcc/6.1.0", "openmpi/1.10.2", "boost/1.61.0", "python/3.5.1" ])
def rc_exec_list(cmd):
    return [ "ssh", "scompile",
             "module load {}; cd {}; {}".format(rc_modules, os.getcwd(), cmd) ]

# get output of a command
def get_output(cmd_list):
    result = subprocess.run(cmd_list, stdout = subprocess.PIPE)
    return result.stdout.decode("utf-8")

# before proceeding further, build the project
print("building project...")
if on_rc_server:
    subprocess.call([ mkfac, "no-build" ])
    fac = get_output([ "which", "fac" ])
    subprocess.call(rc_exec_list(fac))
else:
    subprocess.call([mkfac])

# determine simulation file basename
suffix_cmd = [ "./{}".format(simulate), "--suffix" ]
if sim_args != []:
    suffix_cmd += sim_args
if on_rc_server:
    suffix_cmd = rc_exec_list(" ".join(suffix_cmd))

suffix = get_output(suffix_cmd).split()[-1]
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

submit_cmd = ("sbatch" if on_rc_server else "sh")
subprocess.call([ submit_cmd, job_file ])
