#!/usr/bin/env python3
import sys, os, subprocess, socket

project_dir = os.path.dirname(os.path.abspath(__file__))

whide_flag = "whide"
mkfac = project_dir + "/mkfac.py"
job_dir = project_dir + "/jobs"
simulate = "simulate.exe"

whide = whide_flag in sys.argv
if whide: sys.argv.remove(whide_flag)

if len(sys.argv) < 2:
    time = "00:01:00"
else:
    time = sys.argv[1]
    while time.count(":") < 2:
        time += ":00"

sim_args = sys.argv[2:]

simulate_cmd_list = [ "{}/{}".format(project_dir, simulate) ] + sim_args

on_rc_server = ("colorado.edu" in socket.getfqdn())
rc_modules = [ "gcc/5.1.0", "openmpi/1.8.5", "boost/1.61.0" ]
rc_initialize = "module load " + " ".join(rc_modules)

# before proceeding further, build the project
print("building project...")
if on_rc_server: subprocess.call(rc_initialize, shell = True)
subprocess.call([mkfac])

# determine simulation file basename
suffix_cmd_list = simulate_cmd_list + [ "--suffix" ]
suffix_cmd_result = subprocess.run(suffix_cmd_list, stdout = subprocess.PIPE)
suffix = suffix_cmd_result.stdout.decode("utf-8").split()[-1]
basename = "network" + ".".join(suffix.split(".")[:-1])
job_file = "{}/{}.sh".format(job_dir,basename)
out_file = "{}/{}.out".format(job_dir,basename)
err_file = "{}/{}.err".format(job_dir,basename)

# set batch options
options = [ [ "output", out_file ],
            [ "error", err_file ],
            [ "time", time ],
            [ "nodes", 1 ],
            [ "ntasks", 1 ],
            [ "ntasks-per-node", 1 ],
            [ "account", "clphys72300117" ] ]

# construct and write job file
job_text = "#!/usr/bin/env sh\n"
for option in options:
    job_text += "#SBATCH --{} {}\n".format(option[0], option[1])
job_text += "\n"
if on_rc_server: job_text += rc_initialize + "\n"
job_text += " ".join(simulate_cmd_list) + "\n"

if not os.path.isdir(job_dir):
    os.makedirs(job_dir)
with open(job_file, "w") as f:
    f.write(job_text)

submit_cmd = ("sbatch" if on_rc_server else "sh")
subprocess.call([ submit_cmd, job_file ])
