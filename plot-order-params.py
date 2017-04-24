#!/usr/bin/env python3

import glob, sys, os
from pylab import *

data_dir = "./rc-data/"
fig_dir = "./figures/"
plot_N = [500]

# dictionary keys for energy, state, and distance files
E = "E"
S = "S"
D = "D"

# retrieve identifying labels from filename
def id(file_name):
    return os.path.basename(file_name).split("-")[1:]

# get N (number of nodes), P (number of patterns), and T (temperature)
def NPT(file_name):
    if type(file_name) == dict: file_name = file_name[E]
    parts = id(file_name)
    N = int(parts[0][1:])
    P = int(parts[1][1:])
    T = float(parts[2].split("T")[-1])/100
    return N, P, T

# identify and organize data files
files = {}
energy_files = sorted(glob.glob(data_dir+"energies-*-f100T*"))
state_files = sorted(glob.glob(data_dir+"states-*-f100T*"))
distance_files = sorted(glob.glob(data_dir+"distances-*-f100T*"))
for energy_file in energy_files:
    state_matches = [ state_file for state_file in state_files
                       if id(state_file) == id(energy_file) ]
    distance_matches = [ distance_file for distance_file in distance_files
                         if id(distance_file) == id(energy_file) ]
    if len(state_matches) != 1:
        print("problem matching state file:", energy_file)
        continue
    if len(distance_matches) != 1:
        print("problem matching distance file:", energy_file)
        continue
    state_file = state_matches[0]
    distance_file = distance_matches[0]

    N, P, T = NPT(energy_file)

    ################################################################################
    if 100*T % 10 != 0: continue
    ################################################################################

    if N not in files.keys():
        files[N] = {}
    if P not in files[N].keys():
        files[N][P] = {}
    if T not in files[N][P].keys():
        files[N][P][T] = []

    files[N][P][T].append({ E : energy_file,
                            S : state_file,
                            D : distance_file})

print("sorted all data files")

def compute_q_m_cv(file_set):
    N, P, T = NPT(file_set)
    energies, energy_hist = loadtxt(file_set[E], unpack = True)
    states = loadtxt(file_set[S], ndmin = 1)
    distance = loadtxt(file_set[D], ndmin = 1)

    with open(file_set[S], "r") as f:
        for line in f:
            if "records" in line:
                state_records = int(line.split()[-1])

    states /= state_records
    states = array([ 2*s-1 for s in states ])
    min_distance = distance[1] / (distance[0] * N)
    m_param = ( 2 * min_distance - 1 )**2

    q_param = mean([ s*s for s in states ])

    energies /= N
    energy_hist /= mean(energy_hist)
    Z = sum(energy_hist)
    cv = sum(energies * energies * energy_hist)/Z \
         - (sum(energies * energy_hist)/Z)**2
    cv /= N * T * T

    return array([q_param, m_param, cv])

def mean_q_m_cv(N, P, T):
    mean_q_m_cv = zeros(3)
    for file_set in files[N][P][T]:
        mean_q_m_cv += compute_q_m_cv(file_set)
    return mean_q_m_cv / len(files[N][P][T])

def borders(array):
    borders = zeros(len(array)+1)
    borders[1:-1] = (array[1:] + array[:-1]) / 2
    borders[0] = array[0] - (array[1] - array[0]) / 2
    borders[-1] = array[-1] + (array[-1] - array[-2]) / 2
    return borders

for N in sorted(files.keys()):
    if len(plot_N) > 0:
        if N not in plot_N:
            continue

    q_name = "q-N{}.pdf".format(N)
    m_name = "m-N{}.pdf".format(N)
    cv_name = "cv-N{}.pdf".format(N)

    p_vals = array(sorted(files[N].keys()))

    t_set = set(files[N][p_vals[0]].keys())
    for P in p_vals:
        t_set = t_set.intersection(files[N][P].keys())
    t_vals = array(sorted(list(t_set)))

    q_m_cv = zeros((3,len(t_vals),len(p_vals)))

    for pp in range(len(p_vals)):
        print("P: {}".format(p_vals[pp]))
        for tt in range(len(t_vals)):
            print(" T: {}".format(t_vals[tt]))
            q_m_cv[:,tt,pp] = mean_q_m_cv(N, p_vals[pp], t_vals[tt])

    p_borders = borders(p_vals)/N
    t_borders = borders(t_vals)

    figure(q_name)
    pcolor(p_borders, t_borders, q_m_cv[0,:,:])
    title("$q$ $(N={})$".format(N))
    xlabel(r"$\alpha$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig(fig_dir+q_name)

    figure(m_name)
    pcolor(p_borders, t_borders, q_m_cv[1,:,:])
    title("$m$ $(N={})$".format(N))
    xlabel(r"$\alpha$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig(fig_dir+m_name)

    figure(cv_name)
    pcolor(p_borders, t_borders, q_m_cv[2,:,:])
    title("$c_V$ $(N={})$".format(N))
    xlabel(r"$\alpha$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig(fig_dir+cv_name)

if "show" in sys.argv:
    show()
