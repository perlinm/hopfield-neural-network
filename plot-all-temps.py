#!/usr/bin/env python3

import glob, sys, os
from pylab import *

# some plot options
font_size = 12
font = {"family" : "serif",
        "serif":["Computer Modern"],
        "size" : font_size}
rc("font",**font)
params = {"legend.fontsize": font_size}
rcParams.update(params)

min_T = 0.05
max_temp = 2
temp_steps = 100
temps = array(linspace(min_T, max_temp, temp_steps))

minN = 30
maxN = 120
alpha1D = 0.1
N2D = 100

data_dir = "./rc-data/"
fig_dir = "./figures/"

# dictionary keys for energy, weight, and distance files
E = "E"
W = "W"
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
energy_files = sorted(glob.glob(data_dir+"energies-*-100T*"))
weight_files = sorted(glob.glob(data_dir+"weights-*-100T*"))
distance_files = sorted(glob.glob(data_dir+"distances-*-100T*"))
for energy_file in energy_files:
    N, P, _ = NPT(energy_file)
    if N < minN or N > maxN: continue

    weight_matches = [ weight_file for weight_file in weight_files
                       if id(weight_file) == id(energy_file) ]
    distance_matches = [ distance_file for distance_file in distance_files
                         if id(distance_file) == id(energy_file) ]
    if len(weight_matches) != 1:
        print("problem matching weight file:", energy_file)
        continue
    if len(distance_matches) != 1:
        print("problem matching distance file:", energy_file)
        continue
    weight_file = weight_matches[0]
    distance_file = distance_matches[0]

    if N not in files.keys():
        files[N] = {}
    if P not in files[N].keys():
        files[N][P] = []

    files[N][P].append({ E : energy_file,
                         W : weight_file,
                         D : distance_file})

# method which computes internal energy, heat capacity,
#   configurational entropy, and min distance from any pattern
#   for a system defined by a particular data file pair
#   for each temperature in a "temps" array
def U_CV_S_M(file_set):
    N, P, _ = NPT(file_set)
    energies_hist, hist_input, _ = loadtxt(file_set[E], unpack = True)
    energies_weights, ln_weights_input = loadtxt(file_set[W], unpack = True)
    energies_mist, dist_records, dist_sums = loadtxt(file_set[D], unpack = True)

    energies = array([ e for e in energies_hist
                       if (e in energies_weights and e in energies_mist) ])
    hist = array([ hist_input[ii] for ii in range(len(hist_input))
                   if energies_hist[ii] in energies ])
    ln_weights = array([ ln_weights_input[ii] for ii in range(len(ln_weights_input))
                         if energies_weights[ii] in energies ])
    dist_records = array([ dist_records[ii] for ii in range(len(dist_records))
                      if energies_mist[ii] in energies ])
    dist_sums = array([ dist_sums[ii] for ii in range(len(dist_sums))
                        if energies_mist[ii] in energies ])

    ln_dos = log(hist) - ln_weights

    # correct for the factor of N in the definition of energy in the simulations
    energies /= NPT(file_set)[0]

    U_CV_S_M = zeros((4,temp_steps))
    for ii in range(temp_steps):
        ln_dos_boltz = ln_dos - energies/temps[ii]
        dos_boltz = exp(ln_dos_boltz - ln_dos_boltz.max())
        Z = sum(dos_boltz)

        # U = < E >
        U_CV_S_M[0,ii] = sum(energies*dos_boltz)/Z
        # CV = < E^2 > - < E >^2
        U_CV_S_M[1,ii] = sum((energies/temps[ii])**2 * dos_boltz)/Z \
                         - (sum(energies/temps[ii] * dos_boltz)/Z)**2
        # S = < E/T + ln(Z) >
        U_CV_S_M[2,ii] = sum(dos_boltz*(energies/temps[ii])) / Z \
                         + ln_dos_boltz.max() + log(Z)

        D_ii = sum(dist_sums / dist_records * dos_boltz)/Z
        U_CV_S_M[3,ii] = ( 2 * D_ii - 1 )**2 / N


    # subtract off the asymptotic value of S(T->\infty) from S(T) in order to
    #   "standardize" our measure of entropy across different simulations
    Z_inf = sum(exp(ln_dos - ln_dos.max()))
    S_inf = ln_dos.max() + log(Z_inf)
    U_CV_S_M[2,:] -= S_inf

    return U_CV_S_M

def mean_u_cv_s_m(N, P):
    u_cv_s_m = zeros((4,temp_steps))
    for file_set in files[N][P]:
        if NPT(file_set)[2] > min_T: continue
        u_cv_s_m += U_CV_S_M(file_set)
    u_cv_s_m /= len(files[N][P]) * N
    return u_cv_s_m

######### convergence (1D) plots ##########

if "1d" in sys.argv:

    for N in sorted(files.keys()):
        P = N * alpha1D
        if P not in files[N].keys(): continue

        u_cv_s_m = mean_u_cv_s_m(N, P)

        figure("u_fig")
        plot(temps, u_cv_s_m[0,:], label = N)
        figure("cv_fig")
        plot(temps, u_cv_s_m[1,:], label = N)
        figure("s_fig")
        plot(temps, u_cv_s_m[2,:], label = N)
        figure("m_fig")
        plot(temps, u_cv_s_m[3,:], label = N)

    figure("u_fig")
    xlabel("$T$")
    ylabel("$U/N$")
    legend(loc="best")
    tight_layout()
    savefig(fig_dir+"u-convergence.pdf")

    figure("cv_fig")
    xlabel("$T$")
    ylabel("$C_V/N$")
    legend(loc="best")
    tight_layout()
    savefig(fig_dir+"cv-convergence.pdf")

    figure("s_fig")
    xlabel("$T$")
    ylabel("$S/N$")
    legend(loc="best")
    tight_layout()
    savefig(fig_dir+"s-convergence.pdf")

    figure("m_fig")
    xlabel("$T$")
    ylabel("$m$")
    legend(loc="best")
    tight_layout()
    savefig(fig_dir+"m-convergence.pdf")

######### 2D plots ##########

def borders(array):
    borders = zeros(len(array)+1)
    borders[1:-1] = (array[1:] + array[:-1]) / 2
    borders[0] = array[0] - (array[1] - array[0]) / 2
    borders[-1] = array[-1] + (array[-1] - array[-2]) / 2
    return borders

if "2d" in sys.argv:
    N = N2D

    p_vals = array(sorted(files[N].keys()))
    u_cv_s_m = zeros((4,temp_steps,len(p_vals)))
    for ii in range(len(p_vals)):
        u_cv_s_m[:,:,ii] = mean_u_cv_s_m(N, p_vals[ii])

    p_borders = borders(p_vals)
    temp_borders = borders(temps)

    figure("u_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_m[0,:,:])
    title("$U/N$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig(fig_dir+"u-phase.pdf")

    figure("cv_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_m[1,:,:])
    title("$C_V/N$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig(fig_dir+"cv-phase.pdf")

    figure("s_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_m[2,:,:])
    title("$S/N$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig(fig_dir+"s-phase.pdf")

    figure("m_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_m[3,:,:])
    title("$m$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig(fig_dir+"d-phase.pdf")

if "show" in sys.argv:
    show()
