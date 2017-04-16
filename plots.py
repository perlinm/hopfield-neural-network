#!/usr/bin/env python3

import glob, sys
from pylab import *

# some plot options
max_B = 20
max_temp = 2
temp_steps = 100
temps = array(linspace(1/max_B, max_temp, temp_steps))

# dictionary keys for energy, weight, and distance files
E = "E"
W = "W"
D = "D"

# get N (number of nodes), P (number of patterns), and B (maximal inverse temperature)
def NPB(file_name):
    if type(file_name) == dict: file_name = file_name[E]
    parts = file_name.split("-")
    N = int(parts[1][1:])
    P = int(parts[2][1:])
    B = int(parts[3][1:])
    return N, P, B

# identify and organize data files
files = {}
energy_files = sorted(glob.glob("./data/energies-*"))
weight_files = sorted(glob.glob("./data/weights-*"))
distance_files = sorted(glob.glob("./data/distances-*"))
for energy_file in energy_files:
    weight_matches = [ weight_file for weight_file in weight_files
                       if weight_file.split("-")[1:] == energy_file.split("-")[1:] ]
    distance_matches = [ distance_file for distance_file in distance_files
                         if distance_file.split("-")[1:] == energy_file.split("-")[1:] ]
    if len(weight_matches) != 1:
        # print("problem matching weight file:", energy_file)
        continue
    if len(distance_matches) != 1:
        # print("problem matching distance file:", energy_file)
        continue
    weight_file = weight_matches[0]
    distance_file = distance_matches[0]

    N, P, _ = NPB(energy_file)
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
def U_CV_S_D(file_set):
    P = NPB(file_set)[1]
    energies_hist, hist_input, _ = loadtxt(file_set[E], unpack = True)
    energies_weights, ln_weights_input = loadtxt(file_set[W], unpack = True)
    dist_data = loadtxt(file_set[D])

    energies = array([ e for e in energies_hist
                       if (e in energies_weights and e in dist_data[:,0]) ])
    hist = array([ hist_input[ii] for ii in range(len(hist_input))
                   if energies_hist[ii] in energies ])
    ln_weights = array([ ln_weights_input[ii] for ii in range(len(ln_weights_input))
                         if energies_weights[ii] in energies ])
    samples = array([ dist_data[ii,1] for ii in range(shape(dist_data)[0])
                      if dist_data[ii,0] in energies ])
    distance_sums = array([ dist_data[ii,2:] for ii in range(shape(dist_data)[0])
                            if dist_data[ii,0] in energies ])

    ln_dos = log(hist) - ln_weights

    # correct for the factor of N in the definition of energy in the simulations
    energies /= NPB(file_set)[0]

    U_CV_S_D = zeros((4,temp_steps))
    for ii in range(temp_steps):
        ln_dos_boltz = ln_dos - energies/temps[ii]
        dos_boltz = exp(ln_dos_boltz - ln_dos_boltz.max())
        Z = sum(dos_boltz)

        # U = < E >
        U_CV_S_D[0,ii] = sum(energies*dos_boltz)/Z
        # CV = < E^2 > - < E >^2
        U_CV_S_D[1,ii] = sum((energies/temps[ii])**2 * dos_boltz)/Z \
                         - (sum(energies/temps[ii] * dos_boltz)/Z)**2
        # S = < E/T + ln(Z) >
        U_CV_S_D[2,ii] = sum(dos_boltz*(energies/temps[ii])) / Z \
                         + ln_dos_boltz.max() + log(Z)

        pattern_Ds = zeros(P)
        for p in range(P):
            pattern_Ds[p] = sum(distance_sums[:,p]/samples*dos_boltz) / Z
        U_CV_S_D[3,ii] = min(pattern_Ds)


    # subtract off the asymptotic value of S(T->\infty) from S(T) in order to
    #   "standardize" our measure of entropy across different simulations
    Z_inf = sum(exp(ln_dos - ln_dos.max()))
    S_inf = ln_dos.max() + log(Z_inf)
    U_CV_S_D[2,:] -= S_inf

    return U_CV_S_D

def mean_u_cv_s_d(N, P):
    u_cv_s_d = zeros((4,temp_steps))
    for file_set in files[N][P]:
        if NPB(file_set)[2] < max_B: continue
        u_cv_s_d += U_CV_S_D(file_set)
    u_cv_s_d /= len(files[N][P]) * N
    return u_cv_s_d

######### convergence (1D) plots ##########

if "1d" in sys.argv:
    alpha = 0.1

    for N in sorted(files.keys()):
        P = N * alpha
        if P not in files[N].keys(): continue

        u_cv_s_d = mean_u_cv_s_d(N, P)

        figure("u_fig")
        plot(temps, u_cv_s_d[0,:], label = N)
        figure("cv_fig")
        plot(temps, u_cv_s_d[1,:], label = N)
        figure("s_fig")
        plot(temps, u_cv_s_d[2,:], label = N)
        figure("d_fig")
        plot(temps, u_cv_s_d[3,:], label = N)

    figure("u_fig")
    xlabel("$T$")
    ylabel("$U/N$")
    legend(loc="best")
    tight_layout()
    savefig("u_convergence.pdf")

    figure("cv_fig")
    xlabel("$T$")
    ylabel("$C_V/N$")
    legend(loc="best")
    tight_layout()
    savefig("cv_convergence.pdf")

    figure("s_fig")
    xlabel("$T$")
    ylabel("$S/N$")
    legend(loc="best")
    tight_layout()
    savefig("s_convergence.pdf")

    figure("d_fig")
    xlabel("$T$")
    ylabel("$D/N$")
    legend(loc="best")
    tight_layout()
    savefig("d_convergence.pdf")

######### 2D plots ##########

def borders(array):
    borders = zeros(len(array)+1)
    borders[1:-1] = (array[1:] + array[:-1]) / 2
    borders[0] = array[0] - (array[1] - array[0]) / 2
    borders[-1] = array[-1] + (array[-1] - array[-2]) / 2
    return borders

if "2d" in sys.argv:
    N = 100

    p_vals = array(sorted(files[N].keys()))
    u_cv_s_d = zeros((4,temp_steps,len(p_vals)))
    for ii in range(len(p_vals)):
        u_cv_s_d[:,:,ii] = mean_u_cv_s_d(N, p_vals[ii])

    p_borders = borders(p_vals)
    temp_borders = borders(temps)

    figure("u_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_d[0,:,:])
    title("$U$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig("u_color.pdf")

    figure("cv_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_d[1,:,:])
    title("$C_V$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig("cv_color.pdf")

    figure("s_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_d[2,:,:])
    title("$S$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig("s_color.pdf")

    figure("d_2d_fig")
    pcolor(p_borders/N, temp_borders, u_cv_s_d[3,:,:])
    title("$D/N$")
    xlabel("$P/N$")
    ylabel("$T$")
    colorbar()
    tight_layout()
    savefig("d_color.pdf")

if "show" in sys.argv:
    show()
