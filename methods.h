#pragma once
#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness

#include <eigen3/Eigen/Dense> // linear algebra library

using namespace std;
using namespace Eigen;

// greatest common divisor
int gcd(const int a, const int b);

// generate random state
vector<bool> random_state(const uint nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator);

// make a random change to a given state using a random number on [0,1)
vector<bool> random_change(const vector<bool>& state, const double random);

struct hopfield_network {

  uint nodes;
  MatrixXi couplings;
  uint max_energy;
  uint max_energy_change;
  uint energy_scale;
  uint energy_range;

  hopfield_network(const vector<vector<bool>>& patterns);

  // energy of the network in a given state
  // note: this energy is a factor of [nodes/energy_scale] greater
  //       than the regular definition
  int energy(const vector<bool>& state) const;

  void print_couplings() const;

};

// network simulation object
struct network_simulation {

  const vector<vector<bool>> patterns;
  const hopfield_network network;

  vector<bool> state;
  vector<uint> energy_histogram;
  vector<vector<uint>> state_histogram;

  // the transition matrix tells us how many times we have moved
  //   from a given energy E with a specified energy difference dE
  MatrixXi energy_transitions;

  // have we visited this (negative) energy at least once since
  //   the last observation of states with energy >= 0?
  vector<bool> visited;
  // number of independent samples of a given (negative) energy;
  // two samples of a given energy are considered independent if
  //   we have visited states with energy >= 0 between the samples
  vector<uint> samples;

  // logarithm of density of states
  vector<double> ln_dos;

  // logarithm of weights determining the transition probability
  //   between energies during simulation
  vector<double> ln_weights;

  network_simulation(const vector<vector<bool>>& patterns,
                     const vector<bool>& initial_state);

  // -------------------------------------------------------------------------------------
  // Methods used in simulation
  // -------------------------------------------------------------------------------------

  // energy of a given state
  int energy(const vector<bool>& state) const;
  int energy() const { return energy(state); };

  // initialize all histograms with zeros
  void initialize_histograms();

  // update histograms with an observation of the current state
  void update_histograms();

  // update sample count
  void update_samples(const int new_energy, const int old_energy);

  // reset tally of energies visited since the last observation of states with energy >= 0
  void reset_visit_log();

  // add to transition matrix
  void add_transition(const int energy, const int energy_change);

  // compute density of states from transition matrix
  void compute_dos_and_weights();

  // -------------------------------------------------------------------------------------
  // Access methods for histograms and matrices
  // -------------------------------------------------------------------------------------

  // observations of a given energy
  uint energy_observations(const int energy) const;

  // number of transitions from a given energy with a specified energy change
  uint transitions(const int energy, const int energy_change) const;

  // number of transitions from a given energy to any other energy
  uint transitions_from(const int energy) const;

  // actual transition matrix
  double transition_matrix(const int to, const int from) const;

  // -------------------------------------------------------------------------------------
  // Printing methods
  // -------------------------------------------------------------------------------------

  // print simulation patterns  // print network patterns or state
  void print_patterns() const;

  // print a given network state
  void print_state(const vector<bool>& state) const;
  void print_state() const { return print_state(state); };

};
