#pragma once

#include <random> // for randomness

using namespace std;

// greatest common divisor
int gcd(const int a, const int b);

// distance between two states
int state_distance(const vector<bool>& s1, const vector<bool>& s2);

// generate random state
vector<bool> random_state(const int nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator);

// make a random change to a given state using a random number on [0,1)
vector<bool> random_change(const vector<bool>& state, const double random);

struct hopfield_network {

  int nodes;
  vector<vector<int>> couplings;

  int energy_scale;
  int max_energy;
  int max_energy_change;

  hopfield_network(const vector<vector<bool>>& patterns);

  // energy of the network in a given state
  // note: this energy is a factor of (nodes/energy_scale) greater
  //   than the regular definition, in addition to being shifted up a bit
  int energy(const vector<bool>& state) const;

  void print_couplings() const;

};

// network simulation object
struct network_simulation {

  const vector<vector<bool>> patterns;
  const hopfield_network network;
  const int energy_range;
  const int max_de;

  // energy at which entropy is maximized
  int entropy_peak;

  vector<bool> state;
  vector<unsigned long> energy_histogram;
  vector<vector<unsigned long>> state_histograms;
  vector<vector<unsigned long>> distance_histograms;

  // the transition matrix tells us how many times we have moved
  //   from a given energy with a specified energy difference
  vector<vector<unsigned long>> energy_transitions;

  // have we visited this (negative) energy at least once since
  //   the last observation of states with energy >= 0?
  vector<bool> visit_log;
  // number of independent samples of a given (negative) energy;
  // two samples of a given energy are considered independent if
  //   we have visited states with energy >= 0 between the samples
  vector<unsigned long> sample_histogram;

  // logarithm of weights determining the transition probability
  //   between energies during simulation
  vector<double> ln_weights;

  // logarithm of density of states
  vector<double> ln_dos;

  network_simulation(const vector<vector<bool>>& patterns,
                     const vector<bool>& initial_state);

  // -------------------------------------------------------------------------------------
  // Access methods for histograms and matrices
  // -------------------------------------------------------------------------------------

  // number of transitions from a given energy with a specified energy change
  int transitions(const int energy, const int energy_change) const;

  // number of transitions from a given energy to any other energy
  int transitions_from(const int energy) const;

  // actual transition matrix
  double transition_matrix(const int to, const int from) const;

  // -------------------------------------------------------------------------------------
  // Methods used in simulation
  // -------------------------------------------------------------------------------------

  // energy of a given state
  int energy(const vector<bool>& state) const { return network.energy(state); };
  int energy() const { return energy(state); };

  // initialize all histograms:
  //   energy histogram, sample histogram, state histograms, energy transition
  void initialize_histograms();

  // update histograms with an energy observation (presumably of the current state)
  void update_energy_histogram(const int energy);
  void update_state_histograms(const int energy);
  void update_distance_histograms(const int energy);

  // update sample histogram
  void update_sample_histogram(const int new_energy, const int old_energy);

  // update transition matrix
  void add_transition(const int energy, const int energy_change);

  // expectation value of fractional sample error at a given inverse temperature
  // WARNING: assumes that the density of states is up to date
  double fractional_sample_error(const double beta) const;

  // compute density of states and weight array from transition matrix
  void compute_dos_and_weights_from_transitions(const double beta);

  // compute density of states from the energy histogram
  void compute_dos_from_energy_histogram();

  // -------------------------------------------------------------------------------------
  // Printing methods
  // -------------------------------------------------------------------------------------

  // print simulation patterns
  void print_patterns() const;

  // print energy histogram, sample histogram, and density of states
  void print_energy_data() const;

  // print expectation value of each spin spin at each energy
  void print_expected_states() const;

  // print expectation value of distances from each pattern at each energy
  void print_distances() const;
};
