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
  const double min_temperature;
  const uint probability_factor;

  vector<bool> state;

  MatrixXi transition_matrix;
  vector<double> weights;

  vector<uint> energy_histogram;
  vector<vector<uint>> state_histogram;

  network_simulation(const vector<vector<bool>>& patterns,
                     const vector<bool>& initial_state,
                     const double min_temperature,
                     const uint probability_factor);

  // initialize all histograms with zeros
  void initialize_histograms();

  // update histograms with current state
  void update_histograms();

  // move to a new state and update histograms
  void move(const vector<bool>& new_state);

  // number of transitions from a given energy with a specified energy change
  uint transitions(const int energy, const int energy_change) const;

  // observations of a given energy
  uint energy_observations(const int energy) const;

  // print simulation patterns  // print network patterns or state
  void print_patterns() const;

  // print a given network state
  void print_state(const vector<bool>& state) const;
  void print_state() const { return print_state(state); };

};
