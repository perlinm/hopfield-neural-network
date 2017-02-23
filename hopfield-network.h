#pragma once
#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness

#include <eigen3/Eigen/Dense> // linear algebra library

using namespace std;
using namespace Eigen;

// generate random state
vector<bool> random_state(const uint nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator);

// generate interaction matrix from patterns
// note: these couplings are a factor of [nodes] greater than the regular definition
MatrixXi get_couplings(vector<vector<bool>>& patterns);

// get maximum energy change possible in one spin flip with given interaction matrix
uint get_max_energy_change(const MatrixXi& couplings);

struct hopfield_network {

  const uint nodes;
  const vector<vector<bool>> patterns;
  const MatrixXi couplings;

  const uint max_energy;
  const uint max_energy_change;

  hopfield_network(const vector<vector<bool>>& patterns);

  void print_patterns() const;
  void print_couplings() const;

};

// network simulation object
struct network_simulation {

  const hopfield_network network;
  const double min_temperature;
  const uint probability_factor;

  vector<bool> state;

  MatrixXi transition_matrix;
  vector<uint> energy_histogram;
  vector<vector<uint>> state_histogram;

  network_simulation(const vector<vector<bool>>& patterns,
                     const vector<bool>& initial_state,
                     const double min_temperature,
                     const uint probability_factor);

  // energy of the network in a given state
  // note: this energy is a factor of [2*nodes] greater than the regular definition
  int energy(const vector<bool>& state);
  int energy() { return energy(state); };

  // initialize all histograms with zeros
  void initialize_histograms();

  // update histograms with current state
  void update_histograms();

  // move to a new state and update histograms
  void move(const vector<bool>& new_state);

  // number of transitions from a given energy with a specified energy change
  uint transitions(const int energy, const int energy_change) const {
    return transition_matrix(energy + network.max_energy,
                             energy_change + network.max_energy_change);
  }

  // observations of a given energy
  uint energy_observations(const int energy) const {
    return energy_histogram.at(energy + network.max_energy);
  }

  // print a given network state
  void print_state(const vector<bool>& state) const;
  void print_state() const { return print_state(state); };

};
