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

  void print_patterns() const {
    for (uint ii = 0; ii < patterns.size(); ii++) {
      for (uint jj = 0; jj < nodes; jj++) {
        cout << patterns.at(ii).at(jj) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

   void print_couplings() const {
    const uint width = log10(couplings.array().abs().maxCoeff()) + 2;
    for (uint ii = 0; ii < nodes; ii++) {
      for (uint jj = 0; jj < nodes; jj++) {
        cout << setw(width) << couplings(ii,jj) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

};

// network simulation object
struct network_simulation {

  const hopfield_network network;
  const uint probability_factor;

  vector<bool> state;
  vector<uint> energy_histogram;
  MatrixXi transition_matrix;

  network_simulation(const vector<vector<bool>>& patterns,
                     const vector<bool>& initial_state,
                     const uint probability_factor);

  // energy of network in its current state
  // note: this energy is a factor of [2*nodes] greater than the regular definition
  int energy(vector<bool>& state);
  int energy() { return energy(state); };

  // observations of a given energy
  uint energy_observations(const int energy) const {
    return energy_histogram.at(energy + network.max_energy);
  }

  // transitions from given energy with specified energy change
  uint transitions(const int energy, const int energy_change) const {
    return transition_matrix(energy + network.max_energy,
                             energy_change + network.max_energy_change);
  }

  // print current state of network
  void print_state() const {
    for (uint ii = 0; ii < state.size(); ii++) {
      cout << state.at(ii) << " ";
    }
    cout << endl << endl;
  }

};
