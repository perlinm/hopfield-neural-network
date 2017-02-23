#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <random> // for randomness

#include <eigen3/Eigen/Dense> // linear algebra library

#include "hopfield-network.h"

using namespace std;
using namespace Eigen;

// generate random state
vector<bool> random_state(const uint nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator) {
  vector<bool> state(nodes);
  for (uint ii = 0; ii < nodes; ii++) {
    state.at(ii) = (rnd(generator) < 0.5);
  }
  return state;
}

// generate coupling matrix from patterns
// note: these couplings are a factor of [nodes] greater than the regular definition
MatrixXi get_coupling_matrix(const vector<vector<bool>>& patterns) {
  const uint nodes = patterns.at(0).size();
  MatrixXi coupling = MatrixXi::Zero(nodes,nodes);

  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      for (uint pp = 0; pp < patterns.size(); pp++) {
        coupling(ii,jj) += (2*patterns.at(pp).at(ii)-1)*(2*patterns.at(pp).at(jj)-1);
      }
    }
    coupling(ii,ii) = 0;
  }

  return coupling;
}

// get maximum energy change possible in one spin flip with given interaction matrix
uint get_max_energy_change(const MatrixXi& couplings){
  int max_change = 0;
  for (uint ii = 0; ii < couplings.rows(); ii++) {
    max_change = max(couplings.row(ii).array().abs().sum(), max_change);
  }
  return max_change;
}

// hopfield network constructor
hopfield_network::hopfield_network(const vector<vector<bool>>& patterns) :
  nodes(patterns.at(0).size()),
  patterns(patterns),
  couplings(get_coupling_matrix(patterns)),
  max_energy(couplings.array().abs().sum()),
  max_energy_change(get_max_energy_change(couplings))
{};

void hopfield_network::print_patterns() const {
  for (uint ii = 0; ii < patterns.size(); ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      cout << patterns.at(ii).at(jj) << " ";
    }
    cout << endl;
  }
  cout << endl;
}
void hopfield_network::print_couplings() const {
  const uint width = log10(couplings.array().abs().maxCoeff()) + 2;
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      cout << setw(width) << couplings(ii,jj) << " ";
    }
    cout << endl;
  }
  cout << endl;
}

// network simulation constructor
network_simulation::network_simulation(const vector<vector<bool>>& patterns,
                                       const vector<bool>& initial_state,
                                       const double min_temperature,
                                       const uint probability_factor) :
  network(hopfield_network(patterns)),
  min_temperature(min_temperature),
  probability_factor(probability_factor)
{
  state = initial_state;
  transition_matrix = MatrixXi::Zero(2*network.max_energy + 1,
                                     2*network.max_energy_change + 1);
  initialize_histograms();
};

// energy of network in its current state
// note: this energy is a factor of [2*nodes] greater than the regular definition
int network_simulation::energy(const vector<bool>& state) {
  int sum = 0;
  for (uint ii = 0; ii < network.nodes; ii++) {
    for (uint jj = 0; jj < network.nodes; jj++) {
      sum += network.couplings(ii,jj) * (2*state.at(ii)-1) * (2*state.at(jj)-1);
    }
  }
  return -sum;
}

// initialize all histograms with zeros
void network_simulation::initialize_histograms() {
  const uint energy_range = 2*network.max_energy + 1;
  energy_histogram = vector<uint>(energy_range);
  state_histogram = vector<vector<uint>>(network.nodes);
  for (uint ii = 0; ii < network.nodes; ii++) {
    state_histogram.at(ii) = vector<uint>(energy_range);
  }
}

// update histograms with current state
void network_simulation::update_histograms() {
  const uint energy_index = energy() + network.max_energy;
  energy_histogram.at(energy_index)++;
  for (uint ii = 0; ii < network.nodes; ii++) {
    state_histogram.at(ii).at(energy_index) += state.at(ii);
  }
}

// move to a new state and update histograms
void network_simulation::move(const vector<bool>& new_state) {
  state = new_state;
  update_histograms();
}

// print a given network state
void network_simulation::print_state(const vector<bool>& state) const {
  for (uint ii = 0; ii < state.size(); ii++) {
    cout << state.at(ii) << " ";
  }
  cout << endl << endl;
}
