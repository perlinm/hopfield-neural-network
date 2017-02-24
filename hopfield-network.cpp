#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness

#include <eigen3/Eigen/Dense> // linear algebra library

#include "hopfield-network.h"

using namespace std;
using namespace Eigen;

// greatest common divisor
int gcd(const int a, const int b) {
  if (b == 0) return a;
  else return gcd(b, a % b);
}

// generate random state
vector<bool> random_state(const uint nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator) {
  vector<bool> state(nodes);
  for (uint ii = 0; ii < nodes; ii++) {
    state.at(ii) = (rnd(generator) < 0.5);
  }
  return state;
}

// make a random change to a given state using a random number on [0,1)
vector<bool> random_change(const vector<bool>& state, const double random) {
  const uint node = floor(random*state.size());
  vector<bool> new_state = state;
  new_state.at(node) = !new_state.at(node);
  return new_state;
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

// hopfield network constructor
hopfield_network::hopfield_network(const vector<vector<bool>>& patterns) {
  nodes = patterns.at(0).size();

  // generate interaction matrix from patterns
  // note: these couplings are a factor of [nodes] greater than the regular definition
  couplings = MatrixXi::Zero(nodes,nodes);
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      for (uint pp = 0; pp < patterns.size(); pp++) {
        couplings(ii,jj) += (2*patterns.at(pp).at(ii)-1)*(2*patterns.at(pp).at(jj)-1);
      }
    }
    couplings(ii,ii) = 0;
  }

  max_energy = couplings.array().abs().sum();

  max_energy_change = 0;
  energy_scale = max_energy;
  for (uint ii = 0; ii < nodes; ii++) {
    const uint energy_change = couplings.row(ii).array().abs().sum();
    max_energy_change = max(energy_change, max_energy_change);
    energy_scale = gcd(energy_change, energy_scale);
  }

  max_energy /= energy_scale;
  max_energy_change /= energy_scale;
};

// energy of the network in a given state
// note: this energy is a factor of [nodes/energy_scale] greater
//       than the regular definition
int hopfield_network::energy(const vector<bool>& state) const {
  int sum = 0;
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = ii+1; jj < nodes; jj++) {
      sum += couplings(ii,jj) * (2*state.at(ii)-1) * (2*state.at(jj)-1);
    }
  }
  return -sum/int(energy_scale);
}


void hopfield_network::print_couplings() const {
  const uint width = log10(couplings.array().abs().maxCoeff()) + 2;
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      cout << setw(width) << couplings(ii,jj) << " ";
    }
    cout << endl;
  }
}

// network simulation constructor
network_simulation::network_simulation(const vector<vector<bool>>& patterns,
                                       const vector<bool>& initial_state,
                                       const double min_temperature,
                                       const uint probability_factor) :
  patterns(patterns),
  network(hopfield_network(patterns)),
  min_temperature(min_temperature),
  probability_factor(probability_factor)
{
  state = initial_state;
  transition_matrix = MatrixXi::Zero(2*network.max_energy + 1,
                                     2*network.max_energy_change + 1);
  weights = vector<double>(2*network.max_energy + 1, 1);
  initialize_histograms();
};

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
  const uint energy_index = network.energy(state) + network.max_energy;
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

// number of transitions from a given energy with a specified energy change
uint network_simulation::transitions(const int energy, const int energy_change) const {
  return transition_matrix(energy + network.max_energy,
                           energy_change + network.max_energy_change);
}

// observations of a given energy
uint network_simulation::energy_observations(const int energy) const {
  return energy_histogram.at(energy + network.max_energy);
}

// print simulation patterns
void network_simulation::print_patterns() const {
  for (uint ii = 0; ii < patterns.size(); ii++) {
    for (uint jj = 0; jj < network.nodes; jj++) {
      cout << patterns.at(ii).at(jj) << " ";
    }
    cout << endl;
  }
}

// print a given network state
void network_simulation::print_state(const vector<bool>& state) const {
  for (uint ii = 0; ii < state.size(); ii++) {
    cout << state.at(ii) << " ";
  }
  cout << endl;
}
