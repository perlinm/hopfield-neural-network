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
  vector<bool> state;
  for (uint ii = 0; ii < nodes; ii++) {
    state.push_back(rnd(generator) < 0.5);
  }
  return state;
}

// generate coupling matrix from patterns
// note: these couplings are a factor of [nodes] greater than the regular definition
MatrixXi get_coupling_matrix(std::vector<std::vector<bool>> patterns) {
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
hopfield_network::hopfield_network(const vector<vector<bool>>& patterns,
                                   const vector<bool>& initial_state) :
  patterns(patterns),
  coupling(get_coupling_matrix(patterns)),
  nodes(patterns.at(0).size())
{
  state = initial_state;
};

// energy of network in its current state
// note: this energy is a factor of [2*nodes] greater than the regular definition
int hopfield_network::energy(vector<bool>& state) {
  int sum = 0;
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      sum += coupling(ii,jj) * (2*state.at(ii)-1) * (2*state.at(jj)-1);
    }
  }
  return -sum;
}
