#pragma once
#define EIGEN_USE_MKL_ALL

#include <random> // for randomness

#include <eigen3/Eigen/Dense> // linear algebra library

using namespace std;
using namespace Eigen;

// generate random state
vector<bool> random_state(const uint nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator);

// generate coupling matrix from patterns
// note: these couplings are a factor of [nodes] greater than the regular definition
MatrixXi get_couplings(vector<vector<bool>> patterns);

// hopfield network object
struct hopfield_network{

  const vector<vector<bool>> patterns;
  const MatrixXi coupling;
  const uint nodes;

  hopfield_network(const vector<vector<bool>>& patterns,
                   const vector<bool>& initial_state);

  vector<bool> state;

  // energy of network in its current state
  // note: this energy is a factor of [2*nodes] greater than the regular definition
  int energy(){
    double sum;
    for (uint ii = 0; ii < nodes; ii++) {
      for (uint jj = 0; jj < nodes; jj++) {
        sum += coupling(ii,jj) * (2*state.at(ii)-1) * (2*state.at(jj)-1);
      }
    }
    return -sum;
  }

};
