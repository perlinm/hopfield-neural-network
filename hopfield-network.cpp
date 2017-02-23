#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <eigen3/Eigen/Dense> // linear algebra library

#include "hopfield-network.h"

using namespace std;
using namespace Eigen;

// generate coupling matrix from patterns
MatrixXd get_coupling_matrix(std::vector<std::vector<bool>> patterns) {
  const uint nodes = patterns.at(0).size();
  MatrixXd correlation(nodes,nodes);

  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      for (uint pp = 0; pp < patterns.size(); pp++) {
        correlation(ii,jj) += (2*patterns.at(pp).at(ii)-1)*(2*patterns.at(pp).at(jj)-1);
      }
    }
  }

  return correlation/nodes;
}

hopfield_network:: hopfield_network(const vector<vector<bool>>& patterns) :
  patterns(patterns),
  coupling(get_coupling_matrix(patterns)),
  nodes(patterns.at(0).size())
{};
