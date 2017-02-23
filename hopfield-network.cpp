#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <eigen3/Eigen/Dense> // linear algebra library

#include "hopfield-network.h"

using namespace std;
using namespace Eigen;

hopfield_network:: hopfield_network(const vector<vector<bool>>& patterns) :
  patterns(patterns), nodes(patterns.at(0).size())
{};
