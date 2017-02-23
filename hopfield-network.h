#pragma once
#define EIGEN_USE_MKL_ALL

#include <vector>
#include <eigen3/Eigen/Dense> // linear algebra library

// hopfield network object
struct hopfield_network{

  const std::vector<std::vector<bool>> patterns;
  const uint nodes;

  std::vector<bool> state;

  hopfield_network(const std::vector<std::vector<bool>>& patterns);


};
