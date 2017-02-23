#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <fstream> // for file input
#include <iomanip> // some nice printing functions
#include <random> // for randomness

#include <boost/filesystem.hpp> // filesystem path manipulation library
#include <boost/program_options.hpp> // options parsing library

#include <eigen3/Eigen/Dense> // linear algebra library

#include "hopfield-network.h"

using namespace std;
using namespace Eigen;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(const int arg_num, const char *arg_vec[]) {

  // -------------------------------------------------------------------------------------
  // Define some constants
  // -------------------------------------------------------------------------------------

  const uint help_text_length = 85;

  const string default_pattern_file = "default-patterns.txt";

  // -------------------------------------------------------------------------------------
  // Set input options
  // -------------------------------------------------------------------------------------

  unsigned long long int seed;
  bool testing_mode;

  po::options_description general("General options", help_text_length);
  general.add_options()
    ("help,h", "produce help message")
    ("seed", po::value<unsigned long long int>(&seed)->default_value(0),
     "seed for random number generator")
    ("test" ,po::value<bool>(&testing_mode)->default_value(false)->implicit_value(true),
     "enable testing mode")
    ;

  string pattern_file;
  int nodes = 0;
  int random = 0;

  po::options_description simulation_options("Simulation options",help_text_length);
  simulation_options.add_options()
    ("patterns", po::value<string>(&pattern_file),
     "input file containing patterns stored in the neural network")
    ("nodes", po::value<int>(&nodes), "number of nodes which make up the network")
    ("random", po::value<int>(&random), "number of random patterns to use")
    ;

  po::options_description all("Allowed options");
  all.add(general);
  all.add(simulation_options);

  // collect inputs
  po::variables_map inputs;
  po::store(parse_command_line(arg_num, arg_vec, all), inputs);
  po::notify(inputs);

  // if requested, print help text
  if (inputs.count("help")) {
    cout << all;
    return 0;
  }

  // -------------------------------------------------------------------------------------
  // Process and run sanity checks on inputs
  // -------------------------------------------------------------------------------------

  // we should not have a negative number of nodes
  assert(nodes >= 0);

  // ... or random patterns
  assert(!(nodes && random < 0));

  // by default, use the same number of patterns as there are nodes
  if (nodes && !random) random = nodes;

  // if we specified a pattern file, make sure it exists
  assert(pattern_file.empty() || fs::exists(pattern_file));

  // if we did not specify anything, use a default pattern file
  if (!nodes && pattern_file.empty()) pattern_file = default_pattern_file;
  const bool using_pattern_file = !pattern_file.empty();

  // initialize random number generator
  uniform_real_distribution<double> rnd(0.0,1.0); // uniform distribution on [0,1)
  mt19937_64 generator(seed); // use and seed the 64-bit Mersenne Twister 19937 generator

  // -------------------------------------------------------------------------------------
  // Construct the neural network
  // -------------------------------------------------------------------------------------

  vector<vector<bool>> patterns;

  if (using_pattern_file) {
    ifstream input(pattern_file);
    string line;

    while (getline(input,line)) {
      vector<bool> pattern = {};
      for (uint ii = 0; ii < line.length(); ii++) {
        if (line.at(ii) == '1') pattern.push_back(true);
        if (line.at(ii) == '0') pattern.push_back(false);
      }
      patterns.push_back(pattern);
    }

  }

  hopfield_network network(patterns);

  for (uint ii = 0; ii < network.patterns.size(); ii++) {
    for (uint jj = 0; jj < network.patterns.at(ii).size(); jj++) {
      cout << network.patterns.at(ii).at(jj) << " ";
    }
    cout << endl;
  }

}
