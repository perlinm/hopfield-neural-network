#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness
#include <fstream> // for file input

#include <boost/filesystem.hpp> // filesystem path manipulation library
#include <boost/program_options.hpp> // options parsing library

#include <eigen3/Eigen/Dense> // linear algebra library

#include "methods.h"

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
  bool debug;

  po::options_description general("General options", help_text_length);
  general.add_options()
    ("help,h", "produce help message")
    ("seed", po::value<unsigned long long int>(&seed)->default_value(0),
     "seed for random number generator")
    ("debug", po::value<bool>(&debug)->default_value(false)->implicit_value(true),
     "enable debug mode")
    ;

  uint nodes = 0;
  uint pattern_number = 0;
  string pattern_file;

  po::options_description network_parameters("Network parameters",help_text_length);
  network_parameters.add_options()
    ("nodes", po::value<uint>(&nodes), "number of nodes which make up the network")
    ("patterns", po::value<uint>(&pattern_number), "number of random patterns to use")
    ("pattern_file", po::value<string>(&pattern_file),
     "input file containing patterns stored in the neural network")
    ;


  uint probability_factor;
  double min_temperature;

  po::options_description simulation_options("Simulation options",help_text_length);
  simulation_options.add_options()
    ("min_temp", po::value<double>(&min_temperature)->default_value(0.01),
     "minimum temperature of interest")
    ("probability_factor", po::value<uint>(&probability_factor)->default_value(16),
     "fudge factor in computation of move acceptance probability from transition matrix")
    ;

  po::options_description all("Allowed options");
  all.add(general);
  all.add(network_parameters);
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

  // by default, use the same number of patterns as there are nodes
  if (nodes && !pattern_number) pattern_number = nodes;

  // if we specified a pattern file, make sure it exists
  assert(pattern_file.empty() || fs::exists(pattern_file));

  // if we did not specify anything, use a default pattern file
  if (!nodes && pattern_file.empty()) pattern_file = default_pattern_file;

  // determine whether we are using a pattern file afterall
  const bool using_pattern_file = !pattern_file.empty();

  // initialize random number generator
  uniform_real_distribution<double> rnd(0.0,1.0); // uniform distribution on [0,1)
  mt19937_64 generator(seed); // use and seed the 64-bit Mersenne Twister 19937 generator

  // -------------------------------------------------------------------------------------
  // Construct patterns for network and initialize network simulation
  // -------------------------------------------------------------------------------------

  vector<vector<bool>> patterns;

  // if we are using a pattern file, read it in
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

  } else { // if we are not using a pattern file, generate random patterns

    for (uint ii = 0; ii < pattern_number; ii++) {
      patterns.push_back(random_state(nodes, rnd, generator));
    }

  }

  for (uint ii = 1; ii < patterns.size(); ii++) {
    if (patterns.at(ii-1).size() != patterns.at(ii).size()){
      cout << "patterns " << ii-1 << " and " << ii
           << " do not have the same size!" << endl;
      return -1;
    }
  }
  pattern_number = patterns.size();
  nodes = patterns.at(0).size();

  network_simulation ns(patterns, random_state(nodes, rnd, generator),
                        min_temperature, probability_factor);

  cout << endl
       << "maximum energy: " << ns.network.max_energy
       << " (of " << pattern_number * nodes * (nodes - 1) / ns.network.energy_scale
       << " possible)" << endl
       << "maximum energy change: " << ns.network.max_energy_change
       << " (of " << 2 * pattern_number * (nodes - 1) / ns.network.energy_scale
       << " possible)" << endl
       << endl;

  if (debug) {
    ns.print_patterns();
    cout << endl;
    ns.network.print_couplings();
    cout << endl;
  }


  cout << "max energy change: " << ns.network.max_energy_change << endl;
  cout << "energy scale: " << ns.network.energy_scale << endl;

}
