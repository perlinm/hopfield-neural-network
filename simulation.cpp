#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness
#include <fstream> // for file input

#include <boost/filesystem.hpp> // filesystem path manipulation library
#include <boost/program_options.hpp> // options parsing library

#include "methods.h"

using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(const int arg_num, const char *arg_vec[]) {

  // -------------------------------------------------------------------------------------
  // Set input options
  // -------------------------------------------------------------------------------------

  const uint help_text_length = 85;

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

  uint nodes;
  uint pattern_number;
  string pattern_file;

  po::options_description network_parameters("Network parameters",help_text_length);
  network_parameters.add_options()
    ("nodes", po::value<uint>(&nodes)->default_value(10),
     "number of nodes which make up the network")
    ("patterns", po::value<uint>(&pattern_number)->default_value(10),
     "number of random patterns to use")
    ("pattern_file", po::value<string>(&pattern_file),
     "input file containing patterns stored in the neural network")
    ;


  double min_temp;
  uint tpf; // "transition probability factor"

  po::options_description simulation_options("Simulation options",help_text_length);
  simulation_options.add_options()
    ("min_temp", po::value<double>(&min_temp)->default_value(0.01,"0.01"),
     "minimum temperature of interest")
    ("probability_factor", po::value<uint>(&tpf)->default_value(1),
     "fudge factor in computation of move acceptance probability"
     " during transition matrix initialization")
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

  // if we specified a pattern file, make sure it exists
  assert(pattern_file.empty() || fs::exists(pattern_file));

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

  network_simulation ns(patterns, random_state(nodes, rnd, generator));

  cout << endl
       << "maximum energy: " << ns.network.max_energy
       << " (of " << pattern_number * nodes * (nodes - 1) / ns.network.energy_scale
       << " possible)" << endl
       << "maximum energy change: " << ns.network.max_energy_change
       << " (of " << 2 * pattern_number * (nodes - 1) / ns.network.energy_scale
       << " possible)" << endl
       << "energy scale: " << ns.network.energy_scale << endl
       << endl;

  if (debug) {
    ns.print_patterns();
    cout << endl;
    ns.network.print_couplings();
    cout << endl;
  }

  // -------------------------------------------------------------------------------------
  // Initialize transition matrix
  // -------------------------------------------------------------------------------------

  for (uint ii = 0; ii < pow(10,7); ii++) {

    const vector<bool> new_state = random_change(ns.state,rnd(generator));

    const int old_energy = ns.energy();
    const int new_energy = ns.energy(new_state);
    const int energy_change = new_energy - old_energy;

    ns.add_transition(old_energy, energy_change);

    if (energy_change < 0) {
      // if the energy change for the proposed move is negative, always accept it
      ns.state = new_state;

    } else {
      // otherwise, accept the move with some probability
      const double old_norm = ns.transitions_from(old_energy);
      const double new_norm = ns.transitions_from(new_energy);

      const double forward_flux
        = double(ns.transitions(old_energy, energy_change) + tpf) / (old_norm + tpf);
      const double backward_flux
        = double(ns.transitions(new_energy, -energy_change) + tpf) / (new_norm + tpf);

      const double transition_probability = max(forward_flux/backward_flux,
                                                exp(-energy_change/min_temp));

      if (rnd(generator) < transition_probability) {
        ns.state = new_state;
      }
    }

    // update histograms and sample counts
    ns.update_histograms();
    ns.update_samples(new_energy, old_energy);
  }

  ns.compute_weights_from_transitions();

  for (uint ee = ns.network.energy_range - 1; ee > 0; ee--) {
    const uint observations = ns.energy_observations(ee);
    if (observations > 0) {
      cout << setw(3) << int(ee) - int(ns.network.max_energy) << " "
           << setw(10) << ns.ln_weights.at(ee) << " "
           << setw(8) << observations << " ";
      if (ee < 0){
        cout << ns.samples.at(-ee) << " ";
      }
      cout << endl;
    }
  }
  cout << endl;

  ns.initialize_histograms();

  for (uint ii = 0; ii < pow(10,7); ii++) {

    const vector<bool> new_state = random_change(ns.state,rnd(generator));
    const double acceptance_probability =
      exp(ns.ln_weights.at(ns.energy(new_state)) - ns.ln_weights.at(ns.energy()));

    if (rnd(generator) < acceptance_probability) {
      ns.state = new_state;
    }
    ns.update_histograms();

  }

    for (uint ee = ns.network.energy_range - 1; ee > 0; ee--) {
    const uint observations = ns.energy_observations(ee);
    if (observations > 0) {
      cout << setw(3) << int(ee) - int(ns.network.max_energy) << " "
           << setw(8) << observations << " "
           << endl;
    }
  }


}
