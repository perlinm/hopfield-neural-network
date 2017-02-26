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

  const int help_text_length = 85;

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

  int nodes;
  int pattern_number;
  string pattern_file;

  po::options_description network_parameters("Network parameters",help_text_length);
  network_parameters.add_options()
    ("nodes", po::value<int>(&nodes)->default_value(10),
     "number of nodes which make up the network")
    ("patterns", po::value<int>(&pattern_number)->default_value(0,"nodes"),
     "number of random patterns to use")
    ("pattern_file", po::value<string>(&pattern_file),
     "input file containing patterns stored in the neural network")
    ;

  int log10_iterations;
  bool all_temps;
  bool inf_temp;
  bool fixed_temp;
  double temp_scale;

  po::options_description simulation_options("General simulation options",
                                             help_text_length);
  simulation_options.add_options()
    ("log10_iterations", po::value<int>(&log10_iterations)->default_value(7),
     "log10 of the number of iterations to simulate")
    ("all_temps", po::value<bool>(&all_temps)->default_value(true)->implicit_value(true),
     "run an all-temperature simulation")
    ("inf_temp", po::value<bool>(&inf_temp)->default_value(false)->implicit_value(true),
     "run an infinite temperature simulation")
    ("fixed_temp",
     po::value<bool>(&fixed_temp)->default_value(false)->implicit_value(true),
     "run a fixed-temperature simulation")
    ("temp_scale", po::value<double>(&temp_scale)->default_value(0.1),
     "temperature scale of interest in simulation")

    ;

  int log10_init_cycle;
  double target_sample_error;
  int tpff; // "transition probability fudge factor"

  po::options_description all_temps_options("All temperature simulation options",
                                            help_text_length);
  all_temps_options.add_options()
    ("init_cycle", po::value<int>(&log10_init_cycle)->default_value(7),
     "log10 of the number of iterations in one initialization cycle")
    ("sample_error", po::value<double>(&target_sample_error)->default_value(0.01,"0.01"),
     "the transition matrix initialization routine terminates when it achieves"
     " this expected fractional sample error at given minimum temperature")
    ("transition_factor", po::value<int>(&tpff)->default_value(1),
     "fudge factor in computation of move acceptance probability"
     " during transition matrix initialization")
    ;

  po::options_description all("All options");
  all.add(general);
  all.add(network_parameters);
  all.add(simulation_options);
  all.add(all_temps_options);

  // collect inputs
  po::variables_map inputs;
  po::store(parse_command_line(arg_num, arg_vec, all), inputs);
  po::notify(inputs);

  // if requested, print help text
  if (inputs.count("help")) {
    cout << all;
    return 0;
  }

  // by default, use the same number of patterns as nodes
  if (pattern_number == 0) pattern_number = nodes;

  // -------------------------------------------------------------------------------------
  // Process and run sanity checks on inputs
  // -------------------------------------------------------------------------------------

  // we should have a positive number of nodes and patterns
  assert(nodes > 0);
  assert(pattern_number > 0);

  // we can specify either nodes, or a pattern file; not both
  assert(nodes || !pattern_file.empty());

  // we must choose some temperature option
  assert(inf_temp || fixed_temp || all_temps);

  // the temperature scale cannot be zero (negative temperatures not yet supported)
  assert(temp_scale > 0);

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
      for (int ii = 0; ii < int(line.length()); ii++) {
        if (line[ii] == '1') pattern.push_back(true);
        if (line[ii] == '0') pattern.push_back(false);
      }
      patterns.push_back(pattern);
    }

  } else { // if we are not using a pattern file, generate random patterns

    for (int ii = 0; ii < pattern_number; ii++) {
      patterns.push_back(random_state(nodes, rnd, generator));
    }

  }

  for (int ii = 1; ii < int(patterns.size()); ii++) {
    if (patterns[ii-1].size() != patterns[ii].size()){
      cout << "patterns " << ii-1 << " and " << ii
           << " do not have the same size!" << endl;
      return -1;
    }
  }
  pattern_number = patterns.size();
  nodes = patterns[0].size();

  network_simulation ns(patterns, random_state(nodes, rnd, generator));

  // adjust temperature scale to be compatible with the energy units used in simulation
  temp_scale *= nodes/ns.network.energy_scale;

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
  // Initialize weight array
  // -------------------------------------------------------------------------------------

  // initialize weight array
  if (inf_temp) {

    cout << "starting an infinite temperature simulation" << endl;
    ns.ln_weights = vector<double>(ns.network.energy_range, 1);

  } else if (fixed_temp) {

    cout << "starting a fixed temperature simulation" << endl;
    for (int ee = 0; ee < ns.network.energy_range; ee++) {
      ns.ln_weights[ee] = -ee/temp_scale;
    }

  } else if (all_temps) {

    cout << "initializing transition matrix for an all-temperature simulation..." << endl
         << "cycles sample_error" << endl;
    int cycles = 0;
    double sample_error;
    int update_energy;
    int old_energy = ns.energy();
    do {
      for (int ii = 0; ii < pow(10,log10_init_cycle); ii++) {

        const vector<bool> new_state = random_change(ns.state, rnd(generator));

        const int new_energy = ns.energy(new_state);
        const int energy_change = new_energy - old_energy;

        ns.add_transition(old_energy, energy_change);

        if (energy_change <= 0) {
          // proposed new state does not have a higher energy, always accept it
          ns.state = new_state;
          update_energy = new_energy;

        } else {
          // otherwise, accept the move with some probability
          const double old_norm = ns.transitions_from(old_energy);
          const double new_norm = ns.transitions_from(new_energy);

          const double forward_flux
            = double(ns.transitions(old_energy, energy_change) + tpff) / (old_norm + tpff);
          const double backward_flux
            = double(ns.transitions(new_energy, -energy_change) + tpff) / (new_norm + tpff);

          const double acceptance_probability = max(forward_flux/backward_flux,
                                                    exp(-energy_change/temp_scale));

          if (rnd(generator) < acceptance_probability) {
            ns.state = new_state;
            update_energy = new_energy;
          } else {
            update_energy = old_energy;
          }
        }

        ns.update_energy_histogram(update_energy);
        ns.update_samples(update_energy, old_energy);
        old_energy = update_energy;
      }

      ns.compute_dos_and_weights_from_transitions(temp_scale);

      cycles++;
      sample_error = ns.fractional_sample_error(temp_scale);
      cout << cycles << " " << sample_error << endl;

    } while (sample_error > target_sample_error);
    cout << endl;

    if (debug) {
      ns.print_energy_data();
      cout << endl;
    }

    ns.initialize_histograms();
    cout << "starting an all-temperature simulation" << endl << endl;
  }

  // -------------------------------------------------------------------------------------
  // Run simulation
  // -------------------------------------------------------------------------------------

  int update_energy;
  int old_energy = ns.energy();
  for (int ii = 0; ii < pow(10,log10_iterations); ii++) {

    const vector<bool> new_state = random_change(ns.state, rnd(generator));
    const int new_energy = ns.energy(new_state);

    const double acceptance_probability = exp(ns.ln_weights[new_energy]
                                              - ns.ln_weights[old_energy]);

    if (rnd(generator) < acceptance_probability) {
      ns.state = new_state;
      update_energy = new_energy;
    } else {
      update_energy = old_energy;
    }

    ns.add_transition(old_energy, new_energy - old_energy);
    ns.update_energy_histogram(update_energy);
    ns.update_state_histograms(update_energy);
    ns.update_samples(update_energy, old_energy);
    old_energy = update_energy;
  }

  ns.compute_dos_from_energy_histogram();

  cout << "simulation complete" << endl << endl;

  if (debug) {
    ns.print_energy_data();
    cout << endl;
  }

  for (int ee = ns.network.energy_range - 1; ee >= 0; ee--) {
    const int observations = ns.energy_histogram[ee];
    if (observations > 0) {
      cout << setw(log10(2*ns.network.max_energy)+2)
           << ee - ns.network.max_energy << " ";
      for (int ii = 0; ii < ns.network.nodes; ii++) {
        cout << setw(10)
             << double(ns.state_histograms[ee][ii])/observations << " ";
      }
      cout << endl;
    }
  }
  cout << endl;


}
