#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness
#include <fstream> // for file input

#include <boost/filesystem.hpp> // filesystem path manipulation library
#include <boost/program_options.hpp> // options parsing library
#include <boost/functional/hash.hpp> // for hashing methods

#include "methods.h"

using namespace std;
namespace bo = boost;
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
    ("all_temps", po::value<bool>(&all_temps)->default_value(false)->implicit_value(true),
     "run an all-temperature simulation")
    ("fixed_temp",
     po::value<bool>(&fixed_temp)->default_value(false)->implicit_value(true),
     "run a fixed-temperature simulation")
    ("inf_temp", po::value<bool>(&inf_temp)->default_value(false)->implicit_value(true),
     "run an infinite temperature simulation")
    ("temp_scale", po::value<double>(&temp_scale)->default_value(0.1,"0.1"),
     "temperature scale of interest in simulation")
    ("log10_iterations", po::value<int>(&log10_iterations)->default_value(7),
     "log10 of the number of iterations to simulate")
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
     " this expected fractional sample error at a temperature temp_scale")
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

  // default to an all-temperature simulation
  if (!all_temps && !fixed_temp && !inf_temp) all_temps = true;

  // -------------------------------------------------------------------------------------
  // Process and run sanity checks on inputs
  // -------------------------------------------------------------------------------------

  // we should have at least two nodes, and at least one pattern
  assert(nodes > 1);
  assert(pattern_number > 0);

  // we can specify either nodes, or a pattern file; not both
  assert(nodes || !pattern_file.empty());

  // we can only have one temperature option
  assert(all_temps + fixed_temp + inf_temp == 1);

  // the temperature scale cannot be zero
  assert(temp_scale != 0);

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
      for (int ii = 0, size = line.length(); ii < size; ii++) {
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

  for (int ii = 1, size = patterns.size(); ii < size; ii++) {
    if (patterns[ii-1].size() != patterns[ii].size()){
      cout << "patterns " << ii-1 << " and " << ii
           << " do not have the same size!" << endl;
      return -1;
    }
  }
  pattern_number = patterns.size();
  nodes = patterns[0].size();

  network_simulation ns(patterns, random_state(nodes, rnd, generator));

  // make simulation hash
  const int hash = [&]() -> int {
    size_t running_hash = 0;
    for (int pp = 0, size = patterns.size(); pp < size; pp++) {
      for (int nn = 0; nn < ns.network.nodes; nn++) {
        bo::hash_combine(running_hash, size_t(patterns[pp][nn]));
      }
    }
    bo::hash_combine(running_hash, size_t(all_temps));
    bo::hash_combine(running_hash, size_t(fixed_temp));
    bo::hash_combine(running_hash, size_t(inf_temp));
    bo::hash_combine(running_hash, size_t(temp_scale));
    return running_hash;
  }();

  // adjust temperature scale to be compatible with the energy units used in simulation
  temp_scale *= double(nodes)/ns.network.energy_scale;

  cout << endl
       << "maximum energy: " << ns.network.max_energy << endl
       << "maximum energy change: " << ns.network.max_energy_change << endl
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
         << "sample_error cycle_number" << endl;
    int cycles = 0;
    double sample_error;
    int new_energy;
    int old_energy = ns.energy();
    do {
      for (int ii = 0; ii < pow(10,log10_init_cycle); ii++) {

        const vector<bool> proposed_state = random_change(ns.state, rnd(generator));

        const int proposed_energy = ns.energy(proposed_state);
        const int energy_change = proposed_energy - old_energy;

        ns.add_transition(old_energy, energy_change);

        if (energy_change <= 0) {
          // proposed new state does not have a higher energy, always accept it
          ns.state = proposed_state;
          new_energy = proposed_energy;

        } else {
          // otherwise, accept the move with some probability
          const double old_norm = ns.transitions_from(old_energy);
          const double new_norm = ns.transitions_from(proposed_energy);

          const double forward_flux
            = (double(ns.transitions(old_energy, energy_change) + tpff)
               / (old_norm + tpff));
          const double backward_flux
            = (double(ns.transitions(proposed_energy, -energy_change) + tpff)
               / (new_norm + tpff));

          const double acceptance_probability = max(forward_flux/backward_flux,
                                                    exp(-energy_change/temp_scale));

          if (rnd(generator) < acceptance_probability) {
            ns.state = proposed_state;
            new_energy = proposed_energy;
          } else {
            new_energy = old_energy;
          }
        }

        ns.update_energy_histogram(new_energy);
        ns.update_sample_histogram(new_energy, old_energy);
        old_energy = new_energy;
      }

      ns.compute_dos_and_weights_from_transitions(temp_scale);

      cycles++;
      sample_error = ns.fractional_sample_error(temp_scale);
      cout << fixed << setprecision(ceil(-log10(target_sample_error)) + 2)
           << sample_error << " " << cycles << endl;

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

  int new_energy;
  int old_energy = ns.energy();
  for (int ii = 0; ii < pow(10,log10_iterations); ii++) {

    const vector<bool> proposed_state = random_change(ns.state, rnd(generator));
    const int proposed_energy = ns.energy(proposed_state);

    const double acceptance_probability = exp(ns.ln_weights[proposed_energy]
                                              - ns.ln_weights[old_energy]);

    if (rnd(generator) < acceptance_probability) {
      ns.state = proposed_state;
      new_energy = proposed_energy;
    } else {
      new_energy = old_energy;
    }

    ns.update_energy_histogram(new_energy);
    ns.update_sample_histogram(new_energy, old_energy);
    ns.update_state_histograms(new_energy);
    ns.update_distance_histograms(new_energy);
    old_energy = new_energy;
  }

  ns.compute_dos_from_energy_histogram();

  if (debug) {
    ns.print_energy_data();
    cout << endl;
    ns.print_states();
    cout << endl;
    ns.print_distances();
    cout << endl;
  }

  cout << "simulation complete" << endl << endl;

}
