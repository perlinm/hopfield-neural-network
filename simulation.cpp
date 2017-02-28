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

  bool all_temps;
  bool fixed_temp;
  double beta_cap;
  int init_factor;
  int log10_iterations;

  po::options_description simulation_options("General simulation options",
                                             help_text_length);
  simulation_options.add_options()
    ("all_T", po::value<bool>(&all_temps)->default_value(false)->implicit_value(true),
     "run an all-temperature simulation")
    ("fixed_T", po::value<bool>(&fixed_temp)->default_value(false)->implicit_value(true),
     "run a fixed-temperature simulation")
    ("beta_cap", po::value<double>(&beta_cap)->default_value(1),
     "maximum inverse temperature (equivalently, minimum temperature) scale"
     " of interest in the simulation")
    ("init_factor", po::value<int>(&init_factor)->default_value(5),
     "run for nodes * pattern_number * 10^(init_factor) iterations"
     " per initialization cycle")
    ("log10_iterations", po::value<int>(&log10_iterations)->default_value(7),
     "log10 of the number of iterations to simulate")
    ;

  double target_sample_error;
  int tpff; // "transition probability fudge factor"

  po::options_description all_temps_options("All temperature simulation options",
                                            help_text_length);
  all_temps_options.add_options()
    ("sample_error", po::value<double>(&target_sample_error)->default_value(0.01,"0.01"),
     "the initialization routine terminates when it achieves this"
     " expected fractional sample error at an inverse temperature beta_cap")
    ("transition_factor", po::value<int>(&tpff)->default_value(1),
     "fudge factor in computation of move acceptance probability"
     " during transition matrix initialization")
    ;

  // collect options
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

  // we should have at least two nodes, and at least one pattern
  if (nodes < 2) {
    cout << "the network should consist of at least two nodes" << endl;
    return -1;
  }

  if (pattern_number < 1) {
    cout << "we need at least one pattern to define a (nontrivial) network" << endl;
    return -1;
  }

  // we can specify either nodes, or a pattern file; not both
  if (!nodes && pattern_file.empty()) {
    cout << "either choose a size (number of nodes) for a network"
         << " with random patterns, or provide a pattern file" << endl;
    return -1;
  }

  // we must run either an all-temperature or a fixed-temperature simulation
  if (all_temps + fixed_temp != 1) {
    cout << "you need to choose whether to run a all- "
         << " or fixed-temperature simulation" << endl;
    return -1;
  }

  // if we specified a pattern file, make sure it exists
  if (!pattern_file.empty() && !fs::exists(pattern_file)) {
    cout << "the specified pattern file does not exist!" << endl;
    return -1;
  }

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
  // each line of the pattern file should contain one pattern,
  //   identified by 1s (spin up, or +1) and 0s (spin down, or -1)
  if (using_pattern_file) {
    ifstream input(pattern_file);
    string line;

    // loop over lines
    while (getline(input,line)) {
      vector<bool> pattern = {};
      for (int ii = 0, size = line.length(); ii < size; ii++) {
        if (line[ii] == '1') pattern.push_back(true);
        if (line[ii] == '0') pattern.push_back(false);
      }
      patterns.push_back(pattern);
    }

  } else {

    // if we are not using a pattern file, generate random patterns
    for (int ii = 0; ii < pattern_number; ii++) {
      patterns.push_back(random_state(nodes, rnd, generator));
    }

  }

  // make sure that all patterns are the same size,
  //   i.e. they all contain the same number of nodes
  for (int ii = 1, size = patterns.size(); ii < size; ii++) {
    if (patterns[ii-1].size() != patterns[ii].size()){
      cout << "patterns " << ii-1 << " and " << ii
           << " do not have the same size!" << endl;
      return -1;
    }
  }
  // set the number of nodes and number of patterns
  nodes = patterns[0].size();
  pattern_number = patterns.size();

  // construct network simulation object with a random initial state
  network_simulation ns(patterns, random_state(nodes, rnd, generator));

  // print some info about the simulation
  cout << endl
       << "maximum energy: " << ns.network.max_energy << endl
       << "maximum energy change: " << ns.network.max_energy_change << endl
       << "energy scale: " << ns.network.energy_scale << endl
       << "inverse temperature: " << beta_cap << endl
       << endl;

  // set inverse temperature scale the inverse units of our energies
  beta_cap *= double(ns.network.energy_scale) / nodes;

  // number of iterations per initialization cycle
  const int iterations_per_cycle
    = ns.network.nodes * ns.patterns.size() * pow(10, init_factor);

  // make a hash of the patterns to identify this network
  const int hash = [&]() -> int {
    size_t running_hash = 0;
    for (int pp = 0, size = patterns.size(); pp < size; pp++) {
      for (int nn = 0; nn < ns.network.nodes; nn++) {
        bo::hash_combine(running_hash, size_t(patterns[pp][nn]));
      }
    }
    return running_hash;
  }();

  if (debug) {
    ns.print_patterns();
    cout << endl;
    ns.network.print_couplings();
    cout << endl;
  }

  // -------------------------------------------------------------------------------------
  // Initialize the weight array
  // -------------------------------------------------------------------------------------

  // initialize weight array
  if (fixed_temp) {

    cout << "starting a fixed temperature simulation" << endl << endl;

    // for a fixed (or infinite) temperature simulation, use boltzmann weights
    for (int ee = 0; ee < ns.energy_range; ee++) {
      ns.ln_weights[ee] = -ee * beta_cap;
    }

    // run for one initialization cycle in order to locate the entropy peak
    for (int ii = 0; ii < iterations_per_cycle; ii++) {

      // make a random move and update the energy histogram
      ns.state = random_change(ns.state, rnd(generator));
      ns.update_energy_histogram(ns.energy());
    }

    // locate the entropy peak
    for (int ee = 0; ee < ns.energy_range; ee++) {
      if (ns.energy_histogram[ee] > ns.energy_histogram[ns.entropy_peak]) {
        ns.entropy_peak = ee;
      }
    }

  } else if (all_temps) {

    // if there is no file containing the transition matrix we need,
    //   run the standard initialization routine
    if (true) {

      cout << "initializing transition matrix for an all-temperature simulation..." << endl
           << "sample_error cycle_number" << endl;

      // number of initialization cycles we have finished
      int cycles = 0;
      // expected fractional error in sample count at an inverse temperature beta
      double sample_error;

      int new_energy; // energy of the state we move into
      int old_energy = ns.energy(); // energy of the last state

      do {
        for (int ii = 0; ii < iterations_per_cycle; ii++) {

          // construct the state which we are proposing to move into
          const vector<bool> proposed_state = random_change(ns.state, rnd(generator));

          // energy of the proposed state, and the energy change for the proposed move
          const int proposed_energy = ns.energy(proposed_state);
          const int energy_change = proposed_energy - old_energy;

          // this update should happen regardless of whether we make the move
          ns.update_transition_histogram(old_energy, energy_change);

          if (energy_change <= 0) {
            // always accept moves into states of lower energy
            ns.state = proposed_state;
            new_energy = proposed_energy;

          } else { // accept the moves to higher energy states with some probability

            // normalization factor for transitions from both energies
            const double old_norm = ns.transitions_from(old_energy);
            const double new_norm = ns.transitions_from(proposed_energy);

            // compute the (normalized) flux of proposed moves forward and backward
            //   between the current and proposed energies
            // add fudge factors (tpff) which will make the rejection of transitions
            //   more conservative when we have poor statistics (few counts)
            //   in the transition histogram
            // the effect of these fudge factors vanishes as the number of counts
            //   in the transition matrix increases
            // these fudge factors effectively prevent us from getting stuck
            //   at low energies
            const double forward_flux
              = (double(ns.transitions(old_energy, energy_change) + tpff)
                 / (old_norm + tpff));
            const double backward_flux
              = (double(ns.transitions(proposed_energy, -energy_change) + tpff)
                 / (new_norm + tpff));

            // in order to get good statistics on the transition matrix,
            //   we wish to sample all energies as equally as we can
            // if the forward flux F_{i->f} of proposed moves from E_i to E_f is,
            //   say, twice the backwards flux F_{f->i}, then to sample E_i and E_f
            //   equally we should reject half of the proposed moves from E_i to E_f
            // in general, the acceptance probability to sample E_i and E_f equally
            //   is F_{i->f} / F_{f->i}
            // there is no reason, however, to have acceptance probabilities lower
            //   than the ratio of boltzmann weights on E_i and E_f, as otherwise
            //   we are spending more time sampling E_i (relative to E_f) than we would
            //   at the minimum temperature of the simulation
            const double acceptance_probability = max(forward_flux / backward_flux,
                                                      exp(-energy_change * beta_cap));

            // if we pass a probability test, accept the move
            if (rnd(generator) < acceptance_probability) {
              ns.state = proposed_state;
              new_energy = proposed_energy;
            } else {
              // otherwise reject it
              new_energy = old_energy;
            }
          }

          // update the energy and sample histograms
          // we don't care about other histograms during initialization
          ns.update_energy_histogram(new_energy);
          ns.update_sample_histogram(new_energy, old_energy);

          // as we move on with our lives (and this loop) the new energy turns old
          old_energy = new_energy;
        }

        // increment the cycle cound and compute the density of states
        cycles++;
        ns.compute_dos_from_transitions();

        // compute the expected fractional sample error at an inverse temperature beta_cap
        sample_error = ns.fractional_sample_error(beta_cap);

        cout << fixed << setprecision(ceil(-log10(target_sample_error)) + 2)
             << sample_error << " " << cycles << endl;

        // loop until we satisfy the end condition for initialization
      } while (sample_error > target_sample_error);

      cout << endl;

    } else { // if the transition matrix we need is already exists, read it in

      /************************************************************/
      // read in transition matrix
      // print text telling us what's going on while it's happning
      /************************************************************/

      // compute density of states from the transition matrix we read in
      ns.compute_dos_from_transitions();

    }

    // compute weights appropriately
    ns.compute_weights_from_dos(beta_cap);

    if (debug) {
      ns.print_energy_data();
      cout << endl;
    }

    // initialize a new random state and clear the histograms
    ns.state = random_state(nodes, rnd, generator);
    ns.initialize_histograms();

    cout << "starting an all-temperature simulation" << endl << endl;
  }

  // -------------------------------------------------------------------------------------
  // Run simulation
  // -------------------------------------------------------------------------------------

  int new_energy; // energy of the state we move into
  int old_energy = ns.energy(); // energy of the last state
  for (int ii = 0; ii < pow(10,log10_iterations); ii++) {

    // construct the state which we are proposing to move into,
    //   and compute its energy
    const vector<bool> proposed_state = random_change(ns.state, rnd(generator));
    const int proposed_energy = ns.energy(proposed_state);

    // determine the probability with which we will accept a move into the proposed state
    const double acceptance_probability = exp(ns.ln_weights[proposed_energy]
                                              - ns.ln_weights[old_energy]);

    // if we pass a probability test, accept the move
    if (rnd(generator) < acceptance_probability) {
      ns.state = proposed_state;
      new_energy = proposed_energy;
    } else {
      new_energy = old_energy;
    }

    // update all histograms (except the transition histogram used for initialization)
    ns.update_energy_histogram(new_energy);
    ns.update_sample_histogram(new_energy, old_energy);
    ns.update_state_histograms(new_energy);
    ns.update_distance_histograms(ns.state, new_energy);

    // update the old energy
    old_energy = new_energy;
  }

  // compute the density of states
  ns.compute_dos_from_energy_histogram();

  if (debug) {
    ns.print_energy_data();
    cout << endl;
    ns.print_expected_states();
    cout << endl;
    ns.print_distances();
    cout << endl;
  }

  cout << "simulation complete" << endl << endl;

  // -------------------------------------------------------------------------------------
  // Write data files
  // -------------------------------------------------------------------------------------



}
