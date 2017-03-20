#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness
#include <fstream> // for file input
#include <ctime> // for keeping track of runtime

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

  bool suppress;
  long seed;
  long pattern_seed;
  bool print_suffix;

  po::options_description general("General options", help_text_length);
  general.add_options()
    ("help,h", "produce help message")
    ("suppress", po::value<bool>(&suppress)->default_value(false)->implicit_value(true),
     "suppress some of the text that would normally be printed")
    ("seed", po::value<long>(&seed)->default_value(0),
     "seed for random number generator")
    ("pattern_seed", po::value<long>(&pattern_seed)->default_value(0),
     "random number generator seed when generating patterns")
    ("suffix", po::value<bool>(&print_suffix)->default_value(false)->implicit_value(true),
     "print the file suffix associated with this simulation instead of running it")
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

  bool fixed_temp;
  double input_beta_cap;
  int log10_iterations;
  int init_factor;
  int print_time;

  po::options_description simulation_options("General simulation options",
                                             help_text_length);
  simulation_options.add_options()
    ("fixed_T", po::value<bool>(&fixed_temp)->default_value(false)->implicit_value(true),
     "run a fixed-temperature simulation")
    ("beta_cap", po::value<double>(&input_beta_cap)->default_value(1),
     "maximum (if positive) or minimum (if negative) inverse temperature"
     " of interest in the simulation")
    ("log10_iterations", po::value<int>(&log10_iterations)->default_value(7),
     "log10 of the number of iterations to simulate")
    ("init_factor", po::value<int>(&init_factor)->default_value(5),
     "run for nodes * pattern_number * 10^(init_factor) iterations"
     " per initialization cycle")
    ("print_time", po::value<int>(&print_time)->default_value(30),
     "number of minutes between intermediate data file dumps")
    ;

  double target_sample_error;

  po::options_description all_temps_options("All temperature simulation options",
                                            help_text_length);
  all_temps_options.add_options()
    ("sample_error", po::value<double>(&target_sample_error)->default_value(0.01,"0.01"),
     "the initialization routine terminates when it achieves this"
     " expected fractional sample error at an inverse temperature beta_cap")
    ;

  string data_dir;

  po::options_description io_options("File I/O options", help_text_length);
  io_options.add_options()
    ("data_dir", po::value<string>(&data_dir)->default_value("./data"), "data directory")
    ;

  // collect options
  po::options_description all("All options");
  all.add(general);
  all.add(network_parameters);
  all.add(simulation_options);
  all.add(all_temps_options);
  all.add(io_options);

  // collect inputs
  po::variables_map inputs;
  po::store(parse_command_line(arg_num, arg_vec, all), inputs);
  po::notify(inputs);

  // if requested, print help text
  if (inputs.count("help")) {
    cout << all;
    return 0;
  }

  clock_t simulation_start_time = time(NULL);

  // -------------------------------------------------------------------------------------
  // Process and run sanity checks on inputs
  // -------------------------------------------------------------------------------------

  // we should have at least two nodes, and at least one pattern
  if (nodes < 2) {
    cout << "the network should consist of at least two nodes" << endl;
    return -1;
  }

  // don't allow attempts to simulate too large of a network
  assert(nodes < 300);

  // by default, use the same number of patterns as nodes
  assert(pattern_number >= 0);
  if (pattern_number == 0) pattern_number = nodes;

  assert(log10_iterations > 0);
  assert(init_factor > 0);

  // make sure that iteration counters/factors aren't too large
  assert(log10_iterations < 18);
  assert(log10(nodes) + log10(pattern_number) + init_factor < 10);

  // if we're doing an infinite temperature simulation,
  //   we don't need the machinery of weights, etc.
  if (input_beta_cap == 0) fixed_temp = true;

  // we can specify either nodes, or a pattern file; not both
  if (!nodes && pattern_file.empty()) {
    cout << "either choose a network size (i.e. number of nodes)"
         << " with random number of patterns, or provide a pattern file" << endl;
    return -1;
  }

  // if we specified a pattern file, make sure it exists
  if (!pattern_file.empty() && !fs::exists(pattern_file)) {
    cout << "the specified pattern file does not exist:" << endl
         << pattern_file << endl;
    return -1;
  }

  // determine whether we are using a pattern file afterall
  const bool using_pattern_file = !pattern_file.empty();

  fs::create_directory(data_dir);

  // initialize random number generator
  uniform_real_distribution<double> rnd(0.0,1.0); // uniform distribution on [0,1)
  mt19937_64 generator; // use the 64-bit Mersenne Twister 19937 generator

  // -------------------------------------------------------------------------------------
  // Construct patterns for network and initialize network simulation
  // -------------------------------------------------------------------------------------

  generator.seed(pattern_seed);
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
    input.close();

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
      printf("patterns %u and %u do not have the same size!\n", ii-1, ii);
      return -1;
    }
  }
  // set the number of nodes and number of patterns
  nodes = patterns[0].size();
  pattern_number = patterns.size();

  // construct network simulation object with a random initial state
  generator.seed(seed);
  network_simulation ns(patterns, random_state(nodes, rnd, generator));

  // inverse temperature in units compatible with that for our energies
  const double beta_cap = input_beta_cap * ns.network.energy_scale / ns.network.nodes;

  // number of iterations per initialization cycle
  const long long iterations_per_cycle
    = ns.network.nodes * ns.pattern_number * pow(10, init_factor);
  assert(iterations_per_cycle > 0);

  // print some info about the simulation
  cout << "nodes: " << ns.network.nodes << endl
       << "pattern number: " << ns.pattern_number << endl
       << "inverse temperature: " << input_beta_cap << endl
       << "energy scale: " << ns.network.energy_scale << endl
       << "maximum energy: " << ns.network.max_energy << endl
       << "maximum energy change: " << ns.network.max_energy_change << endl
       << endl;

  // make a hash of the patterns and target sample error to identify this simulation
  const size_t hash = [&]() -> size_t {
    size_t running_hash = 0;
    for (int pp = 0; pp < ns.pattern_number; pp++) {
      for (int nn = 0; nn < ns.network.nodes; nn++) {
        bo::hash_combine(running_hash, size_t(patterns[pp][nn]));
      }
    }
    bo::hash_combine(running_hash, input_beta_cap);
    bo::hash_combine(running_hash, target_sample_error);
    return running_hash;
  }();

  // suffix to all data files read/written by this simulation
  const string beta_tag = ("-" + string(fixed_temp ? "f" : "") + "B"
                           + string(input_beta_cap < 0 ? "n" : "")
                           + to_string(int(round(abs(input_beta_cap)))));
  const string node_tag = "-N" + to_string(ns.network.nodes);
  const string pattern_tag = "-P" + to_string(ns.pattern_number);
  const string file_suffix = (node_tag + pattern_tag + beta_tag
                              + "-h" + to_string(hash) + ".txt");

  // paths to data files
  const string transitions_file
    = (fs::path(data_dir) / fs::path("transitions" + file_suffix)).string();
  const string weights_file
    = (fs::path(data_dir) / fs::path("weights" + file_suffix)).string();
  const string energy_file
    = (fs::path(data_dir) / fs::path("energies" + file_suffix)).string();
  const string distance_file
    = (fs::path(data_dir) / fs::path("distances" + file_suffix)).string();

  // header for all data files
  stringstream file_header_stream;
  file_header_stream << "# nodes: " << ns.network.nodes << endl
                     << "# patterns: " << ns.pattern_number << endl
                     << "# energy_scale: " << ns.network.energy_scale << endl
                     << "# energy_range: " << ns.energy_range << endl
                     << "# max_de: " << ns.max_de << endl
                     << "# beta_cap: " << input_beta_cap << endl
                     << "# target_sample_error: " << target_sample_error << endl;
  const string file_header = file_header_stream.str();

  if (!suppress) {
    ns.print_patterns();
    cout << endl;
    ns.network.print_couplings();
    cout << endl;
  }

  if (print_suffix) {
    cout << "output file suffix:" << endl
         << file_suffix << endl;
    return 0;
  }

  // -------------------------------------------------------------------------------------
  // Initialize the weight array
  // -------------------------------------------------------------------------------------

  // initialize weight array
  if (fixed_temp) {

    cout << "initializing a fixed temperature simulation" << endl;

    // run for one initialization cycle in order to locate the entropy peak
    for (long long ii = 0; ii < iterations_per_cycle; ii++) {

      // make a random move and update the energy histogram
      ns.state = random_change(ns.state, rnd(generator));
      ns.energy_histogram[ns.energy()]++;
    }

    // locate the entropy peak
    for (int ee = 0; ee < ns.energy_range; ee++) {
      if (ns.energy_histogram[ee] > ns.energy_histogram[ns.entropy_peak]) {
        ns.entropy_peak = ee;
      }
    }

    // for a fixed (or infinite) temperature simulation, use boltzmann weights
    for (int ee = 0; ee < ns.energy_range; ee++) {
      ns.ln_weights[ee] = -(ee-ns.entropy_peak) * beta_cap;
    }

  } else { // run an all temperature simulation

    // if there is no file which already contains the weights we need,
    //   run the standard initialization routine
    if (!fs::exists(weights_file)) {

      cout << "starting all-temperature initialization routine..." << endl
           << "iterations per cycle: " << iterations_per_cycle << endl;

      if (fs::exists(transitions_file)) {
        ns.read_transitions_file(transitions_file);
      }

      cout << "sample_error cycle_number" << endl;

      // number of initialization cycles we have finished
      int cycles = 0;
      // expected fractional error in sample count at an inverse temperature beta
      double sample_error;

      int new_energy; // energy of the state we move into
      int old_energy = ns.energy(); // energy of the last state
      assert(old_energy < ns.energy_range);
      do {
        for (long long ii = 0; ii < iterations_per_cycle; ii++) {

          // construct the state which we are proposing to move into
          const vector<bool> proposed_state = random_change(ns.state, rnd(generator));

          // energy of the proposed state, and the energy change for the proposed move
          const int proposed_energy = ns.energy(proposed_state);
          const int energy_change = proposed_energy - old_energy;
          assert(abs(energy_change) <= ns.max_de);

          // this update should happen regardless of whether we make the move
          ns.update_transition_histogram(old_energy, energy_change);

          if ((beta_cap > 0 && energy_change <= 0)
              || (beta_cap < 0 && energy_change >= 0)) {
            // always accept moves to a lower energy in a positive temperature simulation,
            //   and "" higher energy "" negative temperature ""
            ns.state = proposed_state;
            new_energy = proposed_energy;

          } else { // accept the moves to higher energy states with some probability

            // compute the (unnormalized) flux of proposed moves forward and backward
            //   between the current and proposed energies
            const long forward_flux = ns.transitions(old_energy, energy_change);
            const long backward_flux = ns.transitions(proposed_energy, -energy_change);

            // normalization factor for transition fluxes from both energies
            const long forward_norm = ns.transitions_from(old_energy);
            const long backward_norm = ns.transitions_from(proposed_energy);

            // in order to get good statistics on the transition matrix,
            //   we wish to sample all energies as equally as we can
            // if the forward flux F_{i->f} of proposed moves from E_i to E_f is,
            //   say, twice the backwards flux F_{f->i}, then to sample E_i and E_f
            //   equally we should reject half of the proposed moves from E_i to E_f
            // in general, the acceptance probability to sample E_i and E_f equally
            //   is F_{f->i} / F_{i->f}
            // there is no reason, however, to have acceptance probabilities lower
            //   than the ratio of boltzmann weights on E_i and E_f, as otherwise
            //   we are spending more time sampling E_i (relative to E_f) than we would
            //   at the minimum temperature of the simulation
            const double acceptance_probability = [&]() -> double {
              if (backward_flux == 0) return 1;

              const double flux_ratio = (double(backward_flux * forward_norm)
                                         / (forward_flux * backward_norm));
              const double confidence = 1/double(backward_flux);
              const double probability = max(flux_ratio, confidence);
              const double max_probability = exp(-energy_change * beta_cap);

              if (probability < max_probability) return max_probability;
              return probability;
            }();

            // if we pass a probability test, accept the move
            if (rnd(generator) < acceptance_probability) {
              ns.state = proposed_state;
              new_energy = proposed_energy;
            } else {
              // otherwise reject it
              new_energy = old_energy;
            }
          }

          assert(new_energy < ns.energy_range);

          // update the energy and sample histograms
          // we don't care about other histograms during initialization
          ns.energy_histogram[new_energy]++;
          ns.update_sample_histogram(new_energy, old_energy);

          // as we move on with our lives (and this loop) the new energy turns old
          old_energy = new_energy;
        }

        // increment the cycle count and compute the density of states
        cycles++;
        ns.compute_dos_from_transitions();

        // compute the expected fractional sample error at an inverse temperature beta_cap
        sample_error = ns.fractional_sample_error(beta_cap);

        cout << fixed << setprecision(ceil(-log10(target_sample_error)) + 3)
             << sample_error << " " << cycles << endl;

        // write energy and transition data files
        const string header = (file_header +
                               "# initialization iterations: " +
                               to_string(cycles * iterations_per_cycle) + "\n");
        ns.write_energy_file(energy_file, header);
        ns.write_transitions_file(transitions_file, header);

        // loop until we satisfy the end condition for initialization
      } while (sample_error > target_sample_error);

      ns.compute_weights_from_dos(beta_cap);

    } else { // the weights file already exists, so read it in

      ns.read_weights_file(weights_file);

    } // complete determination of weights for an all temperature simulation

  } // complete initialization
  cout << endl;

  if (!suppress) {
    ns.print_energy_data();
    cout << endl;
  }

  // initialize a new random state and clear the histograms
  generator.seed(seed+1);
  ns.state = random_state(nodes, rnd, generator);
  ns.initialize_histograms();

  // -------------------------------------------------------------------------------------
  // Run simulation
  // -------------------------------------------------------------------------------------

  cout << "starting simulation" << endl << endl;
  clock_t last_data_print_time = time(NULL); // keep time to periodically write data files

  int new_energy; // energy of the state we move into
  int old_energy = ns.energy(); // energy of the last state
  assert(old_energy < ns.energy_range);
  for (long long ii = 0; ii < pow(10,log10_iterations); ii++) {

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
    assert(new_energy < ns.energy_range);

    // update energy and sample histograms
    ns.energy_histogram[new_energy]++;
    ns.update_sample_histogram(new_energy, old_energy);

    // update histograms which take O(X) time to update every X moves;
    //   otherwise we will be asymptotically spending all of simulation
    //   time on these updates
    if (ii % (ns.network.nodes * ns.pattern_number) == 0) {
      ns.distance_records[new_energy]++;
      ns.update_distance_logs(new_energy);
    }

    // update the old energy
    old_energy = new_energy;

    // if enough time has passed, write data files
    if ( difftime(time(NULL), last_data_print_time) > print_time * 60 ) {
      cout << "iterations: " << ii << endl;
      const string header = (file_header +
                             "# iterations: " + to_string((long long)ii) + "\n");
      ns.write_energy_file(energy_file, header);
      ns.write_distance_file(distance_file, header);
      cout << endl;
      last_data_print_time = time(NULL);
    }
  }

  // write final data files
  cout << "simulation complete" << endl;
  const string header = (file_header + "# iterations: "
                         + to_string((long long)pow(10,log10_iterations)) + "\n");
  ns.write_energy_file(energy_file, header);
  ns.write_distance_file(distance_file, header);

  if (!suppress) {
    cout << endl;
    ns.print_energy_data();
    cout << endl;
    ns.print_distances();
    cout << endl;
  }

  const int total_time = difftime(time(NULL), simulation_start_time);
  const int seconds = total_time % 60;
  const int minutes = (total_time / 60) % 60;
  const int hours = (total_time / (60 * 60)) % (24);
  const int days = total_time / (60 * 60 * 24);
  printf("total simulation time: %ud %uh %um %us\n", days, hours, minutes, seconds);

}
