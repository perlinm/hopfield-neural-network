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

  const int node_max = 1000; // hard-coded cap on network size

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
     "suppress some of the text that would normally be printed to the console")
    ("seed", po::value<long>(&seed)->default_value(0),
     "seed for random number generator")
    ("pattern_seed", po::value<long>(&pattern_seed)->default_value(0),
     "secondary random number generator seed used for generating patterns")
    ("suffix", po::value<bool>(&print_suffix)->default_value(false)->implicit_value(true),
     "print the file suffix associated with this simulation and exit")
    ;

  int nodes;
  int pattern_number;
  string pattern_file;

  po::options_description network_parameters("Network parameters",help_text_length);
  network_parameters.add_options()
    ("nodes", po::value<int>(&nodes)->default_value(10),
     "number of nodes in the network")
    ("patterns", po::value<int>(&pattern_number)->default_value(0,"nodes"),
     "number of random patterns with which to 'train' the network")
    ("pattern_file", po::value<string>(&pattern_file),
     "input file containing patterns used to 'train' the network")
    ;

  bool fixed_temp;
  double input_temp;
  int log10_iterations;
  int init_factor;
  int print_time;

  po::options_description simulation_options("General simulation options",
                                             help_text_length);
  simulation_options.add_options()
    ("temp", po::value<double>(&input_temp)->default_value(1), "simulation temperature")
    ("fixed_T", po::value<bool>(&fixed_temp)->default_value(false)->implicit_value(true),
     "run a fixed-temperature simulation")
    ("log10_iterations", po::value<int>(&log10_iterations)->default_value(5),
     "log10 of the number of iterations for which to simulate")
    ("init_factor", po::value<int>(&init_factor)->default_value(1),
     "set iterations per inialozation cycle to [pattern_number] * 10^(init_factor)")
    ("print_time", po::value<int>(&print_time)->default_value(30),
     "time (in minutes) between intermediate data file dumps")
    ;

  bool only_init;
  double target_sample_error;

  po::options_description all_temps_options("All temperature simulation options",
                                            help_text_length);
  all_temps_options.add_options()
    ("only_init", po::value<bool>(&only_init)->default_value(false)->implicit_value(true),
     "exit after initialization")
    ("sample_error", po::value<double>(&target_sample_error)->default_value(0.02,"0.02"),
     "the initialization routine terminates when it achieves this"
     " expected fractional sample error at the simulation temperature")
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

  // start the clock!
  clock_t simulation_start_time = time(NULL);

  // -------------------------------------------------------------------------------------
  // Process and run sanity checks on inputs
  // -------------------------------------------------------------------------------------

  fs::create_directory(data_dir);

  // we should have at least two nodes
  if (nodes < 2) {
    cout << "the network should consist of at least two nodes" << endl;
    return -1;
  }

  // in order to avoid accidentally crashing our computer by reserving all of its memory,
  //   don't allow attempts to simulate too large of a network
  assert(nodes <= node_max);

  // by default, use the same number of patterns as nodes
  assert(pattern_number >= 0);
  if (pattern_number == 0) pattern_number = nodes;

  assert(log10_iterations > 0);
  assert(init_factor > 0);

  // make sure that iteration counters/factors aren't too large
  assert(log10(nodes) + log10_iterations < log10(LONG_MAX));
  assert(log10(nodes) + log10(pattern_number) + init_factor < log10(LONG_MAX));

  // if we specified a pattern file, make sure it exists
  if (!pattern_file.empty() && !fs::exists(pattern_file)) {
    cout << "the specified pattern file does not exist:" << endl
         << pattern_file << endl;
    return -1;
  }
  const bool using_pattern_file = !pattern_file.empty();

  // initialize random number generator
  uniform_real_distribution<double> rnd(0.0,1.0); // uniform distribution on [0,1)
  mt19937_64 generator; // use the 64-bit Mersenne Twister 19937 generator

  // randomize seeds using the number of nodes and patterns
  size_t seed_hash = seed;
  bo::hash_combine(seed_hash, pattern_seed);
  bo::hash_combine(seed_hash, nodes);
  bo::hash_combine(seed_hash, pattern_number);
  seed += seed_hash;
  pattern_seed += seed_hash;

  // move pattern_seed far from the regular seed in order to avoid collisions
  pattern_seed +=  nodes + pattern_number + LONG_MAX;

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

    // set the number of nodes and number of patterns
    nodes = patterns[0].size();
    pattern_number = patterns.size();

  } else {

    // if we are not using a pattern file, generate random patterns
    for (int ii = 0; ii < pattern_number; ii++) {
      patterns.push_back(random_state(nodes, rnd, generator));
    }

  }

  // make sure that all patterns contain the same number of nodes
  for (int ii = 1, size = patterns.size(); ii < size; ii++) {
    if (patterns[ii-1].size() != patterns[ii].size()){
      printf("patterns %u and %u do not have the same size!\n", ii-1, ii);
      return -1;
    }
  }

  // make a hash of the temperature, patterns, and (if appropriate) target sample error
  //   to "identify" this simulation
  const size_t hash = [&]() -> size_t {
    size_t running_hash = input_temp;
    for (int pp = 0; pp < pattern_number; pp++) {
      for (int nn = 0; nn < nodes; nn++) {
        bo::hash_combine(running_hash, size_t(patterns[pp][nn]));
      }
    }
    if (!fixed_temp) {
      bo::hash_combine(running_hash, target_sample_error);
    }
    return running_hash;
  }();

  // put together a suffix to tag all data files read/written by this simulation
  const string temp_tag = ("-" + string(fixed_temp ? "f" : "") + "100T"
                           + string(input_temp < 0 ? "n" : "")
                           + to_string(int(round(100*input_temp))));
  const string node_tag = "-N" + to_string(nodes);
  const string pattern_tag = "-P" + to_string(pattern_number);
  const string file_suffix = (node_tag + pattern_tag + temp_tag
                              + "-h" + to_string(hash) + ".txt");

  // if all we were after was the suffix, print it and exit
  if (print_suffix) {
    cout << file_suffix << endl;
    return 0;
  }

  // define paths to data files
  const string transitions_file
    = (fs::path(data_dir) / fs::path("transitions" + file_suffix)).string();
  const string weights_file
    = (fs::path(data_dir) / fs::path("weights" + file_suffix)).string();
  const string energy_file
    = (fs::path(data_dir) / fs::path("energies" + file_suffix)).string();
  const string distance_file
    = (fs::path(data_dir) / fs::path("distances" + file_suffix)).string();
  const string state_file
    = (fs::path(data_dir) / fs::path("states" + file_suffix)).string();

  // construct network simulation object with a random initial state
  generator.seed(seed);
  network_simulation ns(patterns, random_state(nodes, rnd, generator), fixed_temp);

  // header for all data files
  stringstream file_header_stream;
  file_header_stream << "# nodes: " << ns.network.nodes << endl
                     << "# patterns: " << ns.pattern_number << endl
                     << "# input_temp: " << input_temp << endl
                     << "# energy_scale: " << ns.network.energy_scale << endl
                     << "# energy_range: " << ns.energy_range << endl
                     << "# max_de: " << ns.max_de << endl;
  if (!fixed_temp) {
    file_header_stream << "# target_sample_error: " << target_sample_error << endl;
  }
  const string file_header = file_header_stream.str();

  // simulation temperature in the same units as those used for our energies
  const double temp = input_temp * ns.network.nodes / ns.network.energy_scale;

  // number of moves per initialization cycle
  const long moves_per_init_cycle
    = ns.network.nodes * ns.pattern_number * pow(10, init_factor);

  // print some info about the simulation
  cout << "nodes: " << ns.network.nodes << endl
       << "pattern number: " << ns.pattern_number << endl
       << "temperature: " << input_temp << endl
       << "energy scale: " << ns.network.energy_scale << endl
       << "maximum energy: " << ns.network.max_energy << endl
       << "maximum energy change: " << ns.network.max_energy_change << endl;
  if (!fixed_temp) {
    cout << "target sample error: " << target_sample_error << endl;
  }
  cout << endl;

  if (!suppress) {
    ns.print_patterns();
    cout << endl;
    ns.network.print_couplings();
    cout << endl;
  }

  // -------------------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------------------

  clock_t last_data_print_time = time(NULL); // keep time to periodically write data files

  if (fixed_temp) {
    // run for one initialization cycle in order to (approximately) equilibriate

    cout << "starting a fixed temperature initialization routine" << endl;

    int current_energy = ns.energy();
    assert(current_energy < ns.energy_range);
    for (long ii = 0; ii < moves_per_init_cycle; ii++) {

      // pick a random node to possibly flip,
      //   and compute the change in energy from flipping it
      const int node = floor(rnd(generator) * ns.network.nodes);
      const int energy_change = ns.node_flip_energy_change(node);

      // if we pass a probability test, accept this move (i.e. node flip)
      if (rnd(generator) < ns.move_probability(current_energy, energy_change, temp)) {
        ns.state[node] = !ns.state[node];
        current_energy += energy_change;
      }
      assert(current_energy < ns.energy_range);

    }

  } else { // if we are not running a fixed-temperature simulation
    // initialize weight array for an all temperature simulation

    // unless find a file which contains the weights we need for this simulation,
    //   run the standard initialization routine
    if (!fs::exists(weights_file)) {

      cout << "starting all-temperature initialization routine..." << endl
           << "moves per initialization cycle: " << moves_per_init_cycle << endl;

      // if we find a file with a transition matrix for this simulation, read it in!
      // even if this file was generated by an unfinished initialization process,
      //   if the simulation hash is the same -- the data is perfectly good
      if (fs::exists(transitions_file)) {
        ns.read_transitions_file(transitions_file);
      }

      cout << "sample_error cycle_number" << endl;

      // number of initialization cycles we have completed
      int cycles = 0;
      // expected fractional error in sample count at the simulation temperature
      double sample_error;

      int new_energy; // energy of the new state after every move
      int old_energy = ns.energy(); // energy of the network state before the last move
      assert(old_energy < ns.energy_range);
      do { // while (sample_error > target_sample_error)
        // run for one initialization cycle
        for (long ii = 0; ii < moves_per_init_cycle; ii++) {

          // pick a random node to possibly flip,
          //   and compute the change in energy from flipping it
          const int node = floor(rnd(generator) * ns.network.nodes);
          const int energy_change = ns.node_flip_energy_change(node);
          const int proposed_energy = old_energy + energy_change;
          assert(abs(energy_change) <= ns.max_de);

          // the transition histogram contains information about proposed moves, not
          //   accepted moves, so we sample it always (even if the move is later rejected)
          ns.update_transition_histogram(old_energy, energy_change);

          if ((temp > 0 && energy_change <= 0) || (temp < 0 && energy_change >= 0)) {
            // always accept moves to a lower (higher) energy in a positive (negative)
            //   temperature simulation
            ns.state[node] = !ns.state[node];
            new_energy = proposed_energy;

          } else { // accept the moves with some to-be-determined probability

            // in order to get good statistics on the transition matrix,
            //   we wish to sample all energies as equally as we can
            // if the forward flux F_{i->f} of proposed moves from E_i to E_f is,
            //   say, twice the backwards flux F_{f->i}, then to sample E_i and E_f
            //   equally we should reject half of the proposed moves from E_i to E_f
            // in general, the move probability necessary to sample E_i and E_f equally
            //   is F_{f->i} / F_{i->f} (see: detailed balance)
            const double move_probability = [&]() -> double {
              // compute the (unnormalized) flux of proposed moves forward and backward
              //   between the current and proposed energies
              const long backward_moves = ns.transitions(proposed_energy, -energy_change);
              const long forward_moves = ns.transitions(old_energy, energy_change);

              // if we have never made the backward transition f->i, accept this move
              if (backward_moves == 0) return 1;

              // compute normalization factor for both transition fluxes
              const long backward_norm = ns.transitions_from(proposed_energy);
              const long forward_norm = ns.transitions_from(old_energy);

              // compute the flux ratio F_{f->i} / F_{i->f}
              const double flux_ratio = (double(backward_moves * forward_norm)
                                         / (forward_moves * backward_norm));

              // enforce a minimum acceptance probability based on:
              //  i) how many times we have tried to make the backward move f->i;
              //     if we have barely ever tried to move f->i, we want to be more likely
              //     to move into E_f to gather more statistics on transitions out of it
              // ii) the ratio of boltzmann weights on E_i and E_f at the minimum
              //     temperature of the simulation; otherwise, we would be wasting our
              //     time oversampling E_i relative to E_f
              const double sample_floor = 1.0/backward_moves;
              const double boltzmann_floor = exp(-energy_change / temp);
              const double min_probability = max(sample_floor, boltzmann_floor);

              return max(flux_ratio, min_probability);
            }();

            // if we pass a probability test, accept this move (i.e. node flip)
            if (rnd(generator) < move_probability) {
              ns.state[node] = !ns.state[node];
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

        // compute the expected fractional sample error at the simulation temperature
        sample_error = ns.fractional_sample_error(temp);

        // print helpful console text:
        //   the current fractional sample error
        //   and the number of initialization cycles we have completed
        cout << fixed << setprecision(ceil(-log10(target_sample_error)) + 3)
             << sample_error << " " << cycles << endl;

        // if enough time has passed, write energy and transition data files
        if ( difftime(time(NULL), last_data_print_time) > print_time * 60 ) {
          const string header = (file_header +
                                 "# initialization moves: " +
                                 to_string(cycles * moves_per_init_cycle) + "\n");
          ns.write_energy_file(energy_file, header);
          ns.write_transitions_file(transitions_file, header);
          last_data_print_time = time(NULL);
        }

        // repeat initialization cycles until we satisfy the initialization end condition
      } while (sample_error > target_sample_error);

      // once we have initialized, compute the weight array and write it to a file
      ns.compute_weights_from_dos(temp);
      ns.write_weights_file(weights_file, file_header);

    } else { // the weights file already exists, so read it in

      ns.read_weights_file(weights_file);

    } // complete initialization for an all temperature simulation

    if (!suppress) {
      ns.print_energy_data();
      cout << endl;
    }

    // initialize a new random state and clear the data histograms
    generator.seed(seed+1);
    ns.state = random_state(nodes, rnd, generator);
    ns.initialize_histograms();

    const int init_time = difftime(time(NULL), simulation_start_time);
    cout << "initialization time: " << time_string(init_time) << endl;

  } // complete initialization
  cout << endl;

  // if we only wanted to initialize, exit now
  if (only_init) return 0;

  // -------------------------------------------------------------------------------------
  // Run simulation
  // -------------------------------------------------------------------------------------

  cout << "starting simulation" << endl << endl;

  int new_energy; // energy of the state we move into
  int current_energy = ns.energy(); // energy of the last state
  assert(current_energy < ns.energy_range);
  long simulation_moves = ns.network.nodes * pow(10,log10_iterations);
  for (long ii = 0; ii < simulation_moves; ii++) {

    // pick a random node to possibly flip,
    //   and compute the change in energy from flipping it
    const int node = floor(rnd(generator) * ns.network.nodes);
    const int energy_change = ns.node_flip_energy_change(node);

    // if we pass a probability test, accept this move (i.e. node flip)
    if (rnd(generator) < ns.move_probability(current_energy, energy_change, temp)) {
      ns.state[node] = !ns.state[node];
      new_energy = current_energy + energy_change;
    } else {
      // otherwise reject it
      new_energy = current_energy;
    }
    assert(new_energy >= 0);
    assert(new_energy < ns.energy_range);

    // update histograms
    ns.energy_histogram[new_energy]++;
    ns.update_sample_histogram(new_energy, current_energy);
    ns.update_state_histograms();

    // the distace log takes O(pattern_number) time to update,
    //   so only upate it every [pattern_number] moves
    if (ii % ns.pattern_number == 0) {
      ns.update_distance_logs(new_energy);
    }

    // update the old energy
    current_energy = new_energy;

    // if enough time has passed, write data files
    if ( difftime(time(NULL), last_data_print_time) > print_time * 60 ) {
      cout << "moves: " << ii << endl;
      const string header = (file_header +
                             "# moves: " + to_string(ii) + "\n");
      ns.write_energy_file(energy_file, header);
      ns.write_distance_file(distance_file, header);
      last_data_print_time = time(NULL);
    }
  } // finished with simulation!

  // write final data files
  cout << "simulation complete" << endl;
  const string header = (file_header + "# moves: "
                         + to_string(simulation_moves) + "\n");
  ns.write_energy_file(energy_file, header);
  ns.write_distance_file(distance_file, header);
  ns.write_state_file(state_file, header);

  // print possibly helpful console text
  if (!suppress) {
    if (!ns.fixed_temp) {
      ns.compute_dos_from_energy_histogram();
      cout << endl;
      ns.print_energy_data();
      cout << endl;
    }
    ns.print_distances();
    cout << endl;
    if (ns.fixed_temp) {
      ns.print_states();
      cout << endl;
    }
  }

  // print total runtime and exit
  const int total_time = difftime(time(NULL), simulation_start_time);
  cout << "total run time: " << time_string(total_time) << endl;
  return 0;

}
