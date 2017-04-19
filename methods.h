#pragma once

#include <random> // for randomness

using namespace std;

string time_string(const int total_seconds);

// greatest common divisor
int gcd(const int a, const int b);

// generate a random state
vector<bool> random_state(const int nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator);

struct hopfield_network {

  // number of nodes in network
  int nodes;

  // coupling constants between nodes
  vector<vector<int>> couplings;

  // energy resolution necessary to keep track of all distinct energies
  int energy_scale;

  // maximum energy of the network, and maximum by which
  //   this energy can change by flipping one spin
  // in units of energy_scale
  int max_energy;
  int max_energy_change;

  // hopfield network constructor
  hopfield_network(const vector<vector<bool>>& patterns);

  // (index of) energy of the network in a given state
  int energy(const vector<bool>& state) const;

  // convert energy index to an "actual" energy
  int actual_energy(const int energy_index) const;

  // print coupling matrix
  void print_couplings() const;

};

// network simulation object
struct network_simulation {

  // is this a fixed temperature simulation?
  bool fixed_temp;

  // patterns used to construct the network, and the number of them
  const vector<vector<bool>> patterns;
  const int pattern_number;

  // the network itself
  const hopfield_network network;

  // the energy range, and the max amount by which the energy can change in one move
  const int energy_range;
  const int max_de;

  // the energy at which entropy is maximized
  // i.e. the most common energy in the space of all states
  int entropy_peak = 0;

  // the current network state stored in simulation
  vector<bool> state;

  // histogram containing the number of times we have seen every energy
  vector<long> energy_histogram;

  // logarithm of the weights which determine the probability
  //   of accepting a move between two energies during simulation
  // note: only used in all temperature simulations
  vector<double> ln_weights;

  // logarithm of the (unnormalized) density of states
  // note: only used in all temperature simulations
  vector<double> ln_dos;

  // stores the number times we have proposed a move
  //   from a given energy with a specified energy difference
  // indexed by (energy, change in energy)
  // note: only used in all temperature simulations
  vector<vector<long>> transition_histogram;

  // visit_log[ee] answers the question: have we visited the energy ee
  // at least once since the last observation of a maximual entropy state?
  // note: only used in all temperature simulations
  vector<bool> visit_log;

  // stores the number of independent samples of any energy
  // two samples of a given energy are considered independent if
  //   we have made a visit to the maximal entropy state between them
  // note: only used in all temperature simulations
  vector<long> sample_histogram;

  // number of times we have updated the state histogram
  long state_records = 0;

  // stores the number of times we have seen each node in the state 1
  // dividing state_histograms[nn] by state_records tells us the mean state of node nn
  vector<long> state_histograms;

  // number of times we have recorded distance from patterns
  //   either at a given energy (all_temp_distance_records),
  //   or just the total number (fixed_temp_distance_records)
  vector<long> all_temp_distance_records;
  long fixed_temp_distance_records = 0;

  // stores the sum of all distances from every pattern at each energy
  // indexed by (energy, pattern)
  // dividing all_temp_distance_logs[ee][pp] by distance_records[ee] tells us
  //   the mean distance from pattern pp at the energy ee
  // dividing fixed_temp_distance_logs[pp] by distance_records tells us
  //   the mean distance from pattern pp at the simulation temperature
  vector<vector<long>> all_temp_distance_logs;
  vector<long> fixed_temp_distance_logs;

  // constructor for the network simulation object
  network_simulation(const vector<vector<bool>>& patterns,
                     const vector<bool>& initial_state,
                     const bool fixed_temp);

  // -------------------------------------------------------------------------------------
  // Access methods for histograms and matrices
  // -------------------------------------------------------------------------------------

  // number of attempted transitions from a given energy with a specified energy change
  long transitions(const int energy, const int energy_change) const;

  // number of attempted transitions from a given energy into any other energy
  long transitions_from(const int energy) const;

  // elements of the actual normalized transition matrix:
  //   the probability of proposing a move from a given initial energy
  //   into a specific final energy
  double transition_matrix(const int final_energy, const int initial_energy) const;

  // -------------------------------------------------------------------------------------
  // Methods used in simulation
  // -------------------------------------------------------------------------------------

  // compute energy change due to flipping a node from its current state
  int node_flip_energy_change(const int node) const;

  // the energy of a given state
  int energy(const vector<bool>& state) const { return network.energy(state); };
  int energy() const { return energy(state); };

  // probability to accept a move
  double move_probability(const int current_energy, const int energy_change,
                          const double temp);

  // initialize all tables and histograms
  void initialize_histograms();

  // update histograms with an observation
  void update_distance_logs(const int energy);
  void update_state_histograms();
  void update_sample_histogram(const int new_energy, const int old_energy);
  void update_transition_histogram(const int energy, const int energy_change);

  // compute density of states from the transition matrix
  void compute_dos_from_transitions();

  // compute density of states from the energy histogram
  void compute_dos_from_energy_histogram();

  // construct weight array from the density of states
  // WARNING: assumes that the density of states is up to date
  void compute_weights_from_dos(const double temp);

  // expectation value of fractional sample error at the simulation temperature
  // WARNING: assumes that the density of states is up to date
  double fractional_sample_error(const double temp) const;

  // -------------------------------------------------------------------------------------
  // Writing/reading data files
  // -------------------------------------------------------------------------------------

  void write_transitions_file(const string transitions_file,
                              const string file_header) const;
  void write_weights_file(const string weights_file, const string file_header) const;
  void write_energy_file(const string energy_file, const string file_header) const;
  void write_distance_file(const string distance_file, const string file_header) const;
  void write_state_file(const string state_file, const string file_header) const;

  void read_transitions_file(const string transitions_file);
  void read_weights_file(const string weights_file);

  // -------------------------------------------------------------------------------------
  // Printing methods
  // -------------------------------------------------------------------------------------

  // print patterns defining the simulated network
  void print_patterns() const;

  // print energy histogram, sample histogram, density of states, and weights
  void print_energy_data() const;

  // print expectation value of each spin spin at each energy
  void print_expected_states() const;

  // print expectation value of distances from each pattern at each energy
  void print_distances() const;

  // print expectation value of each state at each energy
  void print_states() const;

};
