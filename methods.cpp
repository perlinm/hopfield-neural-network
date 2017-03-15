#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness
#include <sstream> // for string streams
#include <algorithm> // for sort method
#include <fstream> // for stream objects

#include "methods.h"

using namespace std;

// greatest common divisor
int gcd(const int a, const int b) {
  if (b == 0) return a;
  else return gcd(b, a % b);
}

// generate a random state
vector<bool> random_state(const int nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator) {
  vector<bool> state(nodes);
  for (int ii = 0; ii < nodes; ii++) {
    state[ii] = (rnd(generator) < 0.5);
  }
  return state;
}

// make a random change to a given state using a random number on [0,1)
vector<bool> random_change(const vector<bool>& state, const double random) {
  // pick a node determined by the random number,
  // and construct a new state in which that node is flipped
  const int node = floor(random*state.size());
  vector<bool> new_state = state;
  new_state[node] = !new_state[node];
  return new_state;
}

// hopfield network constructor
hopfield_network::hopfield_network(const vector<vector<bool>>& patterns) {
  // number of nodes in network
  nodes = patterns[0].size();

  // generate interaction matrix from patterns
  // note: these couplings are a factor of (nodes) greater than the regular definition
  couplings = vector<vector<int>>(nodes);
  for (int ii = 0; ii < nodes; ii++) {
    couplings[ii] = vector<int>(nodes, 0);
    for (int jj = 0; jj < nodes; jj++) {
      if (jj == ii) continue;
      for (int pp = 0, size = patterns.size(); pp < size; pp++) {
        const int pattern_contribution = (2*patterns[pp][ii]-1)*(2*patterns[pp][jj]-1);
        couplings[ii][jj] += pattern_contribution;
      }
    }
  }

  // determine the maximum energy change possible in one move, as well as
  //   the energy resolution (energy_scale) needed to keep track of all energies
  max_energy_change = 0;
  energy_scale = 0;
  for (int ii = 0; ii < nodes; ii++) {
    int node_energy = 0;
    for (int jj = 0; jj < nodes; jj++) {
      node_energy += 2*abs(couplings[ii][jj]);
    }
    max_energy_change = max(node_energy, max_energy_change);
    energy_scale = gcd(node_energy, energy_scale);
  }

  // compute an upper bound on the maximum energy achievable by network
  // given that the actual energy is
  //   1/2 \sum_{i,j} J_{ij} s_i s_j with s_i, s_j in {-1, 1}
  //   an upper bound is 1/2 \sum_{i,j} |J_{ij}| = \sum_{i,j>i} |J_{ij}|
  max_energy = 0;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = ii + 1; jj < nodes; jj++) {
      max_energy += abs(couplings[ii][jj]);
    }
  }
  // fix up max_energy so that for any energy E we observe,
  //   (E + max_energy) is divisible by energy_scale
  max_energy -= (max_energy + energy(patterns[0])) % energy_scale;

};

// (index of) energy of the network in a given state
int hopfield_network::energy(const vector<bool>& state) const {
  int energy = 0;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = ii + 1; jj < nodes; jj++) {
      energy -= couplings[ii][jj] * (2*state[ii]-1) * (2*state[jj]-1);
    }
  }
  return (energy + max_energy) / energy_scale;
}

// convert energy index to an "actual" energy
int hopfield_network::actual_energy(const int energy_index) const {
  return energy_index * energy_scale - max_energy;
}

// print coupling matrix
void hopfield_network::print_couplings() const {
  // determin the largest coupling constant, which tells us how wide to make
  // the columns of the matrix
  int largest_coupling = 0;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = 0; jj < nodes; jj++) {
      largest_coupling = max(abs(couplings[ii][jj]), largest_coupling);
    }
  }
  // matrix column width
  const int width = log10(largest_coupling) + 2;

  // print the matrix
  cout << "coupling matrix:" << endl;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = 0; jj < nodes; jj++) {
      cout << setw(width) << couplings[ii][jj] << " ";
    }
    cout << endl;
  }
}

// network simulation constructor
network_simulation::network_simulation(const vector<vector<bool>>& patterns,
                                       const vector<bool>& initial_state) :
  patterns(patterns),
  pattern_number(patterns.size()),
  network(hopfield_network(patterns)),
  energy_range(2*network.max_energy/network.energy_scale),
  max_de(network.max_energy_change/network.energy_scale)
{
  entropy_peak = energy_range / 2; // an initial guess
  state = initial_state;
  initialize_histograms();
  ln_weights = vector<double>(energy_range, 0);
  ln_dos = vector<double>(energy_range, 0);
};

// ---------------------------------------------------------------------------------------
// Access methods for histograms and matrices
// ---------------------------------------------------------------------------------------

// number of attempted transitions from a given energy with a specified energy change
int network_simulation::transitions(const int energy, const int energy_change) const {
  return transition_histogram[energy][energy_change + max_de];
}

// number of attempted transitions from a given energy into any other energy
int network_simulation::transitions_from(const int energy) const {
  int count = 0;
  for (int de = -max_de; de <= max_de; de++) {
    count += transitions(energy, de);
  }
  return count;
}

// elements of the actual normalized transition matrix:
//   the probability of moving from a given initial energy into a specific final energy
double network_simulation::transition_matrix(const int final_energy,
                                             const int initial_energy) const {
  const int energy_change = final_energy - initial_energy;
  if (abs(energy_change) > max_de) return 0;

  // normalization factor: sum of all transitions from the initial energy
  const int normalization = transitions_from(initial_energy);

  // if the normalization factor is zero, it's because we have never seen this energy
  // by default, set these elements of the transition energy to zero
  if (normalization == 0) return 0;

  return double(transitions(initial_energy, energy_change)) / normalization;
}

// ---------------------------------------------------------------------------------------
// Methods used in simulation
// ---------------------------------------------------------------------------------------

// initialize all tables: distance log, sample histogram, energy transitions
void network_simulation::initialize_histograms() {
  energy_histogram = vector<unsigned long long int>(energy_range, 0);
  distance_records = vector<unsigned long long int>(energy_range, 0);

  visit_log = vector<bool>(energy_range, true);
  sample_histogram = vector<unsigned long long int>(energy_range, 0);

  distance_logs = vector<vector<unsigned long long int>>(energy_range);
  transition_histogram = vector<vector<unsigned long long int>>(energy_range);
  for (int ee = 0; ee < energy_range; ee++) {
    distance_logs[ee] = vector<unsigned long long int>(pattern_number, 0);
    transition_histogram[ee] = vector<unsigned long long int>(2*max_de + 1, 0);
  }
}

void network_simulation::update_distance_logs(const int energy) {
  // for each pattern
  for (int pp = 0; pp < pattern_number; pp++) {
    // find the distance between the current state and patterns[pp]
    int overlap = 0;
    for (int ii = 0; ii < network.nodes; ii++) {
      overlap += (state[ii] == patterns[pp][ii]);
    }
    const int distance = min(overlap, network.nodes - overlap);
    // add distance from pattern[pp] to the distance histogram
    distance_logs[energy][pp] += distance;
  }
}

void network_simulation::update_sample_histogram(const int new_energy,
                                                 const int old_energy) {
  // if we have not yet visited this energy since the last observation
  //   of a maximual entropy state, add to the sample histogram
  if (!visit_log[new_energy]) {
    visit_log[new_energy] = true;
    sample_histogram[new_energy]++;
  }

  // if we are at the entropy peak, reset the visit log and return
  if (new_energy == entropy_peak) {
    if (old_energy != entropy_peak) {
      visit_log = vector<bool>(energy_range, false);
    } else {
      // if we were at the entropy peak the last time we updated the sample histogram,
      //   we only need to reset the visit log at the entropy peak itself,
      //   as it is already false everywhere else
      visit_log[entropy_peak] = false;
    }
    return;
  }

  // determine whether we have crossed the entropy peak since the last move
  const bool above_peak_now = (new_energy > entropy_peak);
  const bool above_peak_before = (old_energy > entropy_peak);

  // if we did not cross the entropy peak, return
  if (above_peak_now == above_peak_before) return;

  // if we did cross the entropy peak, reset the appropriate parts of the visit log
  if (above_peak_now) {
    // reset visit log below the entropy peak
    for (int ee = 0; ee < entropy_peak; ee++) {
      visit_log[ee] = false;
    }
  } else { // if below peak now
    // reset visit log above the entropy peak
    for (int ee = entropy_peak + 1; ee < energy_range; ee++) {
      visit_log[ee] = false;
    }
  }
}

void network_simulation::update_transition_histogram(const int energy,
                                                     const int energy_change) {
  transition_histogram[energy][energy_change + max_de]++;
}

// compute density of states from the transition matrix
void network_simulation::compute_dos_from_transitions() {

  // keep track of the maximal value of ln_dos
  double max_ln_dos = 0;

  // sweep up through all energies to "bootstrap" the density of states
  ln_dos[0] = 0; // seed a value of ln_dos for the sweep
  for (int ee = 1; ee < energy_range; ee++) {

    // pick an initial guess for the density of states at the energy ee
    ln_dos[ee] = ln_dos[ee-1];

    // if we have never actually seen this energy, we don't care to correct
    //   the previous guess, so go on to the next energy
    if (energy_histogram[ee] == 0) continue;

    // given our guess for the density of states at the energy ee,
    //   compute the net transition fluxes up to ee from below (i.e. from lower energies),
    //   and down from ee (i.e. to lower energies)
    double flux_up_to_this_energy = 0;
    double flux_down_from_this_energy = 0;
    for (int smaller_ee = ee - max_de; smaller_ee < ee; smaller_ee++) {
      if (smaller_ee < 0) continue;
      // we divide both normalized fluxes by the guess for the density of states at ee
      //   in order to avoid potential numerical overflows (and reduce numerical error)
      // as we will actually be interested in the ratio of these fluxes,
      //   multiplying them both by a constant factor has no consequence
      flux_up_to_this_energy += (exp(ln_dos[smaller_ee] - ln_dos[ee])
                                 * transition_matrix(ee, smaller_ee));
      flux_down_from_this_energy += transition_matrix(smaller_ee, ee);
    }

    // in an equilibrium ensemble of simulations, the two fluxes we computed above
    //   should be the same; if they are not, it is because our guess for
    //   the density of states was incorrect
    // we therefore multiply the density of states by the factor which would make
    //   these fluxes equal, which is presicely the ratio of the fluxes
    if (flux_up_to_this_energy > 0 && flux_down_from_this_energy > 0) {
      ln_dos[ee] += log(flux_up_to_this_energy/flux_down_from_this_energy);
    }

    // keep track of the maximum value of ln_dos,
    //  and the energy at which the density of states is maximal (i.e. the entropy peak)
    if (ln_dos[ee] > max_ln_dos) {
      max_ln_dos = ln_dos[ee];
      entropy_peak = ee;
    }

  }

  // subtract off the maximal value of ln_dos from the entire array,
  //   which normalizes the density of states to 1 at the entropy peak
  for (int ee = 0; ee < energy_range; ee++) {
    ln_dos[ee] -= max_ln_dos;
  }

}

// compute density of states from the energy histogram
void network_simulation::compute_dos_from_energy_histogram() {
  // keep track of the maximal value of ln_dos
  double max_ln_dos = 0;
  for (int ee = 0; ee < energy_range; ee++) {
    ln_dos[ee] = log(energy_histogram[ee]) - ln_weights[ee];
    max_ln_dos = max(ln_dos[ee], max_ln_dos);
  }
  // subtract off the maximal value of ln_dos from the entire array,
  //   which normalizes the density of states to 1 at the entropy peak
  for (int ee = 0; ee < energy_range; ee++) {
    ln_dos[ee] -= max_ln_dos;
  }
};

// construct weight array from the density of states
// WARNING: assumes that the density of states is up to date
void network_simulation::compute_weights_from_dos(const double beta_cap) {

  // reset the weight array
  ln_weights = vector<double>(energy_range, 0);

  if (beta_cap > 0) {
    // if we care about positive temperatures, then we are interested in low energies
    // identify the lowest energy we have seen
    int lowest_seen_energy = 0;
    for (int ee = 0; ee < energy_range; ee++) {
      if (energy_histogram[ee] != 0) {
        lowest_seen_energy = ee;
        break;
      }
    }

    // in the relevant range of observed energies, set weights appropriately,
    //   but never set any weight higher than we would at an inverse temperature beta_cap
    double excess_weight = 0;
    const double max_diff = energy_range * beta_cap;
    for (int ee = entropy_peak - 1; ee >= lowest_seen_energy; ee--) {
      ln_weights[ee] = -ln_dos[ee];
      const double diff = ln_weights[ee] - ln_weights[ee+1];
      if (diff > max_diff) {
        excess_weight += max_diff - diff;
        ln_weights[ee] -= excess_weight;
      }
    }
    // below all observed energies, use fixed temperature weights
    for (int ee = 0; ee < lowest_seen_energy; ee++) {
      ln_weights[ee] = (-ln_dos[lowest_seen_energy]
                        + (lowest_seen_energy - ee) * beta_cap);
    }

  } else { // if beta_cap < 0

    // if we care about negative temperatures, then we are interested in high energies
    // identify the highest energy we have seen
    int highest_seen_energy = energy_range;
    for (int ee = energy_range - 1; ee >= 0; ee--) {
      if (energy_histogram[ee] != 0) {
        highest_seen_energy = ee;
        break;
      }
    }

    // in the relevant range of observed energies, set weights appropriately
    //   but never set any weight higher than we would at an inverse temperature beta_cap
    double excess_weight = 0;
    const double max_diff = - energy_range * beta_cap;
    for (int ee = entropy_peak + 1; ee <= highest_seen_energy; ee++) {
      ln_weights[ee] = -ln_dos[ee];
      const double diff = ln_weights[ee] - ln_weights[ee-1];
      if (diff > max_diff) {
        excess_weight += max_diff - diff;
        ln_weights[ee] -= excess_weight;
      }
    }
    // above all observed energies, use fixed temperature weights
    for (int ee = highest_seen_energy + 1; ee < energy_range; ee++) {
      ln_weights[ee] = (-ln_dos[highest_seen_energy]
                        + (highest_seen_energy - ee) * beta_cap);
    }
  }

}

// expectation value of fractional sample error at a given inverse temperature
// WARNING: assumes that the density of states is up to date
double network_simulation::fractional_sample_error(const double beta_cap) const {

  // determine the lowest and highest energies we care about
  int lowest_energy;
  int highest_energy;
  if (beta_cap > 0) { // we care about low energies
    highest_energy = entropy_peak;
    // set lowest_energy to the lowest energy we have sampled
    for (int ee = 0; ee < entropy_peak; ee++) {
      if (sample_histogram[ee] != 0) {
        lowest_energy = ee;
        break;
      }
    }

  } else { // we care about low energies
    lowest_energy = entropy_peak;
    // set highest_energy to the highest energy we have sampled
    for (int ee = energy_range - 1; ee > entropy_peak; ee--) {
      if (sample_histogram[ee] != 0) {
        highest_energy = ee;
        break;
      }
    }
  }
  // the mean energy we care about
  const int mean_energy = (highest_energy + lowest_energy) / 2;

  // sum up the fractional error in sample counts with appropriate boltzmann factors
  long double error = 0;
  long double normalization = 0; // this is the partition function
  for (int ee = lowest_energy; ee < highest_energy; ee++) {
    if (sample_histogram[ee] != 0) {
      // offset ln_dos[ee] and the energy ee by their values at the mean energy
      //   we care about in order to avoid numerical overflows
      // this offset amounts to multiplying both (error) and (normalization) by
      //   a constant factor, which means that it does not affect (error/normalization)
      const double ln_dos_ee = ln_dos[ee] - ln_dos[mean_energy];
      const double energy = ee - mean_energy;
      const long double boltzmann_factor = exp(ln_dos_ee - energy * beta_cap);
      error += boltzmann_factor/sqrt(sample_histogram[ee]);
      normalization += boltzmann_factor;
    }
  }
  return error/normalization;
}

// ---------------------------------------------------------------------------------------
// Writing/reading data files
// ---------------------------------------------------------------------------------------

void network_simulation::write_transitions_file(const string transitions_file,
                                                const string file_header) const {
  cout << "writing transition matrix" << endl;
  ofstream transition_stream(transitions_file);
  transition_stream << file_header << endl
                    << "# (row)x(column) = (energy)x(de)" << endl;
  for (int ee = 0; ee < energy_range; ee++) {
    if (energy_histogram[ee] == 0) continue;
    transition_stream << network.actual_energy(ee)
                      << " " << transitions(ee, -max_de);
    for (int de = -max_de + 1; de <= max_de; de++) {
      transition_stream << " " << transitions(ee, de);
    }
    transition_stream << endl;
  }
  transition_stream.close();
}

void network_simulation::write_weights_file(const string weights_file,
                                            const string file_header) const {
  cout << "writing weight array" << endl;
  ofstream weight_stream(weights_file);
  weight_stream << file_header << endl
                << "# energy, ln_weight" << endl;
  for (int ee = 0; ee < energy_range; ee++) {
    if (energy_histogram[ee] == 0) continue;
    weight_stream << setprecision(numeric_limits<double>::max_digits10)
                  << network.actual_energy(ee) << " "
                  << ln_weights[ee] << endl;
  }
  weight_stream.close();
}

void network_simulation::write_energy_file(const string energy_file,
                                           const string file_header) const {
  cout << "writing energy histogram" << endl;
  ofstream energy_stream(energy_file);
  energy_stream << file_header << endl
                << "# energy, energy histogram, sample histogram" << endl;
  for (int ee = 0; ee < energy_range; ee++) {
    if (energy_histogram[ee] == 0)  continue;
    energy_stream << network.actual_energy(ee) << " "
                  << energy_histogram[ee] << " "
                  << sample_histogram[ee] << endl;
  }
  energy_stream.close();
}

void network_simulation::write_distance_file(const string distance_file,
                                             const string file_header) const {
  cout << "writing distance log" << endl;
  ofstream distance_stream(distance_file);
  distance_stream << file_header << endl
                  << "# energy, distance records, distance log..." << endl;
  for (int ee = 0; ee < energy_range; ee++) {
    if (energy_histogram[ee] == 0)  continue;
    distance_stream << network.actual_energy(ee)
                    << " " << distance_records[ee];
    for (int pp = 0; pp < pattern_number; pp++) {
      distance_stream << " " << distance_logs[ee][pp];
    }
    distance_stream << endl;
  }
  distance_stream.close();
}

void network_simulation::read_weights_file(const string weights_file,
                                           const double beta_cap) {
  ln_weights = vector<double>(energy_range, 0);

  // keep track of first and last zeroes in ln_weights
  bool first_zero_set = false;
  int first_zero, last_zero;

  // keep track of lowest and highest energies seen
  bool lowest_seen_energy_set = false;
  int lowest_seen_energy, highest_seen_energy;


  cout << "reading in weight array from file" << endl;
  ifstream input(weights_file.c_str());
  string line;
  string word;
  while (getline(input,line)) {
    if (line[0] == '#' || line.empty()) continue;
    stringstream line_stream(line);
    line_stream >> word;
    const int ee = (stoi(word) + network.max_energy) / network.energy_scale;
    energy_histogram[ee]++; // mark this energy as seen

    if (!lowest_seen_energy_set) {
      lowest_seen_energy_set = true;
      lowest_seen_energy = ee;
    }
    highest_seen_energy = ee;

    line_stream >> word;
    const double weight = stod(word);
    ln_weights[ee] = weight;

    if (weight == 0) {
      if (!first_zero_set) {
        first_zero_set = true;
        first_zero = ee;
      }
      last_zero = ee;
    }
  }

  // fill in the weight array within the range of seen energies
  for (int ee = lowest_seen_energy + 1; ee < highest_seen_energy; ee++) {
    if (ln_weights[ee] == 0) ln_weights[ee] = ln_weights[ee-1];
  }

  // set entropy peak and fill in the rest of the weight array
  if (beta_cap > 0) {
    entropy_peak = first_zero;
    for (int ee = 0; ee < lowest_seen_energy; ee++) {
      ln_weights[ee] = (ln_weights[lowest_seen_energy]
                        + (lowest_seen_energy - ee) * beta_cap);
    }

  } else { // if beta_cap < 0
    entropy_peak = last_zero;
    for (int ee = highest_seen_energy + 1; ee < energy_range; ee++) {
      ln_weights[ee] = (ln_weights[highest_seen_energy]
                        + (highest_seen_energy - ee) * beta_cap);
    }
  }

}

// ---------------------------------------------------------------------------------------
// Printing methods
// ---------------------------------------------------------------------------------------

// print patterns defining the simulated network
void network_simulation::print_patterns() const {
  const int energy_width = log10(network.max_energy) + 2;
  const int index_width = log10(pattern_number) + 1;

  // make list of the pattern energies
  vector<int> energies(pattern_number);
  for (int pp = 0; pp < pattern_number; pp++) {
    energies[pp] = energy(patterns[pp]);
  }

  // sort the pattern energies
  vector<int> sorted_energies = energies;
  sort(sorted_energies.begin(), sorted_energies.end());

  // printed[pp]: have we printed pattern pp?
  vector<bool> printed(pattern_number,false);

  // print patterns in order of decreasing energy
  cout << "(energy, index) pattern" << endl;
  for (int ss = pattern_number - 1; ss >= 0; ss--) {
    cout << "(" << setw(energy_width)
         << network.actual_energy(sorted_energies[ss]) << ", ";
    for (int pp = 0; pp < pattern_number; pp++) {
      if (energies[pp] == sorted_energies[ss] && !printed[pp]) {
        cout << setw(index_width) << pp << ")";
        for (int ii = 0; ii < network.nodes; ii++) {
          cout << " " << patterns[pp][ii];
        }
        printed[pp] = true;
        break;
      }
    }
    cout << endl;
  }
}

// for each energy observed, print the energy and the corresponding values in
//   the energy histogram, sample histogram, density of states, and weight
void network_simulation::print_energy_data() const {
  // get the maximum value in the energy histogram
  unsigned long long int most_observations = 0;
  for (int ee = 0; ee < energy_range; ee++) {
    most_observations = max(energy_histogram[ee], most_observations);
  }

  cout << "energy observations samples log10_dos ln10_weights" << endl;
  const int energy_width = log10(network.max_energy) + 2;
  const int energy_hist_width = log10(most_observations) + 1;
  const int sample_width = log10(sample_histogram[entropy_peak]) + 1;
  const int double_dec = 6; // decimal precision with which to print doubles
  for (int ee = energy_range - 1; ee >= 0; ee--) {
    const int observations = energy_histogram[ee];
    if (observations == 0) continue;
    cout << fixed
         << setw(energy_width)
         << network.actual_energy(ee) << " "
         << setw(energy_hist_width) << observations << " "
         << setw(sample_width) << sample_histogram[ee] << " "
         << setw(double_dec + 3) << setprecision(double_dec)
         << log10(exp(1)) * ln_dos[ee] << " "
         << setw(double_dec + 2) << setprecision(double_dec)
         << log10(exp(1)) * ln_weights[ee] << endl;
  }
}

// print expectation value of distances from each pattern at each energy
void network_simulation::print_distances() const {
  cout << "energy <d_1>, <d_2>, ..., <d_p>" << endl;
  const int energy_width = log10(network.max_energy) + 2;
  const int distance_dec = 6;
  for (int ee = energy_range - 1; ee >= 0; ee--) {
    // check that we have sampled this energy
    const unsigned long long int observations = distance_records[ee];
    if (observations == 0) continue;

    cout << setw(energy_width) << network.actual_energy(ee);
    for (int pp = 0; pp < pattern_number; pp++) {
      const double val = double(distance_logs[ee][pp]) / observations;
      const int prec = distance_dec - int(max(log10(val),0.));
      cout << " " << setw(distance_dec) << setprecision(prec) << val * 2 / network.nodes;
    }
    cout << endl;
  }
}
