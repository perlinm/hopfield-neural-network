#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness
#include <cassert> // for assertions
#include <algorithm> // for sort method

#include "methods.h"

using namespace std;

// greatest common divisor
int gcd(const int a, const int b) {
  if (b == 0) return a;
  else return gcd(b, a % b);
}

// distance between two states
int state_distance(const vector<bool>& s1, const vector<bool>& s2) {
  int distance = 0;
  for (int ii = 0, size = s1.size(); ii < size; ii++) {
    distance += (s1[ii] == s2[ii]);
  }
  return distance;
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

  // determine maximum energy achievable by network
  // as a heuristic, use 2*max_p(|E_p|), where E_p is the energy of pattern p
  // TODO: use a better estimate of the minimum/maximum energy
  max_energy = 0;
  for (int pp = 0, size = patterns.size(); pp < size; pp++) {
    int pattern_energy = 0;
    for (int ii = 0; ii < nodes; ii++) {
      for (int jj = ii + 1; jj < nodes; jj++) {
        pattern_energy
          -= couplings[ii][jj] * (2*patterns[pp][ii]-1) * (2*patterns[pp][jj]-1);
      }
    }
    max_energy = max(-2*pattern_energy, max_energy);
  }

  // determine the maximum energy change possible in one move, as well as
  //   the energy resolution (energy_scale) needed to keep track of all energies
  max_energy_change = 0;
  energy_scale = 2*max_energy;
  for (int ii = 0; ii < nodes; ii++) {
    int node_energy = 0;
    for (int jj = 0; jj < nodes; jj++) {
      node_energy += abs(couplings[ii][jj]);
    }
    max_energy_change = max(2*node_energy, max_energy_change);
    energy_scale = gcd(node_energy, energy_scale);
  }

};

// energy of the network in a given state
// note: this energy is equal to the "actual" energy (by the normal definition)
//       multiplied by a factor of (nodes/energy_scale),
//       and shifted down by a small, but constant amount determined
//       by our energy resolution
int hopfield_network::energy(const vector<bool>& state) const {
  int energy = 0;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = ii + 1; jj < nodes; jj++) {
      energy -= couplings[ii][jj] * (2*state[ii]-1) * (2*state[jj]-1);
    }
  }
  return (energy + max_energy) / energy_scale;
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
  // set patterns, construct network
  patterns(patterns),
  network(hopfield_network(patterns)),
  energy_range(2*network.max_energy/network.energy_scale),
  max_de(network.max_energy_change/network.energy_scale)
{
  entropy_peak = (energy_range + 1) / 2;
  state = initial_state;
  initialize_histograms();
  ln_weights = vector<double>(energy_range, 1);
  ln_dos = vector<double>(energy_range, 0);
};

// ---------------------------------------------------------------------------------------
// Access methods for histograms and matrices
// ---------------------------------------------------------------------------------------

// number of attempted transitions from a given energy with a specified energy change
int network_simulation::transitions(const int energy, const int energy_change) const {
  return energy_transitions[energy][energy_change + max_de];
}

// number of attempted transitions from a given energy into any other energy
int network_simulation::transitions_from(const int energy) const {
  int sum = 0;
  for (int de = -max_de; de <= max_de; de++) {
    sum += transitions(energy, de);
  }
  return sum;
}

// elements of the actual normalized transition matrix:
//   the probability of moving from a given initial energy into a specific final energy
double network_simulation::transition_matrix(const int final_energy,
                                             const int initial_energy) const {
  const int energy_change = final_energy - initial_energy;
  if (abs(energy_change) > max_de) return 0;

  const int normalization = transitions_from(initial_energy);
  if (normalization == 0) return 0;

  return (double(transitions(initial_energy, energy_change))
          / transitions_from(initial_energy));
}

// ---------------------------------------------------------------------------------------
// Methods used in simulation
// ---------------------------------------------------------------------------------------

// initialize all histograms:
//   energy histogram, visit log, sample histogram,
//   state histograms, distance histograms, energy transitions
void network_simulation::initialize_histograms() {
  energy_histogram = vector<unsigned long>(energy_range, 0);
  visit_log = vector<bool>(energy_range, true);
  sample_histogram = vector<unsigned long>(energy_range, 0);
  state_histograms = vector<vector<unsigned long>>(energy_range);
  distance_histograms = vector<vector<unsigned long>>(energy_range);
  energy_transitions = vector<vector<unsigned long>>(energy_range);
  const int pattern_number = patterns.size();
  for (int ee = 0; ee < energy_range; ee++) {
    state_histograms[ee] = vector<unsigned long>(network.nodes, 0);
    distance_histograms[ee] = vector<unsigned long>(pattern_number, 0);
    energy_transitions[ee] = vector<unsigned long>(2*max_de + 1, 0);
  }
}

// update histograms with an observation
void network_simulation::update_energy_histogram(const int energy) {
  energy_histogram[energy]++;
}

void network_simulation::update_state_histograms(const int energy) {
  for (int ii = 0; ii < network.nodes; ii++) {
    state_histograms[energy][ii] += state[ii];
  }
}

void network_simulation::update_distance_histograms(const vector<boo>& state,
                                                    const int energy) {
  for (int pp = 0, size = patterns.size(); pp < size; pp++) {
    distance_histograms[energy][pp] += state_distance(state,patterns[pp]);
  }
}

void network_simulation::update_sample_histogram(const int new_energy, const int old_energy) {
  if (!visit_log[new_energy]) {
    visit_log[new_energy] = true;
    sample_histogram[new_energy]++;
  }

  if (old_energy == entropy_peak) {
    visit_log[entropy_peak] = false;
    return;
  }
  if (new_energy == entropy_peak) {
    visit_log = vector<bool>(energy_range, false);
    return;
  }

  const bool above_peak_now = (new_energy > entropy_peak);
  const bool above_peak_before = (old_energy > entropy_peak);
  if (above_peak_now == above_peak_before) return;

  if (above_peak_now) {
    // reset visit log at low energies
    for (int ee = 0; ee < entropy_peak; ee++) {
      visit_log[ee] = false;
    }
  } else {
    // reset visit log at high energies
    for (int ee = entropy_peak + 1; ee < energy_range; ee++) {
      visit_log[ee] = false;
    }
  }
}

void network_simulation::add_transition(const int energy, const int energy_change) {
  energy_transitions[energy][energy_change + max_de]++;
}

// expectation value of fractional sample error at a given inverse temperature
// WARNING: assumes that the density of states is up to date
double network_simulation::fractional_sample_error(const double beta_cap) const {
  assert(beta_cap != 0);

  int lowest_energy;
  int highest_energy;
  if (beta_cap > 0) { // low energies
    highest_energy = entropy_peak;
    for (int ee = 0; ee < entropy_peak; ee++) {
      if (sample_histogram[ee] != 0) {
        lowest_energy = ee;
        break;
      }
    }

  } else { // high energies
    lowest_energy = entropy_peak;
    for (int ee = energy_range - 1; ee > entropy_peak; ee--) {
      if (sample_histogram[ee] != 0) {
        highest_energy = ee;
        break;
      }
    }
  }
  const int mean_energy = (highest_energy + lowest_energy) / 2;

  long double error = 1;
  long double normalization = 1;
  for (int ee = lowest_energy; ee < highest_energy; ee++) {
    if (sample_histogram[ee] != 0) {
      const long double boltzmann_factor
        = exp(ln_dos[ee] - ln_dos[mean_energy] - (ee - mean_energy) * beta_cap);
      error += boltzmann_factor/sqrt(sample_histogram[ee]);
      normalization += boltzmann_factor;
    }
  }
  return error/normalization;
}


// compute density of states and appropriate energy weights from the transition matrix
void network_simulation::compute_dos_and_weights_from_transitions(const double beta_cap) {

  ln_dos = vector<double>(energy_range, 0);
  ln_weights = vector<double>(energy_range, 1);

  double max_ln_dos = 0;

  // sweep across energies to construct the density of states
  for (int ee = 1; ee < energy_range; ee++) {

    ln_dos[ee] = ln_dos[ee-1];

    if (energy_histogram[ee] == 0) continue;

    double flux_up_to_this_energy = 0;
    double flux_down_from_this_energy = 0;
    for (int smaller_ee = 0; smaller_ee < ee; smaller_ee++) {
      flux_up_to_this_energy += (exp(ln_dos[smaller_ee] - ln_dos[ee])
                                 * transition_matrix(ee, smaller_ee));
      flux_down_from_this_energy += transition_matrix(smaller_ee, ee);
    }
    if (flux_up_to_this_energy > 0 && flux_down_from_this_energy > 0) {
      ln_dos[ee] += log(flux_up_to_this_energy/flux_down_from_this_energy);
    }

    if (ln_dos[ee] > max_ln_dos) {
      max_ln_dos = ln_dos[ee];
      entropy_peak = ee;
    }

  }

  for (int ee = 0; ee < energy_range; ee++) {
    ln_dos[ee] -= max_ln_dos;
  }

  if (beta_cap > 0) {

    int smallest_seen_energy = 0;
    for (int ee = 0; ee < energy_range; ee++) {
      if (energy_histogram[ee] != 0) {
        smallest_seen_energy = ee;
        break;
      }
    }

    // in the relevant range of observed energies, set weights appropriately
    for (int ee = entropy_peak; ee > smallest_seen_energy; ee--) {
      ln_weights[ee] = -ln_dos[ee];
    }
    // below all observed energies use weights fixed at an inverse temperature beta_cap
    for (int ee = smallest_seen_energy; ee >= 0; ee--) {
      ln_weights[ee] = (-ln_dos[smallest_seen_energy]
                        - abs(smallest_seen_energy - ee) * beta_cap);
    }
    // above the entropy peak, use flat (zero beta) weights
    for (int ee = energy_range - 1; ee > entropy_peak; ee--) {
      ln_weights[ee] = -ln_dos[entropy_peak];
    }

  } else { // if beta_cap < 0

    int largest_seen_energy = energy_range;
    for (int ee = energy_range - 1; ee >= 0; ee++) {
      if (energy_histogram[ee] != 0) {
        largest_seen_energy = ee;
        break;
      }
    }

    // in the relevant range of observed energies, set weights appropriately
    for (int ee = entropy_peak; ee < largest_seen_energy; ee++) {
      ln_weights[ee] = -ln_dos[ee];
    }
    // above all observed energies use weights fixed at an inverse temperature beta_cap
    for (int ee = largest_seen_energy; ee < energy_range; ee++) {
      ln_weights[ee] = (-ln_dos[largest_seen_energy]
                        - abs(largest_seen_energy - ee) * beta_cap);
    }
    // below the entropy peak, use flat (zero beta) weights
    for (int ee = 0; ee < entropy_peak; ee++) {
      ln_weights[ee] = -ln_dos[entropy_peak];
    }

  }

}

// compute density of states from the energy histogram
void network_simulation::compute_dos_from_energy_histogram() {
  ln_dos = vector<double>(energy_range);

  double max_ln_dos = 0;
  for (int ee = 0; ee < energy_range; ee++) {
    ln_dos[ee] = log(energy_histogram[ee]) - ln_weights[ee];
    max_ln_dos = max(ln_dos[ee], max_ln_dos);
  }
  for (int ee = 0; ee < energy_range; ee++) {
    ln_dos[ee] -= max_ln_dos;
  }
};

// ---------------------------------------------------------------------------------------
// Printing methods
// ---------------------------------------------------------------------------------------

// print patterns defining the simulated network
void network_simulation::print_patterns() const {
  const int energy_width = log10(network.max_energy) + 2;
  const int pattern_number = patterns.size();

  // make list of energies
  vector<int> energies(pattern_number);
  for (int pp = 0; pp < pattern_number; pp++) {
    energies[pp] = energy(patterns[pp]);
  }
  vector<int> sorted_energies = energies;
  sort(sorted_energies.begin(), sorted_energies.end());
  vector<bool> printed(pattern_number,false);

  // print patterns in order of decreasing energy
  cout << "(energy) pattern" << endl;
  for (int ss = pattern_number - 1; ss >= 0; ss--) {
    cout << "(" << setw(energy_width)
         << sorted_energies[ss] * network.energy_scale - network.max_energy << ")";
    for (int pp = 0; pp < pattern_number; pp++) {
      if (energies[pp] == sorted_energies[ss] && !printed[pp]) {
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

// print energy histogram, sample histogram, and density of states
void network_simulation::print_energy_data() const {
  cout << "energy observations samples log10_dos" << endl;
  const int energy_width = log10(network.max_energy) + 2;
  const int energy_hist_width = log10(2*energy_histogram[entropy_peak] + 1) + 1;
  const int sample_width = log10(sample_histogram[entropy_peak]) + 1;
  const int dos_dec = 6;
  for (int ee = energy_range - 1; ee >= 0; ee--) {
    const int observations = energy_histogram[ee];
    if (observations != 0) {
      cout << fixed
           << setw(energy_width)
           << ee * network.energy_scale - network.max_energy << " "
           << setw(energy_hist_width) << observations << " "
           << setw(sample_width) << sample_histogram[ee] << " "
           << setw(dos_dec + 3) << setprecision(dos_dec)
           << log10(exp(1)) * ln_dos[ee] << endl;
    }
  }
}

// print expectation value of each spin spin at each energy
void network_simulation::print_expected_states() const {
  cout << "energy <s_1>, <s_2>, ..., <s_n>" << endl;
  const int energy_width = log10(network.max_energy) + 2;
  const int state_dec = 6;
  cout << setprecision(state_dec);
  for (int ee = energy_range - 1; ee >= 0; ee--) {
    const int observations = energy_histogram[ee];
    if (observations > 0) {
      cout << setw(energy_width) << ee * network.energy_scale - network.max_energy;
      for (int ii = 0; ii < network.nodes; ii++) {
        cout << " " << setw(state_dec + 3)
             << 2*double(state_histograms[ee][ii])/observations - 1;
      }
      cout << endl;
    }
  }
}

// print expectation value of distances from each pattern at each energy
void network_simulation::print_distances() const {
  cout << "energy <d_1>, <d_2>, ..., <d_p>" << endl;
  const int energy_width = log10(network.max_energy) + 2;
  const int pattern_number = patterns.size();
  const int distance_dec = 6;
  cout << setprecision(distance_dec);
  for (int ee = energy_range - 1; ee >= 0; ee--) {
    const int observations = energy_histogram[ee];
    if (observations > 0) {
      cout << setw(energy_width) << ee * network.energy_scale - network.max_energy;
      for (int ii = 0; ii < pattern_number; ii++) {
        cout << " " << setw(distance_dec + 2)
             << double(distance_histograms[ee][ii]) / observations;
      }
      cout << endl;
    }
  }
}
