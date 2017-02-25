#include <iostream> // for standard output
#include <iomanip> // for io manipulation (e.g. setw)
#include <random> // for randomness

#include "methods.h"

using namespace std;

// greatest common divisor
int gcd(const int a, const int b) {
  if (b == 0) return a;
  else return gcd(b, a % b);
}

// generate random state
vector<bool> random_state(const int nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator) {
  vector<bool> state(nodes);
  for (int ii = 0; ii < nodes; ii++) {
    state.at(ii) = (rnd(generator) < 0.5);
  }
  return state;
}

// make a random change to a given state using a random number on [0,1)
vector<bool> random_change(const vector<bool>& state, const double random) {
  const int node = floor(random*state.size());
  vector<bool> new_state = state;
  new_state.at(node) = !new_state.at(node);
  return new_state;
}

// hopfield network constructor
hopfield_network::hopfield_network(const vector<vector<bool>>& patterns) {
  nodes = patterns.at(0).size();

  // generate interaction matrix from patterns
  // note: these couplings are a factor of [nodes] greater than the regular definition
  couplings = vector<vector<int>>(nodes);
  max_energy = 0;
  for (int ii = 0; ii < nodes; ii++) {
    couplings.at(ii) = vector<int>(nodes, 0);
    for (int jj = 0; jj < nodes; jj++) {
      if (jj == ii) continue;
      for (int pp = 0; pp < int(patterns.size()); pp++) {
        const int coupling = (2*patterns.at(pp).at(ii)-1)*(2*patterns.at(pp).at(jj)-1);
        couplings.at(ii).at(jj) += coupling;
      }
      max_energy += abs(couplings.at(ii).at(jj));
    }
  }

  max_energy_change = 0;
  energy_scale = max_energy;
  for (int ii = 0; ii < nodes; ii++) {
    int node_energy = 0;
    for (int jj = 0; jj < nodes; jj++) {
      node_energy += abs(couplings.at(ii).at(jj));
    }
    max_energy_change = max(2*node_energy, max_energy_change);
    energy_scale = gcd(node_energy, energy_scale);
  }

  max_energy /= energy_scale;
  max_energy_change /= energy_scale;
  energy_range = 2*max_energy + 1;
};

// energy of the network in a given state
// note: this energy is shifted up by the maximum energy, and is an additional
//       factor of [nodes/energy_scale] greater than the regular definition
int hopfield_network::energy(const vector<bool>& state) const {
  int sum = 0;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = ii+1; jj < nodes; jj++) {
      sum += couplings.at(ii).at(jj) * (2*state.at(ii)-1) * (2*state.at(jj)-1);
    }
  }
  return -sum/energy_scale + max_energy;
}


void hopfield_network::print_couplings() const {
  int largest_coupling = 0;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = 0; jj < nodes; jj++) {
      largest_coupling = max(abs(couplings.at(ii).at(jj)), largest_coupling);
    }
  }

  const int width = log10(largest_coupling) + 2;
  for (int ii = 0; ii < nodes; ii++) {
    for (int jj = 0; jj < nodes; jj++) {
      cout << setw(width) << couplings.at(ii).at(jj) << " ";
    }
    cout << endl;
  }
}

// network simulation constructor
network_simulation::network_simulation(const vector<vector<bool>>& patterns,
                                       const vector<bool>& initial_state) :
  patterns(patterns),
  network(hopfield_network(patterns))
{
  state = initial_state;
  reset_histograms();

  energy_transitions = vector<vector<long int>>(network.energy_range);
  for (int ee = 0; ee < network.energy_range; ee++) {
    energy_transitions.at(ee) = vector<long int>(2*network.max_energy_change + 1, 0);
  }

  reset_visit_log();
  samples = vector<int>(network.energy_range, 0);
  ln_weights = vector<double>(network.energy_range, 1);
  ln_dos = vector<double>(network.energy_range, 0);
};

// ---------------------------------------------------------------------------------------
// Access methods for histograms and matrices
// ---------------------------------------------------------------------------------------

// number of transitions from a given energy with a specified energy change
int network_simulation::transitions(const int energy, const int energy_change) const {
  return energy_transitions.at(energy).at(energy_change + network.max_energy_change);
}

// number of transitions from a given energy to any other energy
int network_simulation::transitions_from(const int energy) const {
  int sum = 0;
  for (int de = -network.max_energy_change; de <= network.max_energy_change; de++) {
    sum += transitions(energy, de);
  }
  return sum;
}

// actual transition matrix
double network_simulation::transition_matrix(const int final_energy,
                                             const int initial_energy) const {
  const int energy_change = final_energy - initial_energy;
  if (abs(energy_change) > network.max_energy_change) return 0;

  const int normalization = transitions_from(initial_energy);
  if (normalization == 0) return 0;

  return (double(transitions(initial_energy, energy_change))
          / transitions_from(initial_energy));
}

// ---------------------------------------------------------------------------------------
// Methods used in simulation
// ---------------------------------------------------------------------------------------

// reset all histograms and the visit log of visited energies
void network_simulation::reset_histograms() {
  energy_histogram = vector<long int>(network.energy_range);
  state_histogram = vector<vector<long int>>(network.nodes);
  for (int ii = 0; ii < network.nodes; ii++) {
    state_histogram.at(ii) = vector<long int>(network.energy_range);
  }
}
void network_simulation::reset_visit_log() {
  visited = vector<bool>(network.energy_range, false);
}

// update histograms with an observation of the current state
void network_simulation::update_histograms() {
  const int ee = energy(state);
  energy_histogram.at(ee)++;
  for (int ii = 0; ii < network.nodes; ii++) {
    state_histogram.at(ii).at(ee) += state.at(ii);
  }
}

// update sample count
void network_simulation::update_samples(const int new_energy, const int old_energy) {
  // we only need to add to the sample count if the new energy is negative
  if (new_energy < network.max_energy) {
    // if we have not visited the new energy yet, now we have; add to the sample count
    if (!visited.at(new_energy)) {
      visited.at(new_energy) = true;
      samples.at(new_energy)++;
    }
  } else if (old_energy < network.max_energy) {
    // if the new energy is >= 0 and the old energy was < 0, reset the visit log
    reset_visit_log();
  }
}

// expectation value of fractional sample error at a given temperature
// WARNING: assumes that the density of states is up to date
double network_simulation::fractional_sample_error(const double temp) const {
  double error = 0;
  double normalization = 0;
  for (int ee = 0; ee < network.energy_range; ee++) {
    if (samples.at(ee) != 0) {
      const double boltzmann_factor = exp(ln_dos.at(ee) - ee/temp);
      error += boltzmann_factor/sqrt(samples.at(ee));
      normalization += boltzmann_factor;
    }
  }
  return error/normalization;
}

// add to transition matrix
void network_simulation::add_transition(const int energy, const int energy_change) {
  energy_transitions.at(energy).at(energy_change + network.max_energy_change)++;
}

// compute density of states and weight array from transition matrix
void network_simulation::compute_dos_and_weights_from_transitions(const double min_temp) {

  ln_dos = vector<double>(network.energy_range, 0);
  ln_weights = vector<double>(network.energy_range, 1);

  double max_ln_dos = 0;
  int max_ln_dos_energy = 0;

  // sweep across energies to construct the density of states
  for (int ee = 1; ee < network.energy_range; ee++) {

    ln_dos.at(ee) = ln_dos.at(ee - 1);

    if (energy_histogram.at(ee) == 0) continue;

    double flux_up_to_this_energy = 0;
    double flux_down_from_this_energy = 0;
    for (int smaller_ee = 0; smaller_ee < ee; smaller_ee++) {
      flux_up_to_this_energy += (exp(ln_dos.at(smaller_ee) - ln_dos.at(ee))
                         * transition_matrix(ee, smaller_ee));
      flux_down_from_this_energy += transition_matrix(smaller_ee, ee);
    }
    if (flux_up_to_this_energy > 0 && flux_down_from_this_energy > 0) {
      ln_dos.at(ee) += log(flux_up_to_this_energy/flux_down_from_this_energy);
    }

    if (ln_dos.at(ee) > max_ln_dos) {
      max_ln_dos = ln_dos.at(ee);
      max_ln_dos_energy = ee;
    }

  }

  int smallest_seen_energy = 0;
  for (int ee = 0; ee < network.energy_range; ee++) {
    if (energy_histogram.at(ee) != 0) {
      smallest_seen_energy = ee;
      break;
    }
  }

  // above the peak density of states, use flat (infinite temperature) weights
  for (int ee = network.energy_range - 1; ee > max_ln_dos_energy; ee--) {
    ln_weights.at(ee) = -ln_dos.at(max_ln_dos_energy);
  }
  // in the range of observed energies, set weights appropriately
  for (int ee = max_ln_dos_energy; ee > smallest_seen_energy; ee--) {
    ln_weights.at(ee) = -ln_dos.at(ee);
  }
  // below all observed energies use weights fixed at the minimum temperature
  for (int ee = smallest_seen_energy; ee >= 0; ee--) {
    ln_weights.at(ee) = (-ln_dos.at(smallest_seen_energy)
                         - abs(smallest_seen_energy - ee) / min_temp);
  }

}

// probability to accept a move into a new state
double network_simulation::acceptance_probability(const vector<bool>& new_state) const {
  return exp(ln_weights.at(energy(new_state)) - ln_weights.at(energy()));
}

// compute density of states from the energy histogram
void network_simulation::compute_dos() {
  ln_dos = vector<double>(network.energy_range);

  double max_ln_dos = 0;
  for (int ee = 0; ee < network.energy_range; ee++) {
    ln_dos.at(ee) = log(energy_histogram.at(ee)) - ln_weights.at(ee);
    max_ln_dos = max(ln_dos.at(ee), max_ln_dos);
  }
  for (int ee = 0; ee < network.energy_range; ee++) {
    ln_dos.at(ee) -= max_ln_dos;
  }
};

// ---------------------------------------------------------------------------------------
// Printing methods
// ---------------------------------------------------------------------------------------

// print simulation patterns
void network_simulation::print_patterns() const {
  for (int ii = 0; ii < int(patterns.size()); ii++) {
    for (int jj = 0; jj < network.nodes; jj++) {
      cout << patterns.at(ii).at(jj) << " ";
    }
    cout << endl;
  }
}

// print a given network state
void network_simulation::print_state(const vector<bool>& state) const {
  for (int ii = 0; ii < int(state.size()); ii++) {
    cout << state.at(ii) << " ";
  }
  cout << endl;
}
