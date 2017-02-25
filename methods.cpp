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
vector<bool> random_state(const uint nodes, uniform_real_distribution<double>& rnd,
                          mt19937_64& generator) {
  vector<bool> state(nodes);
  for (uint ii = 0; ii < nodes; ii++) {
    state.at(ii) = (rnd(generator) < 0.5);
  }
  return state;
}

// make a random change to a given state using a random number on [0,1)
vector<bool> random_change(const vector<bool>& state, const double random) {
  const uint node = floor(random*state.size());
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
  for (uint ii = 0; ii < nodes; ii++) {
    couplings.at(ii) = vector<int>(nodes, 0);
    for (uint jj = 0; jj < nodes; jj++) {
      for (uint pp = 0; pp < patterns.size(); pp++) {
        const int coupling = (2*patterns.at(pp).at(ii)-1)*(2*patterns.at(pp).at(jj)-1);
        couplings.at(ii).at(jj) += coupling;
        max_energy += abs(coupling);
      }
    }
    max_energy -= couplings.at(ii).at(ii);
    couplings.at(ii).at(ii) = 0;
  }

  max_energy_change = 0;
  energy_scale = max_energy;
  for (uint ii = 0; ii < nodes; ii++) {
    uint node_energy = 0;
    for (uint jj = 0; jj < nodes; jj++) {
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
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = ii+1; jj < nodes; jj++) {
      sum += couplings.at(ii).at(jj) * (2*state.at(ii)-1) * (2*state.at(jj)-1);
    }
  }
  return -sum/int(energy_scale) + max_energy;
}


void hopfield_network::print_couplings() const {
  int largest_coupling = 0;
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
      largest_coupling = max(abs(couplings.at(ii).at(jj)), largest_coupling);
    }
  }

  const uint width = log10(largest_coupling) + 2;
  for (uint ii = 0; ii < nodes; ii++) {
    for (uint jj = 0; jj < nodes; jj++) {
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
  initialize_histograms();

  energy_transitions = vector<vector<int>>(network.energy_range);
  for (uint ee = 0; ee < network.energy_range; ee++) {
    energy_transitions.at(ee) = vector<int>(2*network.max_energy_change + 1, 0);
  }

  reset_visit_log();
  samples = vector<uint>(network.max_energy, 0);
  ln_weights = vector<double>(network.energy_range, 1);
};


// ---------------------------------------------------------------------------------------
// Methods used in simulation
// ---------------------------------------------------------------------------------------

// initialize all histograms with zeros
void network_simulation::initialize_histograms() {
  energy_histogram = vector<uint>(network.energy_range);
  state_histogram = vector<vector<uint>>(network.nodes);
  for (uint ii = 0; ii < network.nodes; ii++) {
    state_histogram.at(ii) = vector<uint>(network.energy_range);
  }
}

// update histograms with an observation of the current state
void network_simulation::update_histograms() {
  const uint ee = energy(state);
  energy_histogram.at(ee)++;
  for (uint ii = 0; ii < network.nodes; ii++) {
    state_histogram.at(ii).at(ee) += state.at(ii);
  }
}

// update sample count
void network_simulation::update_samples(const int new_energy, const int old_energy) {
  // we only need to add to the sample count if the new energy is negative
  if (new_energy < 0) {
    // if we have not visited the new energy yet, now we have; add to the sample count
    if (!visited.at(-new_energy)) {
      visited.at(-new_energy) = true;
      samples.at(-new_energy)++;
    }
  } else if (old_energy < 0) {
    // if the new energy is >= 0 and the old energy was < 0, reset the visit log
    reset_visit_log();
  }
}

// reset tally of energies visited since the last observation of states with energy >= 0
void network_simulation::reset_visit_log() {
  visited = vector<bool>(network.max_energy, false);
}

// add to transition matrix
void network_simulation::add_transition(const int energy, const int energy_change) {
  energy_transitions.at(energy).at(energy_change + network.max_energy_change)++;
}

// compute density of states and weight array from transition matrix
void network_simulation::compute_weights_from_transitions() {

  vector<double> ln_dos(network.energy_range, 0);

  double max_dos = 0;
  uint max_dos_energy = 0;

  // sweep across energies to construct the density of states
  for (uint ee = 1; ee < network.energy_range; ee++) {

    ln_dos.at(ee) = ln_dos.at(ee - 1);
    if (transitions_from(ee) == 0) continue;

    double up_to_energy = 0;
    double down_from_energy = 0;
    for (uint smaller_ee = 0; smaller_ee < ee; smaller_ee++) {
      up_to_energy += (exp(ln_dos.at(smaller_ee) - ln_dos.at(ee))
                         * transition_matrix(ee, smaller_ee));
      down_from_energy += transition_matrix(smaller_ee, ee);
    }
    if (up_to_energy > 0 && down_from_energy > 0) {
      ln_dos.at(ee) += log(up_to_energy/down_from_energy);
    }

    if (ln_dos.at(ee) > max_dos) {
      max_dos = ln_dos.at(ee);
      max_dos_energy = ee;
    }

  }

  // uint lowest_seen_energy = 0;
  for (uint ee = 0; ee < network.energy_range; ee++) {
    ln_dos.at(ee) -= max_dos;
  }
  for (uint ee = 0; ee < max_dos_energy; ee++) {
    ln_weights.at(ee) = -ln_dos.at(ee);
  }
  for (uint ee = max_dos_energy; ee < network.energy_range; ee++) {
    ln_weights.at(ee) = -ln_dos.at(max_dos_energy);
  }

}

// ---------------------------------------------------------------------------------------
// Access methods for histograms and matrices
// ---------------------------------------------------------------------------------------

// observations of a given energy
uint network_simulation::energy_observations(const int energy) const {
  return energy_histogram.at(energy);
}
// number of transitions from a given energy with a specified energy change
uint network_simulation::transitions(const int energy, const int energy_change) const {
  return energy_transitions.at(energy).at(energy_change + network.max_energy_change);
}

// number of transitions from a given energy to any other energy
uint network_simulation::transitions_from(const int energy) const {
  uint sum = 0;
  for (int de = -int(network.max_energy_change);
       de <= int(network.max_energy_change); de++) {
    sum += transitions(energy, de);
  }
  return sum;
}

// actual transition matrix
double network_simulation::transition_matrix(const int final_energy,
                                             const int initial_energy) const {
  const int energy_change = final_energy - initial_energy;
  if (abs(energy_change) > network.max_energy_change) return 0;

  const uint normalization = transitions_from(initial_energy);
  if (normalization == 0) return 0;

  return (double(transitions(initial_energy, energy_change))
          / transitions_from(initial_energy));
}

// ---------------------------------------------------------------------------------------
// Printing methods
// ---------------------------------------------------------------------------------------

// print simulation patterns
void network_simulation::print_patterns() const {
  for (uint ii = 0; ii < patterns.size(); ii++) {
    for (uint jj = 0; jj < network.nodes; jj++) {
      cout << patterns.at(ii).at(jj) << " ";
    }
    cout << endl;
  }
}

// print a given network state
void network_simulation::print_state(const vector<bool>& state) const {
  for (uint ii = 0; ii < state.size(); ii++) {
    cout << state.at(ii) << " ";
  }
  cout << endl;
}
