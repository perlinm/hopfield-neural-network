#define EIGEN_USE_MKL_ALL

#include <iostream> // for standard output
#include <fstream> // for file input
#include <iomanip> // some nice printing functions
#include <random> // for randomness

#include <boost/algorithm/string.hpp> // string manipulation library
#include <boost/filesystem.hpp> // filesystem path manipulation library
#include <boost/program_options.hpp> // options parsing library

#include <eigen3/Eigen/Dense> // linear algebra library

using namespace std;
using namespace Eigen;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(const int arg_num, const char *arg_vec[]) {

  // -------------------------------------------------------------------------------------
  // Set input options
  // -------------------------------------------------------------------------------------

  const uint help_text_length = 85;

  unsigned long long int seed;

  po::options_description general("General options", help_text_length);
  general.add_options()
    ("help,h", "produce help message")
    ("seed", po::value<unsigned long long int>(&seed)->default_value(0),
     "seed for random number generator")
    ;

  po::options_description all("Allowed options");
  all.add(general);

  // collect inputs
  po::variables_map inputs;
  po::store(parse_command_line(arg_num, arg_vec, all), inputs);
  po::notify(inputs);


  // if requested, print help text
  if (inputs.count("help")) {
    cout << all;
    return 0;
  }

  uniform_real_distribution<double> rnd(0.0,1.0); // uniform distribution on [0,1)
  mt19937_64 generator(seed); // use and seed the 64-bit Mersenne Twister 19937 generator


}
