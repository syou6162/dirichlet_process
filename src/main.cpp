#include <set>
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include "DirichletProcess.hpp"

void print_result(DirichletProcess& dp, const std::vector<int>& test, const int iter) {
  std::cout << iter << "\t" 
			<< dp.get_num_of_clusters() << "\t" 
			<< dp.log_likelihood() << "\t" 
			<< dp.perplexity(test) << std::endl;
}

int main(int argc, char *argv[]) {
  DirichletProcess dp(0.01, 0.001);
  //  DirichletProcess dp(1.1, 350.0);
  std::string filename = "tmp.txt";
  std::ifstream ifs(filename.c_str());

  std::string line;
  std::vector<std::string> train;
  std::vector<int> test;
  while (getline(ifs, line)) {
  	std::vector<std::string> v;
  	boost::algorithm::split(v, line, boost::is_any_of(" .,?!\""));
  	BOOST_FOREACH(std::string s, v) {
	  train.push_back(s);
  	}
  }

  std::random_shuffle(train.begin(), train.end());
  for (int i = 0; i < 10000; i++){
	dp.add(train[i]);
  }
  for (int i = 10000; i < 13000; i++){
	test.push_back(dp.getID(train[i], false));
  }

  dp.initialize();
  print_result(dp, test, 0);
  for (int i = 1; i < 3000; i++) {
	dp.gibbs_sampling();
	print_result(dp, test, i);
	//	std::cerr << "alpha: " << dp.sample_concentration_parameter(0.5) << std::endl;
	//	std::cerr << "alpha: " << dp.averaging_concentration_parameter(10) << std::endl;
  }

  //  dp.print_result();
  return 0;
};
