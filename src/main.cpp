#include <set>
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include "DirichletProcess.hpp"

int main(int argc, char *argv[]) {
  DirichletProcess dp(0.001, 0.3);

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
  for (int i = 0; i < 3000; i++) {
	dp.gibbs_sampling();
	std::cout << i << "\t" 
			  << dp.get_num_of_clusters() << "\t" 
			  << dp.log_likelihood() << "\t" 
			  << dp.perplexity(test) << std::endl;
  }

  dp.print_result();
  return 0;
};
