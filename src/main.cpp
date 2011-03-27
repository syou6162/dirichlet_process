#include <set>
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>
#include "DirichletProcess.hpp"

int main(int argc, char *argv[]) {
  DirichletProcess dp(0.01, 0.001);

  std::string filename = "tmp.txt";
  std::ifstream ifs(filename.c_str());

  std::string line;
  while (getline(ifs, line)) {
	std::vector<std::string> v;
	boost::algorithm::split(v, line, boost::is_any_of(" .,?!\""));
	BOOST_FOREACH(std::string s, v) {
	  dp.add(s);
	}
  }

  // for (int i = 0; i < 1; i++) {
  // 	dp.add("0"); dp.add("0"); dp.add("0"); dp.add("0");
  // 	dp.add("1"); dp.add("1"); dp.add("1"); dp.add("1"); dp.add("1"); dp.add("1"); dp.add("1"); dp.add("1");
  // 	dp.add("2"); dp.add("2"); dp.add("2");
  // 	dp.add("3"); dp.add("3"); dp.add("3"); dp.add("3"); dp.add("3"); dp.add("3");
  // 	dp.add("4"); dp.add("5"); dp.add("6"); dp.add("7"); dp.add("8"); dp.add("9");
  // }
  
  dp.initialize();
  for (int i = 0; i < 3000; i++) {
	dp.gibbs_sampling();
  }
  dp.print_result();
  return 0;
};
