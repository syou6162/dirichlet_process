#ifndef NEXT_NUMBER_GENERATOR_HPP
#define NEXT_NUMBER_GENERATOR_HPP

#include <vector>
#include <functional>
#include <algorithm>

class NextNumberGenerator {
public:
  NextNumberGenerator(const std::vector<int>& v) {
	v_ = v;
	std::sort(v_.begin(), v_.end(), std::less<int>());
  };
  int next() {
	if (v_.size() == 0) return 0;
	int max = 0; int prev = 0;
	std::vector<int>::const_iterator it = v_.begin();
	if (v_.front() == 1) return 0;
	while (++it != v_.end()) {
	  int i = *it;
	  if (i - prev != 1) return prev + 1;
	  max = std::max(max, i);
	  prev = i;
	}
	return max + 1;
  };
  std::vector<int> v_;
};

#endif
