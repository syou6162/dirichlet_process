#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <vector>
#include "util.hpp"

class Sampler {
public:
  Sampler(const std::vector<double>& posts, const double psum) {
	posts_ = posts; psum_ = psum;
  };
  int sample() {
	const double r = math::random();
	unsigned int k = 0;
	posts_[0] = exp(posts_[0] - psum_);
	if (r > posts_[0]) {
	  for (unsigned int k = 1; k < posts_.size(); k++) {
		posts_[k] = exp(posts_[k] - psum_) + posts_[k - 1];
		if (r <= posts_[k]) {
		  return k;
		}
	  }
	}
	return k;
  };
  double psum_;
  std::vector<double> posts_;
};

#endif
