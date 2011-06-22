#include "util.hpp"
#include <cmath>

namespace math {
  namespace {
	boost::mt19937 gen_(static_cast<unsigned long>(time(0)));
	boost::uniform_real<> dst_(0, 1);
	boost::variate_generator<boost::mt19937, boost::uniform_real<> > rand_(gen_, dst_);
  }

  boost::uniform_real<>::result_type random() { 
	return rand_();
  };
};

double logsumexp(double x, double y, bool flg) {
  if (flg) return y; // init mode
  if (x == y) return x + 0.69314718055;  // log(2)
  double vmin = std::min(x, y);

  double vmax = std::max(x, y);
  if (vmax > vmin + 50) {
    return vmax;
  } else {
    return vmax + log (exp (vmin - vmax) + 1.0);
  }
};
