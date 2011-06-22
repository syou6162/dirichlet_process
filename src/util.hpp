#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <fstream>
#include <queue>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>

#include <boost/random.hpp>

namespace math {
  boost::uniform_real<>::result_type random();
};

double logsumexp(double x, double y, bool flg);

#endif
