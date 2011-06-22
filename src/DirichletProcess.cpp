#include "util.hpp"
#include "DirichletProcess.hpp"
#include "Sampler.hpp"
#include "NextNumberGenerator.hpp"

int DirichletProcess::next_k() {
  std::vector<int> v;
  for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
	v.push_back(it->first);
  }
  NextNumberGenerator g(v);
  return g.next();
};

void DirichletProcess::increase(const int index, const int cluster) {
  z_[index] = cluster;
  Nk_[cluster]++; Nkw_[cluster][w_[index]]++;
};

void DirichletProcess::decrease(const int index, const int cluster) {
  if (--(Nkw_[cluster][w_[index]]) == 0) {
	Nkw_[cluster].erase(w_[index]);
  }
  if (--(Nk_[cluster]) == 0) {
	Nk_.erase(cluster);
	Nkw_.erase(cluster);
  }
};

double DirichletProcess::make_cumsum_vector(std::vector<double>& posts, const int index) {
  int word = w_[index];
  double psum = 0.0;
  // existing tables (existing clusters)
  bool init_flag = true;
  int i = 0;
  for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
	int k = it->first;
	double tmp = 0.0;
	tmp += log(Nk_[k]);
	tmp -= log(N_ - 1 + gamma_);
	tmp += log(Nkw_[k][word] + beta_);
	tmp -= log(Nk_[k] + get_num_of_vocablary() * beta_);
	posts[i] = tmp;
	psum = logsumexp(psum, tmp, init_flag);
	i++; init_flag = false;
  }
  // new table (new cluster)
  double tmp = 0.0;
  tmp += log(gamma_);
  tmp -= log(N_ - 1 + gamma_);
  tmp += log(1.0);
  tmp -= log(get_num_of_vocablary());
  posts[i] = tmp;
  return logsumexp(psum, tmp, false);
};

void DirichletProcess::initialize() {
  increase(0, 0);
  for (int i = 1; i < N_; i++) {
	// number of existing tables + new table
	std::vector<double> posts(get_num_of_clusters() + 1);
	double psum = make_cumsum_vector(posts, i);
	Sampler sampler(posts, psum);
	int next_z = get_current_cluster_id(sampler.sample());
	increase(i, next_z);
  }
  std::cerr << "Number of clusters is " << get_num_of_clusters() << std::endl;
  std::cerr << "Perplexity: " << perplexity(w_) << std::endl;

  // boost::math::beta_distribution<> gen_beta(gamma_ + 1, (double) N_);
  // double x = boost::math::quantile(gen_beta, randomValue_());
  // gamma_ = sample_concentration_parameter(x);
  // std::cerr << "gamma: " << gamma_ << std::endl;
};

void DirichletProcess::gibbs_sampling() {
  for (int i = 0; i < N_; i++) {
	decrease(i, z_[i]);
	std::vector<double> posts(get_num_of_clusters() + 1);

	double psum = make_cumsum_vector(posts, i);
	Sampler sampler(posts, psum);
	int next_z = get_current_cluster_id(sampler.sample());
	increase(i, next_z);

	//	  std::cout << next_z << ":(" << get_num_of_clusters() << ") ";
	//	  for (std::set<int>::iterator it = cluster_numbers.begin(); it != cluster_numbers.end(); it++) {
	//		std::cout << *it << "(" << Nk_[*it] << "), ";
	//	  }
	//	  std::cout << std::endl;
  }
  std::cerr << "Number of clusters is " << get_num_of_clusters() << std::endl;
  std::cerr << "Perplexity: " << perplexity(w_) << std::endl;

  //  boost::math::beta_distribution<> gen_beta(gamma_ + 1, (double) N_);
  //  double x = boost::math::quantile(gen_beta, randomValue_());
  //  gamma_ = sample_concentration_parameter(x);
  std::cerr << "gamma: " << gamma_ << std::endl;

  //	gamma_ = averaging_concentration_parameter(10);

  int sum = 0;
  for (int i = 0; i < get_num_of_vocablary(); i++) {
	for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
	  int k = it->first;
	  sum += Nkw_[k][i];
	}
  }
  std::cerr << "Sum of Nkw is " << sum << std::endl;

  sum = 0;
  for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
	int k = it->first;
	sum += Nk_[k];
  }
  std::cerr << "N is " << N_ << std::endl;
  std::cerr << "Sum of Nk is " << sum << std::endl;
};

int DirichletProcess::get_current_cluster_id(const int index) {
  int i = 0;
  std::vector<int> v;
  for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
	if (index == i) {
	  return it->first;
	}
	v.push_back(it->first);
	i++;
  }
  NextNumberGenerator g(v);
  return g.next();
};

int DirichletProcess::getID(const std::string& str, const bool train) {
  std::map<std::string, int>::const_iterator it = word2id_.find(str);
  if (it != word2id_.end()) {
	return it->second;
  } else if (train) {
	int newID = static_cast<int>(word2id_.size());
	word2id_[str] = newID;
	id2word_.push_back(str);
	return newID;
  } else {
	return N_;
  }
};

double DirichletProcess::perplexity(const std::vector<int>& words) {
  double result = 0.0;
  for (std::vector<int>::const_iterator word_it = words.begin(); word_it != words.end(); word_it++) {
	int w = *word_it;
	double sum = 0.0;
	bool init_flag = true;
	for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
	  int k = it->first;
	  double tmp 
		= log(Nkw_[k][w] + beta_) - log(Nk_[k] + get_num_of_vocablary() * beta_) 
		+ log(Nk_[k]) - log(N_ + gamma_);
	  sum = logsumexp(sum, tmp, init_flag);
	  init_flag = false;
	}
	double tmp = log(1.0) - log(get_num_of_vocablary()) 
	  + log(gamma_) - log(N_ + gamma_); 
	sum = logsumexp(sum, tmp, false);
	result += sum;
  }
  return exp(- result / (double) words.size());
};

double DirichletProcess::log_likelihood() {
  double result = 0.0;
  // log(x | z)

  result += get_num_of_clusters() * lgamma(beta_ * get_num_of_vocablary());
  result -= get_num_of_clusters() * get_num_of_vocablary() * lgamma(beta_);
  for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
  	int k = it->first;
  	result -= lgamma(Nk_[k] + beta_ * get_num_of_vocablary());
  	for (int w = 0; w < get_num_of_vocablary(); w++) {
  	  result += lgamma(Nkw_[k][w] + beta_);
  	}
  }

  // log(z)
  result += get_num_of_clusters() * log(gamma_);
  for (std::tr1::unordered_map<int, int>::iterator it = Nk_.begin(); it != Nk_.end(); it++) {
	int k = it->first;
	for (int i = 1; i < Nk_[k]; i++) {
	  result += log(i);
	}
  }

  // below is constant
  for (int i = 1; i <= N_; i++) {
	result -= log(gamma_ + i - 1);
  }
  return result;
};

double DirichletProcess::sample_concentration_parameter(const double x) {
  double result = 0.0;
  double threshold = (double) (a_ + get_num_of_clusters() - 1) 
	/ (a_ + get_num_of_clusters() - 1 + N_ * (b_ - log(x)));
  std::cerr << threshold << std::endl;
  //  if (threshold > randomValue_()) { // seleceted pi_x part
  if (threshold > math::random()) { // seleceted pi_x part
	//	Rand<boost::gamma_distribution<> > gen_gamma(a_ + get_num_of_clusters());
	//	result = gen_gamma();
  } else { // seleceted (1 - pi_x) part
	//	Rand<boost::gamma_distribution<> > gen_gamma(a_ + get_num_of_clusters() - 1);
	//	result = gen_gamma();
  }
  double scale = b_ - log(x);
  std::cerr << "scale: " << scale << std::endl;
  return result / scale;
};
  
// sample from p(k | gamma, N)
// double sample_number_of_clusters(const double gamma) {
// 	std::vector<double> posts(N_);	
// 	double psum = 0.0;
// 	for (int k = 0; k < N_; k++) {
// 	  posts[k] += k * log(gamma);
// 	  posts[k] += lgamma(gamma) - lgamma(gamma + N_);
// 	  psum = logsumexp(psum, posts[k], k == 0);
// 	}
// 	int next_k = sample_from_multinomial(posts, psum) + 1;
// 	return next_k;
// };

// density of mixture of two gamma p(gamma | x, k) 
double DirichletProcess::density_of_mixture_of_gamma(const double gamma, const double x, const int k) {
  double result = 0.0;
  double pi_x = (double) (a_ + k - 1.0) / (a_ + k - 1 + N_ * (b_ - log(x)));

  boost::math::gamma_distribution<> gamma_density1(a_ + get_num_of_clusters());
  result += pi_x * boost::math::pdf(gamma_density1, gamma) 
	* std::pow(b_ - log(x), a_) * exp(- (b_ - log(x))* x);

  boost::math::gamma_distribution<> gamma_density2(a_ + get_num_of_clusters() - 1.0);
  result += (1.0 - pi_x) * boost::math::pdf(gamma_density2, gamma) 
	* std::pow(b_ - log(x), a_) * exp(- (b_ - log(x))* x);
  return result;
};

// double averaging_concentration_parameter(const int N) {
// 	double result = 0.0;
// 	for (int s = 0; s < N; s++) {
// 	  boost::math::beta_distribution<> gen_beta(gamma_ + 1, (double) N_);
// 	  double x = boost::math::quantile(gen_beta, randomValue_());
// 	  double gamma = sample_concentration_parameter(x);
// 	  int k = sample_number_of_clusters(gamma);
// 	  std::cerr << "k: " << k << std::endl;
// 	  double tmp = density_of_mixture_of_gamma(gamma, x, k);
// 	  result += tmp;
// 	}
// 	return (double) result / N;
// }

void DirichletProcess::print_result() {
  for (int k = 0; k < get_num_of_clusters(); k++) {
	std::priority_queue<value_type, std::vector<value_type>, second_order> q;
	for (int i = 0; i < get_num_of_vocablary(); i++) {
	  q.push(std::make_pair(id2word_[i], 
							(double) (Nkw_[k][i] + beta_) / (Nk_[k] + beta_ * get_num_of_vocablary())));
	}

	std::cout << "Result of Cluster " << k << ": (" << q.size() << ")" << std::endl;
	int i = 0;
	while (!q.empty() && i < 100) {
	  value_type item = q.top();
	  std::cout << item.first << " ";
	  q.pop();
	  i++;
	}
	std::cout << std::endl;
  }
  std::cout << "V: " << get_num_of_vocablary() << std::endl;
};

void DirichletProcess::print_state() {
  for (std::vector<int>::iterator it = z_.begin(); it != z_.end(); it++) {
	std::cout << *it << " ";
  }
  std::cout << std::endl;
  std::cout << "Number of instances in each cluster: ";
  for (int k = 0; k < get_num_of_clusters(); k++) {
	std::cout << Nk_[k] << " ";
  }
  std::cout << std::endl;
};
