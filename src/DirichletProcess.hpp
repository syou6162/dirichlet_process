#ifndef DIRICHLET_PROCESS_HPP
#define DIRICHLET_PROCESS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <limits>
#include <fstream>
#include <queue>
#include <sstream>
#include <tr1/unordered_map>
#include <boost/random.hpp>
#include <boost/foreach.hpp>

template<class D, class G = boost::mt19937>
class Rand {
  G gen_;
  D dst_;
  boost::variate_generator<G, D> rand_;
public:
  Rand() : gen_(static_cast<unsigned long>(time(NULL) + getpid())), rand_(gen_, dst_) {}
  template<typename T1>
  Rand(T1 a1) : gen_(static_cast<unsigned long>(time(NULL) + getpid())), dst_(a1), rand_(gen_, dst_) {}
  template<typename T1, typename T2>
  Rand(T1 a1, T2 a2) : gen_(static_cast<unsigned long>(time(NULL) + getpid())), dst_(a1, a2), rand_(gen_, dst_) {}

  typename D::result_type operator()() { return rand_(); }
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

typedef std::map<std::string, double> container;
typedef std::pair<std::string, double> value_type;

struct second_order {
  bool operator()(const value_type& x, const value_type& y) const {
    return x.second > y.second;
  }
};

class DirichletProcess {
public:
  DirichletProcess(double beta, double gamma) {
	beta_ = beta; gamma_ = gamma; N_ = 0; K_ = -1;
  };

  ~DirichletProcess() {
  };

  void add(const std::string& word) {
	N_++;
	w_.push_back(getID(word, true));
	z_.push_back(-1); // initialize
  };

  void initialize() {
	increase(0, 0);
	for (int i = 1; i < N_; i++) {
	  // number of existing tables + new table
	  std::vector<double> posts(get_num_of_clusters() + 1);
	  double psum = make_cumsum_vector(posts, i);
	  int next_z = sample_from_multinomial(posts, psum);
	  increase(i, next_z);
	}
	std::cerr << "Number of clusters is " << get_num_of_clusters() << std::endl;
	std::cerr << "Perplexity: " << perplexity(w_) << std::endl;
  }

  void increase(int index, int cluster) {
	z_[index] = cluster;
	if (Nk_.find(cluster) == Nk_.end()) {
	  K_++;
	}
	Nk_[cluster]++; Nkw_[cluster][w_[index]]++;
  };

  void decrease(int index, int cluster) {
	if (--(Nkw_[cluster][w_[index]]) == 0) {
	  Nkw_[cluster].erase(w_[index]);
	}
	if (--(Nk_[cluster]) == 0) {
	  Nk_.erase(cluster);
	  Nkw_.erase(cluster);
	  shift_cluster_id(cluster);
	  K_--;
	}
  };

  void shift_cluster_id(int cluster) {
	for (int i = 0; i < N_; i++) {
	  if (z_[i] >= cluster) z_[i]--;
	}
	int max_cluster_id = get_num_of_clusters() - 1;
	// shift left hand side
	for (int k = cluster; k < max_cluster_id; k++) {
	  Nk_[k] = Nk_[k + 1];
	  Nkw_[k] = Nkw_[k + 1];
	}
	// remove tail item
	Nk_.erase(max_cluster_id);
	Nkw_.erase(max_cluster_id);
  };

  int sample_from_multinomial(std::vector<double>& posts, double psum) {
	double r = randomValue_();
	int next_z = 0;
	posts[0] = exp(posts[0] - psum);
	if (r > posts[0]) {
	  //	  for (int k = 1; k < get_num_of_clusters() + 1; k++) {
	  for (unsigned int k = 1; k < posts.size(); k++) {
		posts[k] = exp(posts[k] - psum) + posts[k - 1];
		if (r <= posts[k]) {
		  next_z = k;
		  break;
		}
	  }
	}
	// if (next_z == get_num_of_clusters()) {
	//   std::cout << "created at new table(" << next_z << ")! " << std::endl;
	// }
	return next_z;
  };

  double make_cumsum_vector(std::vector<double>& posts, int index) {
	int word = w_[index];
	double psum = 0.0;
	// existing tables (existing clusters)
	for (int k = 0; k < get_num_of_clusters(); k++) {
	  posts[k] += log(Nk_[k]);
	  posts[k] -= log(N_ - 1 + gamma_);
	  posts[k] += log(Nkw_[k][word] + beta_);
	  posts[k] -= log(Nk_[k] + get_num_of_vocablary() * beta_);
	  psum = logsumexp(psum, posts[k], k == 0);
	}
	// new table (new cluster)
	int new_cluster_id = get_num_of_clusters();
	posts[new_cluster_id] += log(gamma_);
	posts[new_cluster_id] -= log(N_ - 1 + gamma_);
	posts[new_cluster_id] += log(1.0);
	posts[new_cluster_id] -= log(get_num_of_vocablary());
	return logsumexp(psum, posts[new_cluster_id], false);
  };

  void gibbs_sampling() {
	for (int i = 0; i < N_; i++) {
	  decrease(i, z_[i]);
	  std::vector<double> posts(get_num_of_clusters() + 1);
	  double psum = make_cumsum_vector(posts, i);
	  int next_z = sample_from_multinomial(posts, psum);
	  increase(i, next_z);
	}
	std::cerr << "Number of clusters is " << get_num_of_clusters() << std::endl;
	std::cerr << "Perplexity: " << perplexity(w_) << std::endl;
  }

  int getID(const std::string& str, const bool train) {
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

  void print_result() {
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

  void print_state() {
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

  int get_num_of_clusters() {
	return K_ + 1;
  };

  int get_num_of_vocablary() {
	return id2word_.size();
  };

  double perplexity(const std::vector<int>& words) {
	double result = 0.0;
	for (std::vector<int>::const_iterator it = words.begin(); it != words.end(); it++) {
	  double sum = 0.0;
	  for (int k = 0; k < get_num_of_clusters(); k++) {
		double tmp 
		  = log(Nkw_[k][*it] + beta_) - log(Nk_[k] + get_num_of_vocablary() * beta_) 
		  + log(Nk_[k]) - log(N_ + gamma_);
		sum = logsumexp(sum, tmp, (k == 0));
	  }
	  double tmp = log(1.0) - log(get_num_of_vocablary()) 
		+ log(gamma_) - log(N_ + gamma_); 
	  sum = logsumexp(sum, tmp, false);
	  result += sum;
	}
	return exp(- result / (double) words.size());
  };


  double log_likelihood() {
	double result = 0.0;
	// log(x | z)
	result += get_num_of_clusters() * lgamma(beta_ * get_num_of_vocablary()) 
	  - get_num_of_clusters() * get_num_of_vocablary() * lgamma(beta_);
	for (int w = 0; w < get_num_of_vocablary(); w++) {
	  for (int k = 0; k < get_num_of_clusters(); k++) {
		result += lgamma(Nkw_[k][w] + beta_) - lgamma(Nk_[k] + beta_ * get_num_of_vocablary());
	  }
	}
	// log(z)
	result += get_num_of_clusters() * log(gamma_);
	for (int k = 0; k < get_num_of_clusters(); k++) {
	  for (int i = 1; i < Nk_[k]; i++) {
		result += log(i);
	  }
	}
	for (int i = 1; i <= N_; i++) {
	  result -= log(gamma_ + i - 1);
	}
	return result;
  };

protected:
  std::map<std::string, int> word2id_;
  std::vector<std::string> id2word_;

  std::vector<int> w_; // word
  std::vector<int> z_; // table id or cluster id 
  Rand<boost::uniform_real<> > randomValue_;

  std::tr1::unordered_map<int, int> Nk_; // Number of customers in table k;
  std::tr1::unordered_map<int, std::map<int, int> > Nkw_; // Number of customers in table k

  int N_; // Number of Data
  int K_; // Number of hidden states

  ////////////////////////////////////////
  // hyperparameter
  ////////////////////////////////////////
  double beta_;
  double gamma_;

  // ToDo
  //  void sampling();
};

#endif
