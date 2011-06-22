#ifndef DIRICHLET_PROCESS_HPP
#define DIRICHLET_PROCESS_HPP

#include <tr1/unordered_map>
#include <boost/random.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/foreach.hpp>
#include "NextNumberGenerator.hpp"
#include "util.hpp"

typedef std::map<std::string, double> container;
typedef std::pair<std::string, double> value_type;

struct second_order {
  bool operator()(const value_type& x, const value_type& y) const {
    return x.second > y.second;
  }
};

class DirichletProcess {
public:
  DirichletProcess(const double beta, const double gamma) : 
	beta_(beta), gamma_(gamma), N_(0), a_(0.1), b_(0.1) {
  };

  void add(const std::string& word) {
	N_++;
	w_.push_back(getID(word, true));
	z_.push_back(-1); // initialize
  };

  int next_k();
  void increase(const int index, const int cluster);
  void decrease(const int index, const int cluster);
  double make_cumsum_vector(std::vector<double>& posts, const int index);
  void initialize();
  void gibbs_sampling(const bool init = false);
  int getID(const std::string& str, const bool train);

  void print_result();
  void print_state();

  int get_num_of_clusters() const {
	return Nk_.size();
  };

  int get_num_of_vocablary() const {
	return id2word_.size();
  };

  int get_current_cluster_id(const int index);

  double perplexity(const std::vector<int>& words);
  double log_likelihood();
  double sample_concentration_parameter(const double x);
  // density of mixture of two gamma p(gamma | x, k) 
  double density_of_mixture_of_gamma(const double gamma, const double x, const int k);

protected:
  std::map<std::string, int> word2id_;
  std::vector<std::string> id2word_;

  std::vector<int> w_; // word
  std::vector<int> z_; // table id or cluster id 

  //  std::set<int, std::greater<int> > cluster_numbers;
  //  Rand<boost::uniform_real<> > randomValue_;

  std::tr1::unordered_map<int, int> Nk_; // Number of customers in table k;
  std::tr1::unordered_map<int, std::map<int, int> > Nkw_; // Number of customers in table k

  int N_; // Number of Data

  ////////////////////////////////////////
  // hyperparameter
  ////////////////////////////////////////
  double beta_;
  double gamma_;

  ////////////////////////////////////////
  // hyperhyperparameter
  ////////////////////////////////////////
  double a_;
  double b_;

  // ToDo
  //  void sampling();
};

#endif
