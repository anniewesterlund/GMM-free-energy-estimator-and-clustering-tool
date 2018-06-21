/**********************************
*
* 	Free energy estimation using Gaussian mixture model and cross-validation 
*	to determine the number of components.
* 	Annie Westerlund
*
*	2018, KTH
*
**********************************/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <Eigen/Eigen>

struct estimator_attributes{

	estimator_attributes(int nGaussianComponents, Eigen::MatrixXd x);
	
	int nComponents_;
	std::vector<Eigen::VectorXd> centers_;
	std::vector<Eigen::MatrixXd> covariances_;
	Eigen::VectorXd amplitudes_;
	Eigen::MatrixXd Gaussian_projections_;
	Eigen::MatrixXd membership_weights_;
	Eigen::MatrixXd eye_stability_;
};

class GMM {
	// Public functions
	public:
	
	GMM(std::vector<std::vector<double>> data, double convergence_tol, int max_interations, std::string file_label);
	
	double validate_density(estimator_attributes GMM_obj, Eigen::MatrixXd x);
	void fitDensity(int nMinGaussianComponents, int nMaxGaussianComponents);
	
	Eigen::VectorXd getDensity();
	Eigen::VectorXd getDensity(Eigen::VectorXd x);
	std::vector<Eigen::VectorXd> getCenters();
	std::vector<Eigen::MatrixXd> getCovariances();
	Eigen::VectorXd getAmplitudes();
	std::vector<int> getClusterIndicesWithHalo();
	void write_cluster_indices();
	
	
	// Private functions and variables
	private:		
	
	estimator_attributes train_density(int nGaussianComponents, Eigen::MatrixXd x);
	void expectation_step(estimator_attributes &GMM_attributes);
	void maximization_step(estimator_attributes &GMM_attributes, Eigen::MatrixXd x);
	
	void write_to_file(estimator_attributes GMM_obj);
	void extract_clusters(estimator_attributes GMM_attributes);
	
	// Global variables
	Eigen::MatrixXd x_;
	Eigen::MatrixXd training_data_;
	Eigen::MatrixXd validation_data_;
	
	std::vector<int> cluster_indices_;
	
	int max_iter_;
	double convergence_tol_;
	
	std::string file_label_;
	
	int nPoints_;
	int nTrainPoints_;
	int nValidationPoints_;
	int nDims_;
	
	GMDensity densityEvaluator_; 
};

