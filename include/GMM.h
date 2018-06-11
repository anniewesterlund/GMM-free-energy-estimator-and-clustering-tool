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
#include <Eigen/Eigen>

using namespace std;

struct estimator_attributes{

	estimator_attributes(int nGaussianComponents, Eigen::MatrixXd x);

	int nComponents_;
	vector<Eigen::VectorXd> centers_;
	vector<Eigen::MatrixXd> covariances_;
	Eigen::VectorXd amplitudes_;
	Eigen::MatrixXd Gaussian_projections_;
	Eigen::MatrixXd membership_weights_;
	Eigen::MatrixXd eye_stability_;
};

class GMM {
	// Public functions
	public:
	
	GMM(vector<vector<double>> data, double convergence_tol, int max_interations);
	
	double validate_density(estimator_attributes GMM_obj, Eigen::MatrixXd x);
	estimator_attributes train_density(int nGaussianComponents, Eigen::MatrixXd x);
	void fitDensity(int nMinGaussianComponents, int nMaxGaussianComponents);
	
	Eigen::VectorXd getDensity();
	Eigen::VectorXd getDensity(Eigen::VectorXd x);
	vector<Eigen::VectorXd> getCenters();
	vector<Eigen::MatrixXd> getCovariances();
	Eigen::VectorXd getAmplitudes();
	vector<int> getClusterIndices();
	vector<int> getClusterIndicesWithHalo();
	
	// Private functions and variables
	private:	
	void expectation_step(estimator_attributes &GMM_obj);
	void maximization_step(estimator_attributes &GMM_obj, Eigen::MatrixXd x);
	
	void write_to_file(estimator_attributes GMM_obj, Eigen::MatrixXd x);
	
	
	// Global variables
	Eigen::MatrixXd x_;
	Eigen::MatrixXd training_data_;
	Eigen::MatrixXd validation_data_;
	
	vector<int> cluster_indices_;
	
	int max_iter_;
	double convergence_tol_;
	
	
	int nPoints_;
	int nTrainPoints_;
	int nValidationPoints_;
	int nDims_;
	
	GMDensity densityEvaluator_; 
};

