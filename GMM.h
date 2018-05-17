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
#include "eigen/Eigen/Eigen"

using namespace std;

struct estimator_attributes{
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
	estimator_attributes initialize_GMM(int nGaussianComponents, Eigen::MatrixXd x);
	
	Eigen::VectorXd multivariate_normal_pdf(Eigen::MatrixXd x, Eigen::VectorXd mean, Eigen::MatrixXd covariance);
	double multivariate_normal_pdf2(Eigen::VectorXd x, Eigen::VectorXd mean, Eigen::MatrixXd covariance);
	
	void project_on_Gaussians(estimator_attributes &GMM_obj, Eigen::MatrixXd x);
	
	Eigen::VectorXd GM_density(estimator_attributes GMM_obj);
	Eigen::VectorXd GM_density2(estimator_attributes GMM_obj, Eigen::MatrixXd x);
	
	double compute_log_likelihood(estimator_attributes GMM_obj);
	double compute_log_likelihood2(estimator_attributes GMM_obj, Eigen::MatrixXd x);
	
	void expectation_step(estimator_attributes &GMM_obj);
	void expectation_step2(estimator_attributes &GMM_obj, Eigen::MatrixXd x);
	
	void maximization_step(estimator_attributes &GMM_obj, Eigen::MatrixXd x);
	void maximization_step2(estimator_attributes &GMM_obj, Eigen::MatrixXd x);
	
	void write_to_file(estimator_attributes GMM_obj, Eigen::MatrixXd x);
	
	// K-means initialization
	vector<Eigen::VectorXd> kmeans_center_init(int nComponents, Eigen::MatrixXd x);
	void assign_cluster_labels(vector<Eigen::VectorXd> centers, Eigen::MatrixXd x, vector<int> &cluster_labels);
	void update_centers(vector<Eigen::VectorXd> &centers, Eigen::MatrixXd x, vector<int> cluster_labels);
	
	
	// Global variables
	Eigen::MatrixXd x_;
	Eigen::MatrixXd training_data_;
	Eigen::MatrixXd validation_data_;
	
	vector<int> cluster_indices_;
	
	int max_iter_;
	double convergence_tol_;
	double pi_;
	
	int nPoints_;
	int nTrainPoints_;
	int nValidationPoints_;
	int nDims_;
};

