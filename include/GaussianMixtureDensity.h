/**********************************
*
* 	Gaussian mixture density class
*	Annie Westerlund
*	
*	2018, KTH
*
**********************************/
#include <vector>
#include <Eigen/Eigen>
#include <unsupported/Eigen/MatrixFunctions>
#include <omp.h>

class GMDensity {
	
	public:
	GMDensity();
	
	Eigen::MatrixXd project_on_Gaussians(std::vector<Eigen::VectorXd> means, std::vector<Eigen::MatrixXd> covariances, int nComponents, Eigen::MatrixXd x);
	
	Eigen::VectorXd get_density(Eigen::VectorXd amplitudes, Eigen::MatrixXd gaussian_projections);	
	double compute_log_likelihood(Eigen::VectorXd amplitudes, Eigen::MatrixXd gaussian_projections);
	
	private:
	Eigen::VectorXd multivariate_normal_pdf(Eigen::MatrixXd x, Eigen::VectorXd mean, Eigen::MatrixXd covariance);
	
	double pi_;
};