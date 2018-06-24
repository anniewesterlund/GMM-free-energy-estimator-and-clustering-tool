/**********************************
*
* 	Gaussian mixture density class
*	Annie Westerlund
*	
*	2018, KTH
*
**********************************/
#include "GaussianMixtureDensity.h"

GMDensity::GMDensity(){
	GMDensity::pi_ = std::acos(-1.0);
}

Eigen::VectorXd GMDensity::multivariate_normal_pdf(Eigen::MatrixXd x, Eigen::VectorXd mean, Eigen::MatrixXd covariance){
	// Calculating the PDF of one Gaussian
	int nPoints = x.rows();
	int nDims = x.cols();
	Eigen::VectorXd component_projection(nPoints);
	
	Eigen::MatrixXd inv_covariance = covariance.inverse();
	double cov_det = covariance.determinant();
	double normalization_factor = std::sqrt(pow(2.0*GMDensity::pi_,nDims)*cov_det);
	double exponent;
	Eigen::VectorXd deviation;
	
	#pragma omp parallel for
	for(int iPoint = 0; iPoint < nPoints; iPoint++){
		deviation =  x.row(iPoint).transpose()-mean;		
		exponent = deviation.transpose()*inv_covariance*deviation;
		component_projection(iPoint) = std::exp(-exponent/2)/normalization_factor;
	}
	
	return component_projection;
}


Eigen::MatrixXd GMDensity::project_on_Gaussians(std::vector<Eigen::VectorXd> means, std::vector<Eigen::MatrixXd> covariances, int nComponents, Eigen::MatrixXd x){
	
	Eigen::MatrixXd gaussian_projections(x.rows(), nComponents);
	
	for(int iComponent = 0; iComponent < nComponents; iComponent++){
		gaussian_projections.col(iComponent) = GMDensity::multivariate_normal_pdf(x, means[iComponent], covariances[iComponent]);
	}
	
	return gaussian_projections;
}


Eigen::VectorXd GMDensity::get_density(Eigen::VectorXd amplitudes, Eigen::MatrixXd gaussian_projections){
	amplitudes = amplitudes/amplitudes.sum();
	Eigen::VectorXd GM_pdf = gaussian_projections*amplitudes;
	return GM_pdf;
}


double GMDensity::compute_log_likelihood(Eigen::VectorXd amplitudes, Eigen::MatrixXd gaussian_projections){
	
	Eigen::VectorXd GM_pdf = GMDensity::get_density(amplitudes, gaussian_projections);
	GM_pdf = GM_pdf.log();
	double log_likelihood = GM_pdf.sum();
	
	return log_likelihood;
}