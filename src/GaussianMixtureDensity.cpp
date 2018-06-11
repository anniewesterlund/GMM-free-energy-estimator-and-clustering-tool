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
	GMDensity::pi_ = 3.14159265359;
}

Eigen::VectorXd GMDensity::multivariate_normal_pdf(Eigen::MatrixXd x, Eigen::VectorXd mean, Eigen::MatrixXd covariance){
	// Calculating the PDF of one Gaussian
	int nPoints = x.rows();
	int nDims = x.cols();
	Eigen::VectorXd component_projection(nPoints);
	
	Eigen::MatrixXd inv_covariance = covariance.inverse();
	double cov_det = covariance.determinant();
	double normalization_factor = sqrt(pow(2.0*GMDensity::pi_,nDims)*cov_det);
	
	#pragma omp parallel for
	for(int iPoint = 0; iPoint < nPoints; iPoint++){
		Eigen::VectorXd tmp_dev =  x.row(iPoint).transpose()-mean;		
		double tmp_exponent = tmp_dev.transpose()*inv_covariance*tmp_dev;
		component_projection(iPoint) = exp(-tmp_exponent/2)/normalization_factor;
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
	
	double log_likelihood = 0.0;
	#pragma omp parallel for reduction (+:log_likelihood)
	for(int i = 0; i < GM_pdf.size(); i++){
		log_likelihood += log(GM_pdf(i));
	}
	return log_likelihood;
}