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
#include <chrono>
#include <algorithm>
#include <sstream>
#include <cmath>
#include "GMM.h"

using namespace std;

	
GMM::GMM(vector<vector<double>> data, double convergence_tol, int max_iterations){
	GMM::nPoints_ = data.size();
	GMM::convergence_tol_ = convergence_tol;
	GMM::max_iter_ = max_iterations;
	GMM::pi_ = 3.14159265359;
			
	if (GMM::nPoints_ > 0){
		GMM::cluster_indices_ = vector<int>(GMM::nPoints_);
		GMM::nDims_ = data[0].size();
	}else{
		cout << "No points supplied." << endl;
		exit(1);
	}
	
	// Store data in matrix of size [nPoints x nDims]
	// Set training and validation set with 50/50 split
	GMM::x_ = Eigen::MatrixXd(GMM::nPoints_,GMM::nDims_);
	GMM::training_data_ = Eigen::MatrixXd(GMM::nPoints_/2,GMM::nDims_);
	GMM::validation_data_ = Eigen::MatrixXd(int(round(double(GMM::nPoints_)/2.0)),GMM::nDims_);
	
	for(int i = 0; i < GMM::nPoints_; i++){
		for(int j = 0; j < GMM::nDims_; j++){
			GMM::x_(i,j) = data[i][j];
			if(i < GMM::nPoints_/2){
				GMM::training_data_(i,j) = data[i][j];
			}else{
				GMM::validation_data_(i-nPoints_/2,j) = data[i][j];
			}
		}
	}
}

void GMM::assign_cluster_labels(vector<Eigen::VectorXd> centers, Eigen::MatrixXd x, vector<int> &cluster_labels){
	// Assign k-means cluster labels	
	int nPoints = x.rows();
	int nComponents = centers.size();
	Eigen::VectorXd tmpPoint;
	double minDistance;
	double tmpDistance;
	
	for(int iPoint=0; iPoint < nPoints; iPoint++){
		tmpPoint = x.row(iPoint);
		minDistance = (centers[0]-tmpPoint).norm();
		for(int iComponent = 1; iComponent < nComponents; iComponent++){
			tmpDistance = (centers[iComponent]-tmpPoint).norm();
			cluster_labels[iPoint] = 0;
			if (tmpDistance < minDistance){
				tmpDistance = minDistance;
				cluster_labels[iPoint] = iComponent;
			}
		}
	}
	return;
}

void GMM::update_centers(vector<Eigen::VectorXd> &centers, Eigen::MatrixXd x, vector<int> cluster_labels){
	// Update K-means centers
	int nComponents = centers.size();
	int nPoints = x.rows();
	int nDims = centers[0].size();
	double counter = 0.0;
	Eigen::VectorXd tmpCenter(nDims);
	
	for(int iComponent = 0; iComponent < nComponents; iComponent++){
		for(int iDim = 0; iDim < nDims; iDim++){
			tmpCenter(iDim) = 0.0;
		}
		counter = 0.0;
		for(int iPoint = 0; iPoint < nPoints; iPoint++){
			if(cluster_labels[iPoint] == iComponent){
				counter += 1.0;
				tmpCenter += x.row(iPoint);
			}
		}
		
		for(int iDim = 0; iDim < nDims; iDim++){
			tmpCenter(iDim) /= counter;
		}
		if (counter > 0){
			centers[iComponent] = tmpCenter;
		}
	}
	return;
}

vector<Eigen::VectorXd> GMM::kmeans_center_init(int nComponents, Eigen::MatrixXd x){
	int nPoints = x.rows();
	vector<Eigen::VectorXd> centers(nComponents);
	vector<int> cluster_labels(nPoints);
	
	// Initialize centers from points
	vector<int> indices;
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	
	// Get vector with all indices
  	for (int i=0; i < nPoints; ++i){ 
  		indices.push_back(i); 
  	}
	
  	// Permute indices
  	shuffle(indices.begin(), indices.end(), default_random_engine(seed));
	for(int iComponent = 0; iComponent < nComponents; iComponent++){
		centers[iComponent] = x.row(indices[iComponent]);
	}
	
	for(int iIter = 0; iIter < 100; iIter++){
		GMM::assign_cluster_labels(centers, x, cluster_labels);
		GMM::update_centers(centers, x, cluster_labels);
	}
	
	return centers;
}

estimator_attributes GMM::initialize_GMM(int nGaussianComponents, Eigen::MatrixXd x){

	int nPoints = x.rows();

	estimator_attributes new_GMM_object;
	new_GMM_object.nComponents_ = nGaussianComponents;
	
	new_GMM_object.centers_ = vector<Eigen::VectorXd>(nGaussianComponents);
	new_GMM_object.covariances_ = vector<Eigen::MatrixXd>(nGaussianComponents);
	new_GMM_object.amplitudes_ = Eigen::VectorXd(nGaussianComponents);
	new_GMM_object.Gaussian_projections_ = Eigen::MatrixXd(GMM::nPoints_,nGaussianComponents);
	new_GMM_object.membership_weights_ = Eigen::MatrixXd(nGaussianComponents,nPoints);
	new_GMM_object.eye_stability_ = Eigen::MatrixXd(GMM::nDims_,GMM::nDims_);
	
	// Create GMM::eye_stability_ once
	for(int iRow = 0; iRow < GMM::nDims_; iRow++){
		for(int iCol = 0; iCol < GMM::nDims_; iCol++){
			new_GMM_object.eye_stability_(iRow,iCol) = 0;
			if(iRow==iCol) new_GMM_object.eye_stability_(iRow,iCol) = 1e-7;
		}
	}
	
	Eigen::MatrixXd tmp_cov(GMM::nDims_,GMM::nDims_);
	Eigen::VectorXd average_point(GMM::nDims_);
	
	for(int i = 0; i < GMM::nDims_; i++){
		for(int j = 0; j < GMM::nDims_; j++){
			tmp_cov(i,j) = 0;
		}
	}
	
	// Set component means and amplitudes
	for(int iComponent = 0; iComponent < nGaussianComponents; iComponent++){
		new_GMM_object.amplitudes_(iComponent) = 1.0/double(nGaussianComponents);
	}
	
	new_GMM_object.centers_ = GMM::kmeans_center_init(nGaussianComponents, x);
	
	// Centroid of points
	average_point = x.colwise().mean();
	
	// Calculate covariance of points
	for(int iDim = 0; iDim < GMM::nDims_; iDim++){
		for(int jDim = 0; jDim < GMM::nDims_; jDim++){
			for(int iPoint = 0; iPoint < nPoints; iPoint++){
				tmp_cov(iDim,jDim) += (x(iPoint,iDim)-average_point(iDim))*(x(iPoint,jDim)-average_point(jDim))/double((nPoints-1));
			}
		}
	}
	
	// Set initial covariance guess on all components to the covariance of the entire data set.
	for(int iComponent = 0; iComponent < nGaussianComponents; iComponent++){
		new_GMM_object.covariances_[iComponent] = tmp_cov;
	}
	
	return new_GMM_object;
}	

Eigen::VectorXd GMM::multivariate_normal_pdf(Eigen::MatrixXd x, Eigen::VectorXd mean, Eigen::MatrixXd covariance){
	// Calculating the PDF of one Gaussian
	int nPoints = x.rows();
	Eigen::VectorXd tmpPoint(GMM::nDims_);
	Eigen::VectorXd component_projection(nPoints);
	Eigen::VectorXd tmp_dev(GMM::nDims_);
	double tmp_exponent;
	
	Eigen::MatrixXd inv_covariance = covariance.inverse();
	double cov_det = covariance.determinant();
	
	for(int iPoint = 0; iPoint < nPoints; iPoint++){
		tmpPoint = x.row(iPoint);
		tmp_dev = tmpPoint-mean;		
		tmp_exponent = tmp_dev.transpose()*inv_covariance*tmp_dev;
		component_projection(iPoint) = exp(-tmp_exponent/2)/(sqrt(pow(2.0*GMM::pi_,GMM::nDims_)*cov_det));
	}
	
	return component_projection;
}

void GMM::project_on_Gaussians(estimator_attributes &GMM_obj, Eigen::MatrixXd x){
	
	GMM_obj.Gaussian_projections_ = Eigen::MatrixXd(x.rows(),GMM_obj.nComponents_);
	
	for(int iComponent = 0; iComponent < GMM_obj.nComponents_; iComponent++){
		GMM_obj.Gaussian_projections_.col(iComponent) = GMM::multivariate_normal_pdf(x, GMM_obj.centers_[iComponent], GMM_obj.covariances_[iComponent]);
	}
	return;
}

Eigen::VectorXd GMM::GM_density(estimator_attributes GMM_obj){
	GMM_obj.amplitudes_ = GMM_obj.amplitudes_/GMM_obj.amplitudes_.sum();
	Eigen::VectorXd GM_pdf = GMM_obj.Gaussian_projections_*GMM_obj.amplitudes_;
	return GM_pdf;
}


double GMM::compute_log_likelihood(estimator_attributes GMM_obj){
	
	Eigen::VectorXd GM_pdf = GMM::GM_density(GMM_obj);
	
	double log_likelihood = 0.0;
	for(int i = 0; i < GM_pdf.size(); i++){
		log_likelihood += log(GM_pdf(i));
	}
	return log_likelihood;
}

double GMM::multivariate_normal_pdf2(Eigen::VectorXd x, Eigen::VectorXd mean, Eigen::MatrixXd covariance){
	// Calculating the PDF of one Gaussian and one point
	Eigen::VectorXd tmpPoint(GMM::nDims_);
	double component_projection;
	Eigen::VectorXd tmp_dev(GMM::nDims_);
	double tmp_exponent;
	
	Eigen::MatrixXd inv_covariance = covariance.inverse();
	double cov_det = covariance.determinant();

	tmp_dev = x-mean;		
	tmp_exponent = tmp_dev.transpose()*inv_covariance*tmp_dev;
	component_projection = exp(-tmp_exponent/2)/(sqrt(pow(2.0*GMM::pi_,GMM::nDims_)*cov_det));
	
	return component_projection;
}

Eigen::VectorXd GMM::GM_density2(estimator_attributes GMM_obj, Eigen::MatrixXd x){
	GMM_obj.amplitudes_ = GMM_obj.amplitudes_/GMM_obj.amplitudes_.sum();
	int nPoints = x.rows();
	Eigen::VectorXd GM_pdf(nPoints);
	
	for(int iPoint = 0; iPoint < nPoints; iPoint++){
		GM_pdf(iPoint) = 0.0;
		for(int iComponent = 0; iComponent < GMM_obj.nComponents_; iComponent++){
			GM_pdf(iPoint) += GMM::multivariate_normal_pdf2(x.row(iPoint), GMM_obj.centers_[iComponent], GMM_obj.covariances_[iComponent])*GMM_obj.amplitudes_(iComponent);
		}
	}
	return GM_pdf;
}


double GMM::compute_log_likelihood2(estimator_attributes GMM_obj, Eigen::MatrixXd x){
	
	Eigen::VectorXd GM_pdf = GMM::GM_density2(GMM_obj,x);
	
	double log_likelihood = 0.0;
	for(int i = 0; i < GM_pdf.cols(); i++){
		log_likelihood += log(GM_pdf(i));
	}
	return log_likelihood;
}

void GMM::expectation_step2(estimator_attributes &GMM_obj, Eigen::MatrixXd x){
	
	Eigen::VectorXd normalization_factor = GMM::GM_density2(GMM_obj,x);
	int nPoints = x.rows();
	
	for(int iComponent = 0; iComponent < GMM_obj.nComponents_; iComponent++){
		for(int iPoint = 0; iPoint < nPoints; iPoint++){
			GMM_obj.membership_weights_(iComponent,iPoint) = GMM_obj.amplitudes_(iComponent)*GMM::multivariate_normal_pdf2(x.row(iPoint), GMM_obj.centers_[iComponent], GMM_obj.covariances_[iComponent]);
			GMM_obj.membership_weights_(iComponent,iPoint) = GMM_obj.membership_weights_(iComponent,iPoint)/(normalization_factor(iPoint)+1e-5);
		}
	}
	return;
}

void GMM::maximization_step2(estimator_attributes &GMM_obj, Eigen::MatrixXd x){
	
	int nPoints = x.rows();
	Eigen::VectorXd weighted_points;
	Eigen::MatrixXd tmp_exp1(GMM::nDims_,nPoints);
	Eigen::MatrixXd tmp_exp2(nPoints,GMM::nDims_);
	Eigen::MatrixXd tmp_cov(GMM::nDims_,GMM::nDims_);
	
	// Update amplitudes
	Eigen::VectorXd component_weights = GMM_obj.membership_weights_.rowwise().sum();
	GMM_obj.amplitudes_ = component_weights/double(nPoints);
	GMM_obj.amplitudes_ /= GMM_obj.amplitudes_.sum();
	
	// Update means and covariances of each component
	for(int iComponent = 0; iComponent < GMM_obj.nComponents_; iComponent++){
		// Update means
		weighted_points = GMM_obj.membership_weights_.row(iComponent)*x;		
		GMM_obj.centers_[iComponent] = 1.0/component_weights(iComponent)*weighted_points;
		
		// Update covariances
		for(int iDim = 0; iDim < GMM::nDims_; iDim++){
    		tmp_exp1.col(iDim) = x.row(iDim)-GMM_obj.centers_[iComponent].transpose();
    		for(int iPoint = 0; iPoint < nPoints; iPoint++){
    			tmp_exp2(iPoint,iDim) = tmp_exp1(iDim,iPoint)*GMM_obj.membership_weights_(iComponent,iPoint);
    		}
    	}
    	
   	 	tmp_cov = tmp_exp1*tmp_exp2;
    	GMM_obj.covariances_[iComponent] = 1.0/component_weights(iComponent)*tmp_cov + GMM_obj.eye_stability_;	
    	
	}
	return;
}


void GMM::expectation_step(estimator_attributes &GMM_obj){
	
	Eigen::VectorXd normalization_factor = GMM::GM_density(GMM_obj);
	int nPoints = GMM_obj.Gaussian_projections_.rows();
	
	for(int iComponent = 0; iComponent < GMM_obj.nComponents_; iComponent++){
		for(int iPoint = 0; iPoint < nPoints; iPoint++){
			GMM_obj.membership_weights_(iComponent,iPoint) = GMM_obj.amplitudes_(iComponent)*GMM_obj.Gaussian_projections_(iPoint,iComponent)/normalization_factor(iPoint);
		}
	}
	return;
}

void GMM::maximization_step(estimator_attributes &GMM_obj, Eigen::MatrixXd x){
	
	int nPoints = x.rows();
	Eigen::VectorXd weighted_points(GMM::nDims_);
	Eigen::MatrixXd tmp_exp1(GMM::nDims_,nPoints);
	Eigen::MatrixXd tmp_exp2(nPoints,GMM::nDims_);
	Eigen::MatrixXd tmp_cov(GMM::nDims_,GMM::nDims_);
		
	// Update amplitudes
	Eigen::VectorXd component_weights = GMM_obj.membership_weights_.rowwise().sum();
	GMM_obj.amplitudes_ = component_weights/double(nPoints);
	GMM_obj.amplitudes_ /= GMM_obj.amplitudes_.sum();
	
	// Update means and covariances of each component
	for(int iComponent = 0; iComponent < GMM_obj.nComponents_; iComponent++){
		// Update centers
		weighted_points = GMM_obj.membership_weights_.row(iComponent)*x;	
		GMM_obj.centers_[iComponent] = 1.0/component_weights(iComponent)*weighted_points;
		
		// Update covariances
		for(int iDim = 0; iDim < GMM::nDims_; iDim++){
    		for(int iPoint = 0; iPoint < nPoints; iPoint++){
    			tmp_exp1(iDim,iPoint) = x(iPoint,iDim)-GMM_obj.centers_[iComponent](iDim);
    		}
    	}
    	
    	for(int iDim = 0; iDim < GMM::nDims_; iDim++){
    		for(int iPoint = 0; iPoint < nPoints; iPoint++){
    			tmp_exp2(iPoint,iDim) = tmp_exp1(iDim,iPoint)*GMM_obj.membership_weights_(iComponent,iPoint);
    		}
    	}
   	 	tmp_cov = tmp_exp1*tmp_exp2;
    	GMM_obj.covariances_[iComponent] = tmp_cov/component_weights(iComponent)+ GMM_obj.eye_stability_;	
    	
	}
	return;
}

estimator_attributes GMM::train_density(int nGaussianComponents, Eigen::MatrixXd x){
	
	estimator_attributes GMM_obj;
	double log_likelihood;
	double previous_log_likelihood = 0.0;
	int counter = 0;
	
	for(int iIter = 0; iIter < 10; iIter++){
		GMM_obj = GMM::initialize_GMM(nGaussianComponents, x);
		
		GMM::project_on_Gaussians(GMM_obj, x);
		
		log_likelihood = GMM::compute_log_likelihood(GMM_obj);
		
		previous_log_likelihood = -1e24;
		counter = 0;
		
		while (abs(previous_log_likelihood-log_likelihood) > GMM::convergence_tol_){
			previous_log_likelihood = log_likelihood;
			// Run expectation-maximization
			GMM::expectation_step(GMM_obj);
			GMM::maximization_step(GMM_obj, x);
			
			GMM::project_on_Gaussians(GMM_obj, x);
			log_likelihood = GMM::compute_log_likelihood(GMM_obj);
			
			counter++;
			if (counter >= GMM::max_iter_){
				break;
			}
		}
		
		if (isnormal(log_likelihood)){
			break;
		}
		
	}
	clog << "nComponents: " << nGaussianComponents << ", nIterations: " << counter << "\n";
	
	return GMM_obj;
}

double GMM::validate_density(estimator_attributes GMM_obj, Eigen::MatrixXd x){
	GMM::project_on_Gaussians(GMM_obj, x);
	double log_likelihood = GMM::compute_log_likelihood(GMM_obj);
	return log_likelihood;
}

void GMM::fitDensity(int nMinGaussianComponents, int nMaxGaussianComponents){
	
	Eigen::VectorXd log_likelihood(nMaxGaussianComponents-nMinGaussianComponents+1);
	
	for( int nGaussians = nMinGaussianComponents; nGaussians <= nMaxGaussianComponents; nGaussians++){
		estimator_attributes GMM_obj = GMM::train_density(nGaussians, GMM::training_data_);
		log_likelihood(nGaussians-nMinGaussianComponents) = GMM::validate_density(GMM_obj, GMM::validation_data_);
	}
	
	clog << "Log-likelihoods: " << log_likelihood << "\n";
	int argmax = 0;
	double max_log_likelihood = round(log_likelihood(0)*100);
	
	for(int i = 0; i < log_likelihood.rows(); i++){
		if (max_log_likelihood < round(log_likelihood(i)*100) && isnormal(log_likelihood(i))){
			argmax = i;
			max_log_likelihood = round(log_likelihood(i)*100);
		}
	}
	clog << "Argmax: " << argmax << "\n";
	
	// Estimate density with the best parameter set
	estimator_attributes GMM_obj_final = GMM::train_density(nMinGaussianComponents+argmax, GMM::x_);
	
	GMM::write_to_file(GMM_obj_final, GMM::x_);
	return;
}

void GMM::write_to_file(estimator_attributes GMM_obj, Eigen::MatrixXd x){
	GMM::project_on_Gaussians(GMM_obj, x);
	Eigen::VectorXd density = GMM::GM_density(GMM_obj);
	
	cout << density << endl;
	
	for(int iComp = 0; iComp < GMM_obj.nComponents_;iComp++){
		clog << GMM_obj.centers_[iComp].transpose() << "\n";
	}
	
	clog << GMM_obj.amplitudes_ << endl;
	return;
}

/*
// To do
Eigen::VectorXd GMM::getDensity(){
	return;
}

vector<Eigen::VectorXd> getCenters(){
	return GMM::centers_;
}

vector<Eigen::MatrixXd> getCovariances(){
	return GMM::covariances_;
}

Eigen::VectorXd getAmplitudes(){
	return GMM::amplitudes_;
}*/

//vector<int> GMM::getClusterIndices();
	
//vector<int> GMM::getClusterIndicesWithHalo();
