/**********************************
*
* 	Free energy estimation using Gaussian mixture model and cross-validation 
*	to determine the number of components.
*	Annie Westerlund
*	
*	2018, KTH
*
**********************************/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "Kmeans.cpp"
#include "GaussianMixtureDensity.cpp"
#include "GMM.h"

estimator_attributes::estimator_attributes(int nGaussianComponents, Eigen::MatrixXd x):nComponents_{nGaussianComponents}{

	int nPoints = x.rows();
	int nDims = x.cols();
	estimator_attributes::nComponents_ = nGaussianComponents;
	
	estimator_attributes::centers_ = std::vector<Eigen::VectorXd>(nGaussianComponents);
	estimator_attributes::covariances_ = std::vector<Eigen::MatrixXd>(nGaussianComponents);
	estimator_attributes::amplitudes_ = Eigen::VectorXd(nGaussianComponents);
	estimator_attributes::Gaussian_projections_ = Eigen::MatrixXd(nPoints,nGaussianComponents);
	estimator_attributes::membership_weights_ = Eigen::MatrixXd(nGaussianComponents,nPoints);
	estimator_attributes::eye_stability_ = Eigen::MatrixXd(nDims,nDims);
	
	// Create GMM::eye_stability_ once
	for(int iRow = 0; iRow < nDims; iRow++){
		for(int iCol = 0; iCol < nDims; iCol++){
			estimator_attributes::eye_stability_(iRow,iCol) = 0;
			if(iRow==iCol) estimator_attributes::eye_stability_(iRow,iCol) = 1e-7;
		}
	}
	
	Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(nDims,nDims);
	Eigen::VectorXd average_point(nDims);
	
	// Set component means and amplitudes
	for(int iComponent = 0; iComponent < nGaussianComponents; iComponent++){
		estimator_attributes::amplitudes_(iComponent) = 1.0/double(nGaussianComponents);
	}
	
	Kmeans kmeans(nGaussianComponents,100);
	kmeans.cluster_data(x);
	estimator_attributes::centers_ = kmeans.get_cluster_centers();
	
	// Centroid of points
	average_point = x.colwise().mean();
	
	// Calculate covariance of points
	for(int iDim = 0; iDim < nDims; iDim++){
		for(int jDim = 0; jDim < nDims; jDim++){
			#pragma omp parallel for
			for(int iPoint = 0; iPoint < nPoints; iPoint++){
				covariance(iDim,jDim) += (x(iPoint,iDim)-average_point(iDim))*(x(iPoint,jDim)-average_point(jDim))/double((nPoints-1));
			}
		}
	}
	
	// Set initial covariance guess on all components to the covariance of the entire data set.
	for(int iComponent = 0; iComponent < nGaussianComponents; iComponent++){
		estimator_attributes::covariances_[iComponent] = covariance;
	}
};

GMM::GMM(std::vector<std::vector<double>> data, double temperature, double convergence_tol,
	 int max_iterations, std::string file_label){
	GMM::nPoints_ = data.size();
	GMM::temperature_ = temperature;
	GMM::convergence_tol_ = convergence_tol;
	GMM::max_iter_ = max_iterations;
	
	GMM::boltzmann_constant_ = 0.0019872041; // [kcal/(mol K)]
	
	GMM::file_label_ = file_label;
	
	if (GMM::nPoints_ > 0){
		GMM::cluster_indices_ = std::vector<int>(GMM::nPoints_);
		GMM::nDims_ = data[0].size();
	}else{
		std::cout << "No points supplied." << std::endl;
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
};

void GMM::expectation_step(estimator_attributes &GMM_attributes){
	
	Eigen::VectorXd normalization_factor = GMM::densityEvaluator_.get_density(GMM_attributes.amplitudes_,GMM_attributes.Gaussian_projections_);
	int nPoints = GMM_attributes.Gaussian_projections_.rows();
	
	for(int iComponent = 0; iComponent < GMM_attributes.nComponents_; iComponent++){
		#pragma omp parallel for
		for(int iPoint = 0; iPoint < nPoints; iPoint++){
			GMM_attributes.membership_weights_(iComponent,iPoint) = GMM_attributes.amplitudes_(iComponent)*GMM_attributes.Gaussian_projections_(iPoint,iComponent)/normalization_factor(iPoint);
		}
	}
	return;
}

void GMM::maximization_step(estimator_attributes &GMM_attributes, Eigen::MatrixXd x){
	
	int nPoints = x.rows();
	Eigen::VectorXd weighted_points(GMM::nDims_);
	Eigen::MatrixXd exponent1(GMM::nDims_,nPoints);
	Eigen::MatrixXd exponent2(nPoints,GMM::nDims_);
	Eigen::MatrixXd covariance(GMM::nDims_,GMM::nDims_);
	
	// Update amplitudes
	Eigen::VectorXd component_weights = GMM_attributes.membership_weights_.rowwise().sum();
	GMM_attributes.amplitudes_ = component_weights/double(nPoints);
	GMM_attributes.amplitudes_ /= GMM_attributes.amplitudes_.sum();
	
	// Update means and covariances of each component
	for(int iComponent = 0; iComponent < GMM_attributes.nComponents_; iComponent++){
		// Update centers
		weighted_points = GMM_attributes.membership_weights_.row(iComponent)*x;	
		GMM_attributes.centers_[iComponent] = 1.0/component_weights(iComponent)*weighted_points;
		
		// Update covariances
		for(int iDim = 0; iDim < GMM::nDims_; iDim++){
			#pragma omp parallel for
    		for(int iPoint = 0; iPoint < nPoints; iPoint++){
    			exponent1(iDim,iPoint) = x(iPoint,iDim)-GMM_attributes.centers_[iComponent](iDim);
    		}
    	}
    	
    	for(int iDim = 0; iDim < GMM::nDims_; iDim++){
    		#pragma omp parallel for
    		for(int iPoint = 0; iPoint < nPoints; iPoint++){
    			exponent2(iPoint,iDim) = exponent1(iDim,iPoint)*GMM_attributes.membership_weights_(iComponent,iPoint);
    		}
    	}
   	 	covariance = exponent1*exponent2;
    	GMM_attributes.covariances_[iComponent] = covariance/component_weights(iComponent)+ GMM_attributes.eye_stability_;	
    	
	}
	return;
}

estimator_attributes GMM::train_density(int nGaussianComponents, Eigen::MatrixXd x){
	
	double log_likelihood;
	double previous_log_likelihood = 0.0;
	int counter = 0;
	estimator_attributes GMM_attributes = estimator_attributes(nGaussianComponents, x);
	
	for(int iIter = 0; iIter < 10; iIter++){
		GMM_attributes.Gaussian_projections_ = GMM::densityEvaluator_.project_on_Gaussians(GMM_attributes.centers_,
			GMM_attributes.covariances_,GMM_attributes.nComponents_, x);
		
		log_likelihood = GMM::densityEvaluator_.compute_log_likelihood(GMM_attributes.amplitudes_,GMM_attributes.Gaussian_projections_);
		
		previous_log_likelihood = -1e24;
		counter = 0;
		
		while (abs(previous_log_likelihood-log_likelihood) > GMM::convergence_tol_){
			previous_log_likelihood = log_likelihood;
			// Run expectation-maximization
			GMM::expectation_step(GMM_attributes);
			GMM::maximization_step(GMM_attributes, x);
			
			GMM_attributes.Gaussian_projections_ = GMM::densityEvaluator_.project_on_Gaussians(GMM_attributes.centers_, GMM_attributes.covariances_,GMM_attributes.nComponents_, x);
			log_likelihood = GMM::densityEvaluator_.compute_log_likelihood(GMM_attributes.amplitudes_,GMM_attributes.Gaussian_projections_);
			
			counter++;
			if (counter >= GMM::max_iter_){
				break;
			}
		}
		
		if (std::isnormal(log_likelihood)){
			break;
		}
		
	}
	std::cout << "nComponents: " << nGaussianComponents << ", nIterations: " << counter << "\n";
	
	return GMM_attributes;
}

double GMM::validate_density(estimator_attributes GMM_attributes, Eigen::MatrixXd x){
	GMM_attributes.Gaussian_projections_ = GMM::densityEvaluator_.project_on_Gaussians(GMM_attributes.centers_,
			GMM_attributes.covariances_,GMM_attributes.nComponents_, x);;
	double log_likelihood = GMM::densityEvaluator_.compute_log_likelihood(GMM_attributes.amplitudes_, GMM_attributes.Gaussian_projections_);
	return log_likelihood;
}

void GMM::fitDensity(int nMinGaussianComponents, int nMaxGaussianComponents){
	
	Eigen::VectorXd log_likelihood(nMaxGaussianComponents-nMinGaussianComponents+1);
	GMM::densityEvaluator_ = GMDensity();
	//GMM::expectationMaximization_ = EM(GMM::nDims_);
	
	for( int nGaussians = nMinGaussianComponents; nGaussians <= nMaxGaussianComponents; nGaussians++){
		estimator_attributes GMM_attributes = GMM::train_density(nGaussians, GMM::training_data_);
		log_likelihood(nGaussians-nMinGaussianComponents) = GMM::validate_density(GMM_attributes, GMM::validation_data_);
	}
	
	std::cout << "Log-likelihoods: " << log_likelihood << "\n";
	int argmax = 0;
	double max_log_likelihood = round(log_likelihood(0)*100);
	
	for(int i = 0; i < log_likelihood.rows(); i++){
		if (max_log_likelihood < round(log_likelihood(i)*100) && std::isnormal(log_likelihood(i))){
			argmax = i;
			max_log_likelihood = round(log_likelihood(i)*100);
		}
	}
	std::cout << "Argmax: " << argmax << "\n";
	
	// Estimate density with the best parameter set
	estimator_attributes GMM_attributes_final = GMM::train_density(nMinGaussianComponents+argmax, GMM::x_);
	
	GMM::extract_clusters(GMM_attributes_final);
	GMM::write_to_file(GMM_attributes_final);
	
	return;
}

void GMM::write_to_file(estimator_attributes GMM_attributes){
	GMM_attributes.Gaussian_projections_ = GMM::densityEvaluator_.project_on_Gaussians(GMM_attributes.centers_,
			GMM_attributes.covariances_,GMM_attributes.nComponents_, GMM::x_);
	Eigen::VectorXd density = GMM::densityEvaluator_.get_density(GMM_attributes.amplitudes_,GMM_attributes.Gaussian_projections_);
	
	#pragma omp parallel for
	for(int i = 0; i< density.rows(); i++){
		density(i) = -GMM::temperature_*GMM::boltzmann_constant_*log(density(i));  // Free energy in [kT] unit
	}
	
	// Write the free energy to file
	std::ofstream fID;
	fID.open ("free_energy_" + GMM::file_label_ + ".txt");
	fID << density;
	fID.close();
	
	std::cout << "Free energy saved to file: " << "free_energy_" << GMM::file_label_ << ".txt" << std::endl;
	
	return;
}

void GMM::write_cluster_indices(){
	std::string file_name = "cluster_indices_" + GMM::file_label_ + ".txt";
	
	std::ofstream fID;
	fID.open (file_name);
	
	for(int i = 0; i < GMM::cluster_indices_.size(); i++){
		fID << GMM::cluster_indices_[i] << "\n";
	}
	fID.close();
	std::cout << "Cluster indices saved to file: " << file_name << std::endl;
	return;
}

void GMM::extract_clusters(estimator_attributes GMM_attributes){
		
	std::cout << "Extracting clusters " << std::endl;
	int nPoints = GMM::x_.rows();
	
	// Project on the estimated Gaussians. A point belongs to the cluster with largest component (Gaussian) contribution.
	GMM_attributes.Gaussian_projections_ = GMM::densityEvaluator_.project_on_Gaussians(GMM_attributes.centers_,
			GMM_attributes.covariances_,GMM_attributes.nComponents_, GMM::x_);
	
	Eigen::VectorXd component_contributions(GMM_attributes.nComponents_);	
	GMM::cluster_indices_ = std::vector<int>(nPoints);
	double max_contribution = 0.0;
	
	for(int i = 0; i < nPoints; i++){
		component_contributions = GMM_attributes.Gaussian_projections_.row(i);
		GMM::cluster_indices_[i] = 1;
		max_contribution = component_contributions[0];
		for (int j = 1; j < GMM_attributes.nComponents_; j++){
			if(max_contribution < component_contributions[j]){
				max_contribution = component_contributions[j];
				GMM::cluster_indices_[i] = j+1;
			}
		}
	}
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
