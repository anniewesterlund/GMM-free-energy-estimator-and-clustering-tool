#include "Kmeans.h"
#include <chrono>
#include <random>
#include <algorithm>

Kmeans::Kmeans(int nComponents, int nIterations):nComponents_{nComponents},nIterations_{nIterations} 
{
	Kmeans::cluster_centers_= std::vector<Eigen::VectorXd>(Kmeans::nComponents_);
}

void Kmeans::assign_cluster_labels(Eigen::MatrixXd x){
	// Assign k-means cluster labels	
	int nPoints = x.rows();
	
	#pragma omp parallel for
	for(int iPoint=0; iPoint < nPoints; iPoint++){
		Eigen::VectorXd tmpPoint = x.row(iPoint);
		double minDistance = (Kmeans::cluster_centers_[0]-tmpPoint).norm();
		for(int iComponent = 1; iComponent < Kmeans::nComponents_; iComponent++){
			double tmpDistance = (Kmeans::cluster_centers_[iComponent]-tmpPoint).norm();
			Kmeans::cluster_labels_[iPoint] = 0;
			if (tmpDistance < minDistance){
				tmpDistance = minDistance;
				Kmeans::cluster_labels_[iPoint] = iComponent;
			}
		}
	}
	return;
}

void Kmeans::update_centers(Eigen::MatrixXd x){
	// Update K-means centers
	int nPoints = x.rows();
	int nDims = Kmeans::cluster_centers_[0].size();
	double counter = 0.0;
	Eigen::VectorXd tmpCenter(nDims);
	
	for(int iComponent = 0; iComponent < Kmeans::nComponents_; iComponent++){
		for(int iDim = 0; iDim < nDims; iDim++){
			double tmpCenterCoord = 0.0;
			
			#pragma omp parallel for reduction (+:tmpCenterCoord)
			for(int iPoint = 0; iPoint < nPoints; iPoint++){
				if(Kmeans::cluster_labels_[iPoint] == iComponent){
					tmpCenterCoord += x(iPoint,iDim);
				}
			}
			tmpCenter(iDim) = tmpCenterCoord;
		}
		
		counter = 0.0;
		#pragma omp parallel for reduction (+:counter)
		for(int iPoint = 0; iPoint < nPoints; iPoint++){
			if(Kmeans::cluster_labels_[iPoint] == iComponent){
				counter += 1.0;
			}
		}
		
		for(int iDim = 0; iDim < nDims; iDim++){
			tmpCenter(iDim) /= counter;
		}
		
		if (counter > 0){
			Kmeans::cluster_centers_[iComponent] = tmpCenter;
		}
	}
	return;
}

void Kmeans::cluster_data(Eigen::MatrixXd x){
	int nPoints = x.rows();
	Kmeans::cluster_labels_ = std::vector<int>(nPoints);
	
	// Initialize centers from points
	std::vector<int> indices;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	
	// Get vector with all indices
  	for (int i=0; i < nPoints; ++i){ 
  		indices.push_back(i); 
  	}
	
  	// Permute indices
  	std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
  	
	for(int iComponent = 0; iComponent < Kmeans::nComponents_; iComponent++){
		Kmeans::cluster_centers_[iComponent] = x.row(indices[iComponent]);
	}
	
	for(int iIter = 0; iIter < Kmeans::nIterations_; iIter++){
		Kmeans::assign_cluster_labels(x);
		Kmeans::update_centers(x);
	}
	
	return;
}

std::vector<Eigen::VectorXd> Kmeans::get_cluster_centers(){
	return Kmeans::cluster_centers_;
}

std::vector<int> Kmeans::get_cluster_labels(){
	return Kmeans::cluster_labels_;
}
