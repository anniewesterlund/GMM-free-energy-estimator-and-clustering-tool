/**********************************
*
* 	ID_estimator.h
* 	Estimating the intrinsic dimension of a data set given the matrix of
* 	distances between points. 
*
*	Method from the paper:
*	Facco et al., Estimating the intrinsic dimension of datasets by a minimal neighborhood information (2017)
*
*	Annie Westerlund
*	2018, KTH
*
**********************************/
#include <vector>
#include <Eigen/Eigen>
#include <iostream>

class ID_estimator{
	
	public:
		ID_estimator();
		int estimate_ID(std::vector<std::vector<double>> distances);
		Eigen::MatrixXd get_distance_matrix();
		
	private:
		void set_distance_matrix(std::vector<std::vector<double>> distances);
		std::vector<double> get_2NN_ratios();
		std::vector<Eigen::VectorXd> compute_ratio_and_empirical_cummulate_pair(std::vector<double> ratios, int nPointsKeep);
		std::vector<Eigen::VectorXd> transform_xy_linear(Eigen::VectorXd ratios, Eigen::VectorXd empirical_cummulate);
		Eigen::VectorXd linear_regression(Eigen::VectorXd x, Eigen::VectorXd y);
		
		int nPoints_;
		Eigen::MatrixXd distance_matrix_;
};