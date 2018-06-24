#include "ID_estimator.h"

ID_estimator::ID_estimator(){}

int ID_estimator::estimate_ID(std::vector<std::vector<double>> distances){
	std::cout << "Estimating intrinsic dimension." << std::endl;
	
	// Construct the square-formed distance matrix
	ID_estimator::set_distance_matrix(distances);
	
	int nPointsKeep = int(std::floor(0.9*ID_estimator::nPoints_));	
	
	// Compute nearest neighbor ratios and empirical cummulate
	std::vector<double> nearest_neighbor_ratios = ID_estimator::get_2NN_ratios();
	std::vector<Eigen::VectorXd> xy;
	
	xy = ID_estimator::compute_ratio_and_empirical_cummulate_pair(nearest_neighbor_ratios, nPointsKeep);
	xy = ID_estimator::transform_xy_linear(xy[0], xy[1]);
	
	// Perform linear regression to get line parameters
	Eigen::VectorXd line_parameters = ID_estimator::linear_regression(xy[0], xy[1]);
	
	// Intrinsic dimension is the estimated slope
	int intrinsic_dimension = int(std::round(line_parameters[0]));
	std::cout << "Estimated intrinsic dimension: " << intrinsic_dimension << std::endl;
	return intrinsic_dimension;
}

void ID_estimator::set_distance_matrix(std::vector<std::vector<double>> distances){
	// Construct the square-formed distance matrix
	double nDistances = distances.size();
	double nRows = 1.0/2.0 + std::sqrt(1.0/4.0+2.0*nDistances);
	ID_estimator::nPoints_ = int(nRows);
	
	std::cout << "Number of points: " << ID_estimator::nPoints_ << std::endl;
	
	ID_estimator::distance_matrix_ = Eigen::MatrixXd::Zero(ID_estimator::nPoints_,ID_estimator::nPoints_);
	
	int counter = 0;
	double distance;
	for(int iRow = 0; iRow < ID_estimator::nPoints_; iRow++){
		for(int iCol = iRow+1; iCol < ID_estimator::nPoints_; iCol++){
			distance = distances[counter][0];
			ID_estimator::distance_matrix_(iRow,iCol) = distance;
			ID_estimator::distance_matrix_(iCol,iRow) = distance;
			counter++;
		}
	}
}

Eigen::MatrixXd ID_estimator::get_distance_matrix(){
	// Return the square-formed distance matrix 
	return ID_estimator::distance_matrix_;
}

std::vector<double> ID_estimator::get_2NN_ratios(){
	// Looping through each row (and column) of the distance matrix to find 
	// the 2 nearest neighbor distances of each point.
	std::vector<double> nearest_neighbor_ratios(ID_estimator::nPoints_);
	Eigen::VectorXd current_row;
	
	double first_neighbor, second_neighbor, max_distance;
	
	std::ptrdiff_t index;
	
	for(int iRow = 0; iRow < ID_estimator::nPoints_; iRow++){
		
		current_row = ID_estimator::distance_matrix_.row(iRow);
		max_distance = current_row.maxCoeff()+0.1;
		
		// Set the element (iRow,iRow) in distance matrix to the largest 
		// number to disregard it when searching for neighbors.
		current_row(iRow) = max_distance;
		
		first_neighbor = current_row.minCoeff(&index);
		current_row(index) = max_distance;
		
		second_neighbor = current_row.minCoeff();
		
		// Update the nearest neighbor matrix
		nearest_neighbor_ratios[iRow] = second_neighbor/first_neighbor;		
	}
	
	return nearest_neighbor_ratios;
}

std::vector<Eigen::VectorXd> ID_estimator::compute_ratio_and_empirical_cummulate_pair(std::vector<double> ratios, int nKeepPoints){
	// Sort the 2 nearest neighbor ratios and construct an empirical uniform cummulate
	std::sort(ratios.begin(),ratios.end());
	
	double nTotalPoints = double(ID_estimator::nPoints_);
	
	Eigen::VectorXd ratios_keep(nKeepPoints);
	Eigen::VectorXd empirical_cummulate(nKeepPoints);
	
	for(int i = 0; i < nKeepPoints; i++){
		ratios_keep(i) = ratios[i];
		empirical_cummulate(i) = double(i)/nTotalPoints;
	}
	
	std::vector<Eigen::VectorXd> ratio_emp_cum_pair(2);
	ratio_emp_cum_pair[0] = ratios_keep;
	ratio_emp_cum_pair[1] = empirical_cummulate;
	
	return ratio_emp_cum_pair;
}

std::vector<Eigen::VectorXd> ID_estimator::transform_xy_linear(Eigen::VectorXd ratios, Eigen::VectorXd empirical_cummulate){
	
	int nPoints = ratios.rows();
	
	// Construct matrix A, and b
	Eigen::VectorXd x(nPoints,2);
	Eigen::VectorXd y(nPoints);
	for(int iRow = 0; iRow < nPoints; iRow++){
		x(iRow) = std::log(ratios(iRow));
		y(iRow) = -std::log(1.0-empirical_cummulate(iRow));
	}
	
	std::vector<Eigen::VectorXd> xy(2);
	xy[0] = x;
	xy[1] = y;
	
	return xy;
}

Eigen::VectorXd ID_estimator::linear_regression(Eigen::VectorXd x, Eigen::VectorXd y){
	// Perform linear regression on x and y to find the slope and intercepts (p = [slope, intercept]) by solving
	// Ap=y, where A = [x,1]
	
	int nPoints = x.rows();
	
	// Construct matrix A
	Eigen::MatrixXd A(nPoints,2);
	Eigen::VectorXd b(nPoints);
	for(int iRow = 0; iRow < nPoints; iRow++){
		A(iRow,0) = x(iRow);
		A(iRow,1) = 1;
	}
	
	// Solve system of equations
	Eigen::VectorXd p = A.colPivHouseholderQr().solve(y);
   	std::cout << "Fitted line parameters:\n" << p << std::endl;
   	
   	return p;
}
