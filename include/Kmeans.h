/*
*	K-means clustering
*
*	Annie Westerlund
* 	2018, KTH
*/

#include <vector>
#include <Eigen/Eigen>

class Kmeans{

	public:
	Kmeans(int nComponents, int nIterations);
	void cluster_data(Eigen::MatrixXd x);
	std::vector<Eigen::VectorXd> get_cluster_centers();
	std::vector<int> get_cluster_labels();
	
	private:
	void assign_cluster_labels(Eigen::MatrixXd x);
	void update_centers(Eigen::MatrixXd x);

	int nComponents_;
	int nIterations_;
	std::vector<Eigen::VectorXd> cluster_centers_;
	std::vector<int> cluster_labels_;

};