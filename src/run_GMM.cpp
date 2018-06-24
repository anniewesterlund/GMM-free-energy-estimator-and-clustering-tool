/**********************************
*
* 	Gaussian mixture model free energy estimator main function
* 	Annie Westerlund
*
*	2018, KTH
*
**********************************/

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "ArgParse.cpp"
#include "GMM.cpp"
#include "ID_estimator.cpp"

void read_data(std::string data_file, int nDims, std::vector<std::vector<double>> &data){
	std::cout << "Reading data (may take a while)." << std::endl;
	std::ifstream in_file(data_file, std::ios::in);
	
	//check if file opened correctly
	if (!in_file.is_open()) {
		std::cerr << "Cannot open input file: " << data_file << "\n";
		exit(1);//exit or do additional error checking
	}
	
	double num = 0.0;
	// Store all data in the data matrix
	std::vector<double> tmp_vec(nDims);
	while (in_file >> num) {
		tmp_vec[0] = num;
		for(int i = 1; i < nDims; i++){
			in_file >> num;
			tmp_vec[i] = num;
		}
		data.push_back(tmp_vec);
	}
}

int main(const int argc, const char* argv[]){
	
	std::string data_file("");
	std::string file_label;
	std::vector<std::vector<double>> data;
	double convergence_tol = 1e-4;	
	int maxIterations = 1000;
	int nMinGaussians;
	int nMaxGaussians;
	int nDims;
	double temperature;
	bool extract_clusters;
	bool estimate_intrinsic_dimension;
	
	std::string epilog;
	epilog = "\n \n ##################  GMM free energy estimator 1.2  ################## \n \n - "
		"Estimates free energy and clusters using Gaussian mixture model with cross-validation."
		" The free energy is estimated in [kcal/mol] units. \n - Author: Annie Westerlund 2018 "
		"\n \n - Usage: \n >> GMM_free_energy \n\n [-f] name of input file (string) " 
		"\n [-d] number of dimensions (int). Default = 1. \n [-c] flag for saving cluster indices\n [-fe] file label (string) "
		"\n [-id] flag for estimating intrinsic dimensions. The input file should then contain "
		"a (squareform-compressed) distance matrix. \n [-min_base] min number of Gaussian components (int). Default = 2."
		"\n [-max_base] max number of Gaussian components (int). Default = 12. \n "
		"[-T] temperature (double). Default = 300.0 [K] \n";
	
	ArgParser parser(epilog);
	parser.add_parameter("-f", "file_name", "str");
	parser.add_parameter("-fe", "file_end_name", "str");
	parser.add_parameter("-d", "n_dimensions", "int",int(1));
	parser.add_parameter("-min_base", "n_min_gaussians", "int",int(2));
	parser.add_parameter("-max_base", "n_max_gaussians", "int",int(12));
	parser.add_parameter("-T", "temperature", "double",double(300.0));
	parser.add_parameter("-c", "extract_clusters", "bool");
	parser.add_parameter("-id", "estimate_intrinsic_dimension", "bool");
	
	parser.parse_arguments(argc, argv);
	
	parser.get_value("file_name",data_file);
	parser.get_value("n_dimensions",nDims);
	parser.get_value("extract_clusters",extract_clusters);
	parser.get_value("estimate_intrinsic_dimension",estimate_intrinsic_dimension);
	parser.get_value("file_end_name",file_label);
	parser.get_value("n_min_gaussians",nMinGaussians);
	parser.get_value("n_max_gaussians",nMaxGaussians);
	parser.get_value("temperature",temperature);
	
	if (estimate_intrinsic_dimension){
		nDims = 1;
	}
	
	read_data(data_file, nDims, data);
	
	if (estimate_intrinsic_dimension){
		ID_estimator ID_est;
		nDims = ID_est.estimate_ID(data);
	}
	
	GMM GMM_FE(data, temperature, convergence_tol, maxIterations, file_label);
	GMM_FE.fitDensity(nMinGaussians, nMaxGaussians);
	
	if(extract_clusters){
		GMM_FE.write_cluster_indices();
	}
		
}