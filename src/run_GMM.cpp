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

void read_data(std::string data_file, int nDims, std::vector<std::vector<double>> &data){
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
	int nMinGaussians = 2;
	int nMaxGaussians = 12;
	int nDims = 1;
	bool extract_clusters = false;
	bool showed_docs = false;
	
	std::string epilog;
	epilog = "\n \n ##################  GMM free energy estimator 1.2  ################## \n \n - "
		"Estimates free energy and clusters using Gaussian mixture model with cross-validation."
		" The free energy is estimated in [kT] units. \n - Author: Annie Westerlund 2018 "
		"\n \n - Usage: \n >> GMM_free_energy -f file_name(string) " 
		"-d number_of_dimensions(int) -c (flag for saving cluster indices) -fe file_label(string)\n";
	
	ArgParser parser(epilog);
	parser.add_parameter("-f", "file_name", "str");
	parser.add_parameter("-fe", "file_end_name", "str");
	parser.add_parameter("-d", "n_dimensions", "int",1);
	parser.add_parameter("-c", "extract_clusters", "bool");
	
	parser.parse_arguments(argc, argv);
	
	parser.get_value("file_name",data_file);
	parser.get_value("n_dimensions",nDims);
	parser.get_value("extract_clusters",extract_clusters);
	parser.get_value("file_end_name",file_label);
	
	read_data(data_file, nDims, data);
	
	GMM GMM_FE(data, convergence_tol, maxIterations, file_label);
	GMM_FE.fitDensity(nMinGaussians, nMaxGaussians);
	
	if(extract_clusters){
		GMM_FE.write_cluster_indices();
	}
		
}