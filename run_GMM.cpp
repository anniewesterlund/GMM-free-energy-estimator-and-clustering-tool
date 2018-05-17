/**********************************
*
* 	Cellular automata main function
* 	Annie Westerlund
*
*	2017 - SI3080
*
**********************************/

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

#include "GMM.cpp"

using namespace std;

int main(const int argc, const char* argv[]){
	
	string data_file("");
	vector<vector<double>> data;
	double convergence_tol = 1e-4;	
	int maxIterations = 1000;
	int nMinGaussians = 2;
	int nMaxGaussians = 12;
	int nDims = 1;
	
	for(int i = 0; i < argc; i++){
		if(i==1){data_file = argv[i];}
		if(i==2){nDims = atoi(argv[i]);}
	}
	
	if(argc > 1){
		std::ifstream in_file(data_file, std::ios::in);
		
		//check if file opened correctly
		if (!in_file.is_open()) {
			std::cerr << "Cannot open input file!\n";
			exit(1);//exit or do additional error checking
		}
		
		double num = 0.0;
		// Store all data in the data matrix
		vector<double> tmp_vec(nDims);
		while (in_file >> num) {
			tmp_vec[0] = num;
			for(int i = 1; i < nDims; i++){
				in_file >> num;
				tmp_vec[i] = num;
			}
			data.push_back(tmp_vec);
		}
	}
	
	GMM GMM_FE(data, convergence_tol, maxIterations);
	GMM_FE.fitDensity(nMinGaussians, nMaxGaussians);
	
	clog << "Free energy estimated" << endl;
	
}