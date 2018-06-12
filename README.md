# GMM_free_energy_estimator_Cpp

Implementation of free energy estimator based on Gaussian Mixture models and cross-validation in C++:
https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00346

This requires Eigen (https://eigen.tuxfamily.org/dox/GettingStarted.html) - downloaded and in the path.

Installing estimator:
- mkdir build
- cd build
- cmake ..
- make
- sudo make install

Running free energy estimation (anywhere) on data set "data.dat" with dimensionality 3:
- GMM_free_energy data.dat 3 > free_energy.txt

free_energy.txt contains the free energy corresponding to each point in data.dat.
