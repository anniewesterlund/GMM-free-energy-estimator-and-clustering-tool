# GMM_free_energy_estimator_Cpp

Implementation of free energy estimator based on Gaussian Mixture models and cross-validation in C++:
https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00346

This requires Eigen (https://eigen.tuxfamily.org/dox/GettingStarted.html) - downloaded and in the path.

Installing estimator:
- mkdir build
- cd build
- cmake ..
- sudo make install

Example of running free energy estimation and clustering on data set "data.dat" with dimensionality 2 and label "sys1":
 $ GMM_free_energy -f data.dat -d 2 -c -fe sys1

This generates free_energy_kT_sys1.txt and cluster_indices_sys1.txt which contain the free energy and cluster index corresponding to each point in data.dat, respectively. 

Remove option -c if you don't want the cluster_indices_label.txt.
