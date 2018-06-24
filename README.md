# GMM free energy estimator and clustering tool

Implementation of free energy estimator based on Gaussian Mixture models and cross-validation in C++:
https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00346

Each point is assigned to a cluster based on the estimated density.

This requires Eigen (https://eigen.tuxfamily.org/dox/GettingStarted.html) - downloaded and in the path.

# _Installing tool:_
- mkdir build
- cd build
- cmake ..
- sudo make install

# _Showing (brief) documentation:_

$ GMM_free_energy -h



# _Example of running:_ 
Free energy estimation and clustering on data set "data.dat" with dimensionality 2 and label "sys1":

 $ GMM_free_energy -f data.dat -d 2 -c -fe sys1



_This generates_
1. free_energy_sys1.txt 
2. cluster_indices_sys1.txt 

which contain the free energy and cluster index corresponding to each point in data.dat, respectively. 



Remove option -c if you don't want the cluster_indices_label.txt.
